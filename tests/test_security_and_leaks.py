"""test_security_and_leaks.py — Tests for sealed store isolation, repository leak guard (Rule 4),
rater screen hygiene, and minimum font size >= 12px."""
from __future__ import annotations

import hashlib
import json
import re
import struct
import threading
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import pytest

from blindpanel.leak_guard import main as leak_guard_main
from blindpanel.leak_guard import scan_repository
from blindpanel.server import BlindPanelSession, create_http_server

FORBIDDEN_RATER_TOKENS = [
    "ARM_1",
    "ARM_2",
    "ARM_3",
    "ARM_4",
    "repeat",
    "Repeat",
    "warm-up",
    "warmup",
    "Warm-up",
    "expected",
    "Expected",
    "half",
    "Half",
    "State 4A",
    "State 4B",
    "Return to Pre-Flight",
]


def _start_test_server(tmp_path: Path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    arms = ["ARM_1", "ARM_2", "ARM_3"]
    arm_specs = {}
    for arm in arms:
        p = clips_dir / f"{arm}_M1-s1111.mp4"
        payload = f"MP4DATA_{arm}".encode("utf-8")
        p.write_bytes(payload)
        arm_specs[arm] = {"path": str(p), "sha256": hashlib.sha256(payload).hexdigest()}

    cfg = {
        "schema_version": "blindpanel-config-v1",
        "study_id": "leak-check-study",
        "blocks": [
            {
                "block_id": "block1",
                "title": "Main Block 1",
                "method": "bws_3",
                "arms": arms,
                "items": [
                    {
                        "cell_id": "M1-s1111",
                        "prompt_id": "M1",
                        "prompt_type": "manipulation",
                        "prompt": "A single robot arm in a warehouse grasps a cardboard box on a moving conveyor belt and lifts it clear of the belt. Static camera, overhead fluorescent light, realistic.",
                        "rater_cue": "the box and the gripper through the clip",
                        "is_warmup": True,
                        "ask_q2": True,
                        "arms": arm_specs,
                    },
                    {
                        "cell_id": "M1-s2026-rep2",
                        "repeat_of_cell_id": "M1-s1111",
                        "prompt_id": "M2",
                        "prompt_type": "manipulation",
                        "prompt": "A two-armed robot at a kitchen counter grasps a ceramic bowl with one gripper and lifts it off the counter. Static camera, soft daylight, realistic.",
                        "rater_cue": "the ceramic bowl and the gripper through the clip",
                        "is_repeat": True,
                        "ask_q2": True,
                        "arms": arm_specs,
                    },
                ],
            }
        ],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    session = BlindPanelSession(
        config_path=str(cfg_path),
        sealed_map_path=str(tmp_path / "sealed_store" / "sealed_map_session.bin"),
        answers_path=str(tmp_path / "answers.jsonl"),
    )
    httpd = create_http_server(session, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://127.0.0.1:{port}", session


def test_sealed_map_and_manifest_never_reach_browser(tmp_path: Path) -> None:
    """Verify every HTTP path attempting to fetch sealed_map, sealed_manifest, or .bin returns 404,
    and /api/state contains zero arm names, file paths, or study internals, serving clips only under salted names."""
    httpd, base_url, _session = _start_test_server(tmp_path)
    try:
        for forbidden_path in [
            "/sealed_map.json",
            "/sealed_map_session.bin",
            "/sealed_manifest.json",
            "/sealed_store/sealed_map_session.bin",
            "/test.sealed.jsonl",
            "/anything.bin",
        ]:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(base_url + forbidden_path)
            assert exc_info.value.code == 404

        with urllib.request.urlopen(base_url + "/api/state") as resp:
            raw_state = resp.read().decode("utf-8")
        state_obj = json.loads(raw_state)

        for token in FORBIDDEN_RATER_TOKENS:
            assert token not in raw_state, f"Forbidden token '{token}' leaked in /api/state"

        for forbidden_key in [
            "letter_to_arm",
            "arm_to_letter",
            "sealed_manifest",
            "secret_seed",
            "is_warmup",
            "is_repeat",
            "repeat_of_cell_id",
            "sha256",
        ]:
            assert forbidden_key not in state_obj
            for clip_entry in state_obj["clips"]:
                assert forbidden_key not in clip_entry
                assert clip_entry["url"].startswith("/clip/clip_")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_repository_leak_guard_passes_on_tracked_files() -> None:
    """Rule 4: Verify `blindpanel.leak_guard` passes with 0 violations on all tracked files in the repo."""
    repo_root = Path(__file__).resolve().parent.parent
    violations = scan_repository(repo_root)
    assert violations == [], f"Leak guard failed on repo: {violations}"
    assert leak_guard_main([str(repo_root)]) == 0


def test_repository_leak_guard_catches_all_five_leak_classes_in_scratch_dir(tmp_path: Path) -> None:
    """Rule 4 proof: Verify `blindpanel.leak_guard` fails on a scratch copy outside the repo containing:
    1. A per-item slot or arm order (`"slot" + "_arms"`).
    2. An arm label (`"BF" + "16"`) within N chars of a sha256 or `.mp4` clip path.
    3. A seed or salt key (`"rng_seed"` or voided `"2026" + "1030"`).
    4. An absolute local path (Windows drive or user-home form).
    5. An image under `docs/screenshots/` without the mock-clip generator metadata marker and manifest entry.
    """
    scratch_repo = tmp_path / "unsealed_scratch_repo"
    (scratch_repo / "configs").mkdir(parents=True, exist_ok=True)
    (scratch_repo / "docs" / "screenshots").mkdir(parents=True, exist_ok=True)

    # Construct the prohibited tokens dynamically so this test file itself stays 100% clean
    tok_slot_key = "slot" + "_arms"
    tok_arm_bf = "BF" + "16"
    tok_arm_q1 = "Q" + "1"
    tok_arm_q3 = "Q" + "3"
    tok_cell_prefix = tok_arm_bf + "_cell_M1-s1111.mp4"
    tok_void_seed = int("2026" + "1030")
    tok_abs_win = "E" + ":/AI/rnd-local/clips/" + tok_cell_prefix
    fake_sha = "26738e93925931849239a9ee0aa53f88a98c89c7b7046b946fea21df4b259a5b"

    bad_config = {
        "schema_version": "blindpanel-config-v1",
        "study_id": "leaked-study",
        "rng_seed": tok_void_seed,
        "blocks": [
            {
                "block_id": "b1",
                "method": "bws_3",
                "arms": [tok_arm_bf, tok_arm_q1, tok_arm_q3],
                "items": [
                    {
                        "cell_id": "M1-s1111",
                        tok_slot_key: [tok_arm_bf, tok_arm_q1, tok_arm_q3],
                        "arms": {
                            tok_arm_bf: {"path": tok_abs_win, "sha256": fake_sha},
                        },
                    }
                ],
            }
        ],
    }
    (scratch_repo / "configs" / "bad_config.json").write_text(json.dumps(bad_config, indent=2), encoding="utf-8")

    # Write an unmarked PNG image under docs/screenshots/ simulating a real-clip screenshot
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    c_ihdr = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
    raw_px = zlib.compress(b"\x00\x40\x40\x40")
    c_idat = struct.pack(">I", len(raw_px)) + b"IDAT" + raw_px + struct.pack(">I", zlib.crc32(b"IDAT" + raw_px) & 0xFFFFFFFF)
    c_iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
    unmarked_png = sig + c_ihdr + c_idat + c_iend
    (scratch_repo / "docs" / "screenshots" / "block1-real-clips-3up-1440x900.png").write_bytes(unmarked_png)

    violations = scan_repository(scratch_repo)
    joined = "\n".join(violations)

    assert any("per-item slot or arm order" in v for v in violations), joined
    assert any("arm label" in v and "within" in v for v in violations), joined
    assert any("seed or salt" in v or "voided study seed" in v for v in violations), joined
    assert any("absolute local path" in v for v in violations), joined
    assert any("missing PNG metadata marker" in v for v in violations), joined
    assert leak_guard_main([str(scratch_repo)]) == 1


def test_css_and_static_minimum_font_size_12px() -> None:
    """Verify every font-size declaration in styles.css and static files is >= 12px."""
    repo_root = Path(__file__).resolve().parent.parent
    css_path = repo_root / "blindpanel" / "static" / "styles.css"
    js_path = repo_root / "blindpanel" / "static" / "app.js"
    html_path = repo_root / "blindpanel" / "static" / "index.html"

    for path in (css_path, js_path, html_path):
        content = path.read_text(encoding="utf-8")
        matches = re.findall(r"font-size\s*:\s*([0-9.]+)\s*px", content, flags=re.IGNORECASE)
        for m in matches:
            val = float(m)
            assert val >= 12.0, f"Font size {val}px (< 12px) found in {path.name}"


def test_no_external_urls_in_static_assets() -> None:
    """Verify zero external http:// or https:// network calls exist in index.html, styles.css, or app.js."""
    repo_root = Path(__file__).resolve().parent.parent
    static_dir = repo_root / "blindpanel" / "static"
    for fname in ("index.html", "styles.css", "app.js"):
        text = (static_dir / fname).read_text(encoding="utf-8")
        cleaned = text.replace("http://www.w3.org/2000/svg", "")
        assert "http://" not in cleaned, f"External http:// URL found in {fname}"
        assert "https://" not in cleaned, f"External https:// URL found in {fname}"
