"""test_security_and_leaks.py — Tests for sealed map isolation, rater screen hygiene (leak scan), and minimum font size >= 12px."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from blindpanel.server import BlindPanelSession, create_http_server

FORBIDDEN_RATER_TOKENS = [
    "BF16",
    "bf16",
    "Q1",
    "q1",
    "Q3",
    "q3",
    "NANO",
    "Nano",
    "nano",
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
    arms = ["BF16", "Q1", "Q3"]
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
        sealed_map_path=str(tmp_path / "sealed_map_20261030.bin"),
        answers_path=str(tmp_path / "answers.jsonl"),
    )
    httpd = create_http_server(session, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://127.0.0.1:{port}", session


def test_sealed_map_never_reaches_browser(tmp_path: Path) -> None:
    """Verify every HTTP path attempting to fetch sealed_map or .bin returns 404,
    and /api/state contains zero arm names, file paths, or study internals."""
    httpd, base_url, session = _start_test_server(tmp_path)
    try:
        for forbidden_path in [
            "/sealed_map.json",
            "/sealed_map_20261030.bin",
            "/sealed_store/sealed_map_20261030.bin",
            "/test.sealed.jsonl",
            "/anything.bin",
        ]:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(base_url + forbidden_path)
            assert exc_info.value.code == 404

        with urllib.request.urlopen(base_url + "/api/state") as resp:
            raw_state = resp.read().decode("utf-8")
        state_obj = json.loads(raw_state)

        # Verify no forbidden keys or tokens appear in /api/state JSON
        for token in FORBIDDEN_RATER_TOKENS:
            assert token not in raw_state, f"Forbidden token '{token}' leaked in /api/state"

        for forbidden_key in ["letter_to_arm", "arm_to_letter", "is_warmup", "is_repeat", "repeat_of_cell_id", "sha256", "slot_arms"]:
            assert forbidden_key not in state_obj
            for clip_entry in state_obj["clips"]:
                assert forbidden_key not in clip_entry
    finally:
        httpd.shutdown()
        httpd.server_close()


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
        # Exclude SVG namespace http://www.w3.org/2000/svg if any
        cleaned = text.replace("http://www.w3.org/2000/svg", "")
        assert "http://" not in cleaned, f"External http:// URL found in {fname}"
        assert "https://" not in cleaned, f"External https:// URL found in {fname}"
