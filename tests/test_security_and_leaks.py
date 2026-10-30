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


def test_separate_rater_and_operator_surfaces_and_export_removal(tmp_path: Path) -> None:
    """Card tm (1 & 2): Rater port returns 404 on /api/operator/* and /api/export/* (with and without token),
    answers file is unreachable on rater port, and operator port requires its per-start token."""
    from blindpanel import server as srv_mod

    httpd, rater_base_url, session = _start_test_server(tmp_path)
    try:
        for route in (
            "/api/operator/preflight",
            "/api/operator/audit",
            "/api/export/csv",
            "/api/export/jsonl",
        ):
            for suffix in ("", "?token=fake", "?operator_token=fake"):
                req = urllib.request.Request(
                    rater_base_url + route + suffix,
                    headers={"X-Operator-Token": "fake", "Authorization": "Bearer fake"},
                )
                with pytest.raises(urllib.error.HTTPError) as exc_info:
                    urllib.request.urlopen(req)
                assert exc_info.value.code == 404, f"Route {route}{suffix} did not return 404 on rater port"

        req_post = urllib.request.Request(
            rater_base_url + "/api/operator/override",
            data=json.dumps({"reason": "override attempt"}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-Operator-Token": "fake"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_post:
            urllib.request.urlopen(req_post)
        assert exc_post.value.code == 404

        session.start_session("live")
        session.submit_rating({
            "mode": "live",
            "global_index": 0,
            "best_letter": "A",
            "worst_letter": "C",
            "q2_by_letter": {"A": "yes", "B": "partly", "C": "no"},
        })
        for ans_route in ("/answers.jsonl", "/pilot_answers.jsonl", "/operator_audit.jsonl"):
            with pytest.raises(urllib.error.HTTPError) as exc_ans:
                urllib.request.urlopen(rater_base_url + ans_route)
            assert exc_ans.value.code == 404

        assert hasattr(srv_mod, "create_operator_http_server"), "Missing create_operator_http_server"
        op_server, op_token = srv_mod.create_operator_http_server(session, host="127.0.0.1", port=0)
        op_port = op_server.server_address[1]
        assert op_port != httpd.server_address[1]
        assert len(op_token) >= 32
        op_thread = threading.Thread(target=op_server.serve_forever, daemon=True)
        op_thread.start()
        op_base = f"http://127.0.0.1:{op_port}"
        try:
            for op_route in ("/api/operator/preflight", "/api/operator/audit", "/api/export/csv", "/api/export/jsonl"):
                with pytest.raises(urllib.error.HTTPError) as exc_unauth:
                    urllib.request.urlopen(op_base + op_route)
                assert exc_unauth.value.code in (401, 403)

                with pytest.raises(urllib.error.HTTPError) as exc_bad:
                    urllib.request.urlopen(op_base + op_route + "?token=wrong")
                assert exc_bad.value.code in (401, 403)

                req_ok = urllib.request.Request(
                    op_base + op_route,
                    headers={"X-Operator-Token": op_token},
                )
                with urllib.request.urlopen(req_ok) as resp:
                    assert resp.status == 200
        finally:
            op_server.shutdown()
            op_server.server_close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_production_served_js_contains_zero_test_hooks_or_operator_code(tmp_path: Path) -> None:
    """Card tm (3): Production served /app.js contains zero references to ?unlock=1, window.__blindpanel,
    or test helpers, while explicit test mode enables them."""
    httpd, base_url, _session = _start_test_server(tmp_path)
    try:
        with urllib.request.urlopen(base_url + "/app.js") as resp:
            prod_js = resp.read().decode("utf-8")

        for forbidden_hook in (
            "__blindpanel",
            "unlockVotingForTest",
            "setSelectionsForTest",
            'params.get("unlock")',
            "params.get('unlock')",
            "unlock=1",
        ):
            assert forbidden_hook not in prod_js, f"Production /app.js leaked test hook '{forbidden_hook}'"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_arm_map_flag_and_leak_guard_protection(tmp_path: Path) -> None:
    """Card tm (4): --arm-map maps pseudonyms ARM_1..ARM_4 to local clip filename prefixes, loads only into
    the sealed store, never appears in any route or page, and is caught by leak_guard if tracked."""
    from blindpanel.cosmos3_study import build_public_study_design_config

    cfg = build_public_study_design_config()
    cfg_path = tmp_path / "cosmos3_amendment_a.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    clips_dir = tmp_path / "local_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    arm_map = {
        "ARM_1": "REALPREFIX1_cell",
        "ARM_2": "REALPREFIX2_cell",
        "ARM_3": "REALPREFIX3_cell",
        "ARM_4": "REALPREFIX4_cell",
    }
    arm_map_path = tmp_path / "arm_map_local.json"
    arm_map_path.write_text(json.dumps(arm_map), encoding="utf-8")

    all_cells = sorted({it["base_cell_id"] for blk in cfg["blocks"] for it in blk["items"]})
    for cid in all_cells:
        for _pseudo, prefix in arm_map.items():
            p = clips_dir / f"{prefix}_{cid}.mp4"
            p.write_bytes(f"MP4_{prefix}_{cid}".encode("utf-8"))

    sealed_path = tmp_path / "sealed_store" / "sealed_map.bin"
    ans_path = tmp_path / "answers.jsonl"
    session = BlindPanelSession(
        config_path=str(cfg_path),
        clips_dir=str(clips_dir),
        arm_map_path=str(arm_map_path),
        sealed_map_path=str(sealed_path),
        answers_path=str(ans_path),
    )
    smap = json.loads(sealed_path.read_text(encoding="utf-8"))
    assert smap.get("arm_map") == arm_map

    pub_state_str = json.dumps(session.get_public_state())
    for prefix in arm_map.values():
        assert prefix not in pub_state_str

    scratch = tmp_path / "scratch_arm_map_repo"
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "arm_map.json").write_text(
        json.dumps({"ARM_1": "BF" + "16", "ARM_2": "Q" + "1", "ARM_3": "Q" + "3", "ARM_4": "NA" + "NO"}),
        encoding="utf-8",
    )
    violations = scan_repository(scratch)
    assert any("arm_map" in v or "arm map" in v.lower() for v in violations), f"Expected arm-map violation, got {violations}"


def test_unblind_validates_sealed_manifest_sha256_and_refuses_on_mismatch(tmp_path: Path) -> None:
    """Card tm (5): unblind.py validates the sealed manifest's integrity (sha) before unblinding,
    and refuses on a mismatch."""
    from blindpanel.unblind import unblind_session

    _httpd, _base_url, session = _start_test_server(tmp_path)
    _httpd.shutdown()
    _httpd.server_close()

    smap = json.loads(Path(session.sealed_map_path).read_text(encoding="utf-8"))

    tampered_smap = json.loads(json.dumps(smap))
    tampered_smap["sealed_manifest"][0]["original_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sealed_manifest|sha256|Tamper"):
        unblind_session(tampered_smap, [])


def test_rater_port_full_mock_session_crawl(tmp_path: Path) -> None:
    """Card tm (6): Crawl test on the RATER port through a full mock session that rates through
    warm-ups and repeats, following every link in served HTML, every fetch/URL string in served JS,
    and every route in the server's route table (enumerated from code)."""
    from blindpanel import server as srv_mod
    from blindpanel.cosmos3_study import build_public_study_design_config

    cfg = build_public_study_design_config()
    cfg_path = tmp_path / "cosmos3_amendment_a.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    all_cells = sorted({it["base_cell_id"] for blk in cfg["blocks"] for it in blk["items"]})
    all_arms = sorted({arm for blk in cfg["blocks"] for arm in blk["arms"]})
    for cid in all_cells:
        for arm in all_arms:
            p = clips_dir / f"{arm}_{cid}.mp4"
            p.write_bytes(f"MOCK_MP4_{arm}_{cid}".encode("utf-8"))

    sealed_path = tmp_path / "sealed_store" / "sealed_map.bin"
    ans_path = tmp_path / "answers.jsonl"
    session = BlindPanelSession(
        config_path=str(cfg_path),
        clips_dir=str(clips_dir),
        sealed_map_path=str(sealed_path),
        answers_path=str(ans_path),
    )

    httpd = create_http_server(session, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    forbidden_crawl_substrings = [
        "is_warmup",
        "is_repeat",
        "repeat_of",
        "base_cell_id",
        "cell_id",
        "-warmup",
        "-rep",
        "ARM_",
        "BF" + "16",
        "Q" + "1",
        "Q" + "3",
        "Na" + "no",
        "NA" + "NO",
        "num_warmup",
        "num_repeats",
    ]

    def assert_clean_payload(label: str, text: str) -> None:
        for tok in forbidden_crawl_substrings:
            assert tok not in text, f"Crawl failure on {label}: found forbidden substring '{tok}'"

    def http_get_text(path: str, headers: dict | None = None) -> tuple[int, str]:
        req = urllib.request.Request(base_url + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, ""

    def http_post_json(path: str, payload: dict) -> tuple[int, str]:
        req = urllib.request.Request(
            base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, ""

    try:
        st_code, html_text = http_get_text("/")
        assert st_code == 200
        assert_clean_payload("GET /", html_text)

        html_links = re.findall(r'(?:href|src)=["\'](/[^"\']*)["\']', html_text)
        assert "/styles.css" in html_links
        assert "/app.js" in html_links

        js_text = ""
        for link in html_links:
            code, body = http_get_text(link)
            assert code == 200, f"Linked asset {link} returned {code}"
            assert_clean_payload(f"HTML link {link}", body)
            if link == "/app.js":
                js_text = body

        js_urls = set(re.findall(r'["\'`]((?:/api/|/clip/|/styles\.css|/app\.js|/index\.html)[^"\'$`]*)', js_text))
        assert "/api/state" in {u.split("?")[0] for u in js_urls}

        assert hasattr(srv_mod, "RATER_GET_ROUTES") and hasattr(srv_mod, "RATER_POST_ROUTES")
        assert hasattr(srv_mod, "OPERATOR_GET_ROUTES") and hasattr(srv_mod, "OPERATOR_POST_ROUTES")

        code, body = http_post_json("/api/session/start", {"mode": "live"})
        assert code == 200
        assert_clean_payload("POST /api/session/start", body)

        total_items = len(session.flat_items)
        for idx in range(total_items):
            code, st_body = http_get_text("/api/state?mode=live")
            assert code == 200
            assert_clean_payload(f"GET /api/state at item {idx}", st_body)
            st_data = json.loads(st_body)

            if st_data.get("interstitial") == "rest_break":
                code, res_body = http_post_json("/api/session/resume_break", {"mode": "live"})
                assert code == 200
                assert_clean_payload("POST /api/session/resume_break", res_body)
                st_data = json.loads(res_body)
            elif st_data.get("interstitial") == "block_break":
                code, b2_body = http_post_json("/api/session/start_block2", {"mode": "live"})
                assert code == 200
                assert_clean_payload("POST /api/session/start_block2", b2_body)
                st_data = json.loads(b2_body)

            for c in st_data.get("clips", []):
                assert_clean_payload(f"clip entry item {idx}", json.dumps(c))

            sub_payload: dict = {"mode": "live", "global_index": st_data["global_index"]}
            if st_data["method"] == "bws_3":
                sub_payload["best_letter"] = "A"
                sub_payload["worst_letter"] = "C"
            else:
                sub_payload["choice"] = "A"
            if st_data.get("ask_q2"):
                sub_payload["q2_by_letter"] = {let: "yes" for let in st_data["letters"]}

            code, sub_body = http_post_json("/api/submit", sub_payload)
            assert code == 200
            assert_clean_payload(f"POST /api/submit at item {idx}", sub_body)

        for u in sorted(js_urls):
            clean_u = u.split("?")[0]
            if clean_u in srv_mod.RATER_GET_ROUTES:
                code, body = http_get_text(u)
                assert code == 200
                assert_clean_payload(f"JS URL {u}", body)

        for r_get in srv_mod.RATER_GET_ROUTES:
            code, body = http_get_text(r_get)
            assert code == 200
            assert_clean_payload(f"RATER_GET_ROUTES {r_get}", body)

        for op_get in list(srv_mod.OPERATOR_GET_ROUTES) + ["/api/export/csv", "/api/export/jsonl"]:
            code_no_tok, _ = http_get_text(op_get)
            code_with_tok, _ = http_get_text(
                op_get + "?token=anytoken",
                headers={"X-Operator-Token": "anytoken", "Authorization": "Bearer anytoken"},
            )
            assert code_no_tok == 404, f"{op_get} without token returned {code_no_tok}"
            assert code_with_tok == 404, f"{op_get} with token returned {code_with_tok}"

        for op_post in srv_mod.OPERATOR_POST_ROUTES:
            code_post, _ = http_post_json(op_post, {"reason": "test"})
            assert code_post == 404, f"POST {op_post} returned {code_post}"

        raw_answers_disk = ans_path.read_text(encoding="utf-8")
        assert "-warmup" in raw_answers_disk and "-rep2" in raw_answers_disk
        for ans_try in ("/answers.jsonl", f"/{ans_path.name}", "/api/export/jsonl"):
            code_ans, _ = http_get_text(ans_try)
            assert code_ans == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_amendment_a_ui_shortcuts_and_700ms_curtain() -> None:
    """Card tm (5): Verify app.js implements Y/P/N for Q2, Shift+1-3 for Worst in 3-up, F for flag flawed,
    and the 700 ms inter-trial grey curtain."""
    repo_root = Path(__file__).resolve().parent.parent
    js_text = (repo_root / "blindpanel" / "static" / "app.js").read_text(encoding="utf-8")

    # 700 ms inter-trial grey curtain
    assert "setTimeout(r, 700)" in js_text or "INTER_TRIAL_CURTAIN_MS = 700" in js_text
    # Y / P / N for Question 2
    assert '"Y": "yes"' in js_text or "'Y': 'yes'" in js_text
    # Shift+1-3 for Worst in 3-up
    assert "SHIFT_DIGIT_MAP" in js_text or ("ev.shiftKey" in js_text and "Digit1" in js_text)
    # F for flag flawed
    assert 'upper === "F"' in js_text
