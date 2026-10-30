"""test_unit.py — Unit tests for blindpanel schema, session engine, counterbalancing, honesty checks, and unblinding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pytest

from blindpanel.cosmos3_study import (
    B200_144_SHA256,
    build_cosmos3_amendment_a_config,
)
from blindpanel.schema import (
    AMENDMENT_A_Q2_PROMPT,
    validate_and_normalize_config,
    verify_bws3_counterbalancing,
    verify_no_adjacent_prompts,
)
from blindpanel.server import BlindPanelSession
from blindpanel.unblind import load_jsonl_safe, unblind_session
from blindpanel.unblind import main as unblind_main


def _make_dummy_clips(tmp_path: Path, arms: List[str], cells: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Create tiny deterministic MP4-like files on disk and return arm->path/sha256 mapping per cell."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for cid in cells:
        out[cid] = {}
        for arm in arms:
            p = clips_dir / f"{arm}_{cid}.mp4"
            payload = f"FAKE_MP4_{arm}_{cid}".encode("utf-8")
            p.write_bytes(payload)
            out[cid][arm] = {
                "path": str(p),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    return out


def test_cosmos3_amendment_a_counterbalancing_and_schema() -> None:
    """Verify the Cosmos 3 Amendment A study config validates cleanly:
    - Block 1 has 43 items (3 warm-up + 36 core + 4 repeats), method bws_3, arms BF16/Q1/Q3
    - All 6 permutations of (BF16, Q1, Q3) appear 6 times each (2 per prompt type)
    - Every prompt's 3 seeds form a Latin square over slots A, B, C
    - No adjacent items share the same prompt_id
    - Question 2 is enabled on M1-M4 and disabled on L1-L4 and S1-S4
    - Block 2 has 42 items (2 warm-up + 36 core + 4 repeats), method 2afc, arms BF16/NANO
    """
    cfg = build_cosmos3_amendment_a_config(clips_dir="/nonexistent/clips", verify_files_exist=False)
    norm = validate_and_normalize_config(cfg)

    assert norm["schema_version"] == "blindpanel-config-v1"
    assert len(norm["blocks"]) == 2

    b1 = norm["blocks"][0]
    assert b1["method"] == "bws_3"
    assert b1["arms"] == ["BF16", "Q1", "Q3"]
    assert b1["num_items"] == 43
    assert b1["num_warmup"] == 3
    assert b1["num_recorded"] == 40
    assert b1["num_repeats"] == 4
    assert b1["break_after_item"] == 23

    cb_stats = verify_bws3_counterbalancing(b1["items"], arms=b1["arms"], block_id=b1["block_id"])
    assert cb_stats["core_triples"] == 36
    assert cb_stats["repeat_triples"] == 4
    assert all(v == 6 for v in cb_stats["permutation_counts"].values())

    for item in b1["items"]:
        if item["prompt_type"] == "manipulation":
            assert item["ask_q2"] is True
            assert item["question2_spec"]["prompt"] == AMENDMENT_A_Q2_PROMPT
            labels = [o["label"] for o in item["question2_spec"]["options"]]
            assert labels == [
                "Yes (completes it)",
                "Partly (clearly under way)",
                "No (not achieved)",
            ]
        else:
            assert item["ask_q2"] is False

    b2 = norm["blocks"][1]
    assert b2["method"] == "2afc"
    assert b2["optional"] is True
    assert b2["arms"] == ["BF16", "NANO"]
    assert b2["num_items"] == 42
    assert b2["num_warmup"] == 2
    assert b2["num_recorded"] == 40
    assert b2["num_repeats"] == 4
    assert len(B200_144_SHA256) == 144


def test_schema_rejects_invalid_configs() -> None:
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        validate_and_normalize_config({"schema_version": "0.0", "blocks": []})

    with pytest.raises(ValueError, match="unsupported method"):
        validate_and_normalize_config({
            "schema_version": "blindpanel-config-v1",
            "study_id": "bad",
            "blocks": [{"block_id": "b1", "method": "invalid_method", "arms": ["A", "B"], "items": []}],
        })

    with pytest.raises(ValueError, match="requires exactly 3 arms"):
        validate_and_normalize_config({
            "schema_version": "blindpanel-config-v1",
            "study_id": "bad",
            "blocks": [{"block_id": "b1", "method": "bws_3", "arms": ["A", "B"], "items": []}],
        })

    with pytest.raises(ValueError, match="consecutive duplicate prompt_id"):
        verify_no_adjacent_prompts([{"prompt_id": "M1"}, {"prompt_id": "M1"}], block_id="b1")


def test_refused_at_load_on_sha256_mismatch_and_missing_file(tmp_path: Path) -> None:
    arms = ["BF16", "Q1", "Q3"]
    clips = _make_dummy_clips(tmp_path, arms, ["M1-s1111"])
    # Corrupt the expected sha256
    clips["M1-s1111"]["BF16"]["sha256"] = "0" * 64

    cfg = {
        "schema_version": "blindpanel-config-v1",
        "study_id": "sha-test",
        "blocks": [
            {
                "block_id": "b1",
                "method": "bws_3",
                "arms": arms,
                "items": [{"cell_id": "M1-s1111", "prompt_id": "M1", "prompt_type": "scene", "arms": clips["M1-s1111"]}],
            }
        ],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="REFUSED AT LOAD: sha256 mismatch"):
        BlindPanelSession(
            config_path=str(cfg_path),
            sealed_map_path=str(tmp_path / "sealed_map.bin"),
            answers_path=str(tmp_path / "answers.jsonl"),
        )

    # Missing file check
    clips["M1-s1111"]["BF16"]["path"] = str(tmp_path / "missing.mp4")
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="REFUSED AT LOAD: missing clip file"):
        BlindPanelSession(
            config_path=str(cfg_path),
            sealed_map_path=str(tmp_path / "sealed_map.bin"),
            answers_path=str(tmp_path / "answers.jsonl"),
        )


def test_preflight_gate_and_logged_override_and_pilot_isolation(tmp_path: Path) -> None:
    arms = ["BF16", "Q1", "Q3"]
    clips = _make_dummy_clips(tmp_path, arms, ["M1-s1111"])
    cfg = {
        "schema_version": "blindpanel-config-v1",
        "study_id": "preflight-study",
        "clip_spec": {"width": 832, "height": 480, "frames": 61, "fps": "24.0000", "seconds": 2.54},
        "blocks": [
            {
                "block_id": "b1",
                "method": "bws_3",
                "arms": arms,
                "items": [
                    {
                        "cell_id": "M1-s1111",
                        "prompt_id": "M1",
                        "prompt_type": "manipulation",
                        "prompt": "Robot arm grasps box and places on pallet.",
                        "required_action": "4-step grasp and palletize",
                        "estimated_seconds": "4.0–5.0 s",
                        "estimated_max_seconds": 4.5,
                        "ask_q2": True,
                        "arms": clips["M1-s1111"],
                    }
                ],
            }
        ],
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    session = BlindPanelSession(
        config_path=str(cfg_path),
        sealed_map_path=str(tmp_path / "sealed_map.bin"),
        answers_path=str(tmp_path / "answers.jsonl"),
        pilot_answers_path=str(tmp_path / "pilot_answers.jsonl"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )

    pf = session.get_preflight_status()
    assert pf["flagged_count"] == 1
    assert pf["live_start_blocked"] is True

    # Live start must be blocked before override
    with pytest.raises(ValueError, match="blocked by Operator Pre-Flight"):
        session.start_session(mode="live")

    # Pilot mode works even while live is blocked and writes ONLY to pilot_answers.jsonl
    session.start_session(mode="pilot")
    session.submit_rating({
        "mode": "pilot",
        "global_index": 0,
        "best_letter": "A",
        "worst_letter": "C",
        "q2_by_letter": {"A": "yes", "B": "partly", "C": "no"},
        "notes": "pilot pass",
    })
    assert not (tmp_path / "answers.jsonl").exists()
    assert (tmp_path / "pilot_answers.jsonl").exists()

    # Record logged operator override
    with pytest.raises(ValueError, match="at least 5 characters"):
        session.record_preflight_override("abc")

    pf_after = session.record_preflight_override("Approved by Publisher after M1 prompt revision")
    assert pf_after["live_start_blocked"] is False
    assert pf_after["override_recorded"] is True

    # Now live session can start and submit
    session.start_session(mode="live")
    st = session.submit_rating({
        "mode": "live",
        "global_index": 0,
        "best_letter": "B",
        "worst_letter": "A",
        "q2_by_letter": {"A": "no", "B": "yes", "C": "partly"},
        "notes": "live rating",
    })
    assert st["interstitial"] == "session_complete"
    audit = session.get_operator_audit()
    assert audit["live_session"]["recorded_completed"] == 1
    assert audit["pilot_session"]["completed_items"] == 1


def test_all_six_methods_and_unblinding_and_crash_resume(tmp_path: Path) -> None:
    """Exercise all 6 test methods (bws_3, bws, 2afc, pairwise with unsure, mos_likert, abx),
    warmup exclusion, hidden repeat concordance, undo, and truncated JSONL recovery."""
    arms3 = ["BF16", "Q1", "Q3"]
    arms2 = ["BF16", "Q1"]
    clips3 = _make_dummy_clips(tmp_path, arms3, ["W1", "C1", "C2", "C1-rep", "BWS4"])
    # Add a 4th arm to BWS4 for iterative BWS test
    p4 = tmp_path / "clips" / "NANO_BWS4.mp4"
    p4.write_bytes(b"FAKE_MP4_NANO_BWS4")
    clips3["BWS4"]["NANO"] = {"path": str(p4), "sha256": hashlib.sha256(b"FAKE_MP4_NANO_BWS4").hexdigest()}

    clips2 = _make_dummy_clips(tmp_path, arms2, ["P1", "PW1", "PW2", "MOS1", "ABX1"])

    cfg = {
        "schema_version": "blindpanel-config-v1",
        "study_id": "all-methods-study",
        "blocks": [
            {
                "block_id": "b_bws3",
                "method": "bws_3",
                "arms": arms3,
                "break_after_item": 2,
                "items": [
                    {
                        "cell_id": "W1",
                        "prompt_id": "M3",
                        "prompt_type": "manipulation",
                        "is_warmup": True,
                        "ask_q2": True,
                        "slot_arms": ["BF16", "Q1", "Q3"],
                        "arms": {k: clips3["W1"][k] for k in arms3},
                    },
                    {
                        "cell_id": "C1",
                        "prompt_id": "M1",
                        "prompt_type": "manipulation",
                        "ask_q2": True,
                        "slot_arms": ["BF16", "Q1", "Q3"],
                        "arms": {k: clips3["C1"][k] for k in arms3},
                    },
                    {
                        "cell_id": "C2",
                        "prompt_id": "L1",
                        "prompt_type": "locomotion",
                        "ask_q2": False,
                        "slot_arms": ["Q1", "Q3", "BF16"],
                        "arms": {k: clips3["C2"][k] for k in arms3},
                    },
                    {
                        "cell_id": "C1-rep",
                        "repeat_of_cell_id": "C1",
                        "prompt_id": "M1",
                        "prompt_type": "manipulation",
                        "is_repeat": True,
                        "ask_q2": True,
                        "slot_arms": ["Q1", "Q3", "BF16"],
                        "arms": {k: clips3["C1"][k] for k in arms3},
                    },
                ],
            },
            {
                "block_id": "b_bws4",
                "method": "bws",
                "arms": ["BF16", "Q1", "Q3", "NANO"],
                "items": [
                    {
                        "cell_id": "BWS4",
                        "prompt_id": "S1",
                        "prompt_type": "scene",
                        "arms": clips3["BWS4"],
                    }
                ],
            },
            {
                "block_id": "b_2afc",
                "method": "2afc",
                "arms": arms2,
                "items": [
                    {
                        "cell_id": "P1",
                        "prompt_id": "S2",
                        "prompt_type": "scene",
                        "slot_arms": ["BF16", "Q1"],
                        "arms": clips2["P1"],
                    }
                ],
            },
            {
                "block_id": "b_pairwise",
                "method": "pairwise",
                "allow_unsure": True,
                "arms": arms2,
                "items": [
                    {"cell_id": "PW1", "prompt_id": "S3", "arms": clips2["PW1"]},
                    {"cell_id": "PW2", "prompt_id": "S4", "arms": clips2["PW2"]},
                ],
            },
            {
                "block_id": "b_mos",
                "method": "mos_likert",
                "arms": arms2,
                "items": [{"cell_id": "MOS1", "prompt_id": "L2", "arms": clips2["MOS1"]}],
            },
            {
                "block_id": "b_abx",
                "method": "abx",
                "arms": arms2,
                "items": [
                    {
                        "cell_id": "ABX1",
                        "prompt_id": "L3",
                        "slot_arms": ["BF16", "Q1"],
                        "abx_target_arm": "BF16",
                        "arms": clips2["ABX1"],
                    }
                ],
            },
        ],
    }

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    sealed_path = tmp_path / "sealed_map.bin"
    ans_path = tmp_path / "answers.jsonl"

    session = BlindPanelSession(
        config_path=str(cfg_path),
        sealed_map_path=str(sealed_path),
        answers_path=str(ans_path),
    )

    session.start_session(mode="live")

    # Item 0 (W1 warmup, bws_3): A=BF16, B=Q1, C=Q3
    session.submit_rating({
        "global_index": 0,
        "best_letter": "A",
        "worst_letter": "C",
        "q2_by_letter": {"A": "yes", "B": "partly", "C": "no"},
    })

    # Item 1 (C1 core, bws_3): A=BF16, B=Q1, C=Q3 -> rank BF16 (1) > Q1 (2) > Q3 (3)
    st = session.submit_rating({
        "global_index": 1,
        "best_letter": "A",
        "worst_letter": "C",
        "q2_by_letter": {"A": "yes", "B": "partly", "C": "no"},
    })
    # Break after item 2 in block 1 triggers rest_break
    assert st["interstitial"] == "rest_break"
    st = session.resume_after_break(mode="live")
    assert st["interstitial"] is None

    # Item 2 (C2 core, bws_3): slot_arms=["Q1", "Q3", "BF16"] -> A=Q1, B=Q3, C=BF16
    # Test undo: submit wrong then undo then submit right
    session.submit_rating({"global_index": 2, "best_letter": "A", "worst_letter": "C"})
    session.record_undo(mode="live")
    session.submit_rating({"global_index": 2, "best_letter": "C", "worst_letter": "B"})

    # Item 3 (C1-rep repeat, bws_3): slot_arms=["Q1", "Q3", "BF16"] -> A=Q1, B=Q3, C=BF16
    # Rank BF16 (C=1) > Q1 (A=2) > Q3 (B=3) for 100% repeat concordance with Pass 1!
    session.submit_rating({
        "global_index": 3,
        "best_letter": "C",
        "worst_letter": "B",
        "q2_by_letter": {"A": "partly", "B": "no", "C": "yes"},
    })

    # Block transition to Block 2
    session.start_block2(mode="live")

    # Item 4 (BWS4 iterative 4-arm peel-off): 2 steps
    st4 = session.get_public_state(mode="live")
    rem = st4["remaining_letters"]
    session.submit_rating({"global_index": 4, "best_letter": rem[0], "worst_letter": rem[-1]})
    st4b = session.get_public_state(mode="live")
    rem2 = st4b["remaining_letters"]
    session.submit_rating({"global_index": 4, "best_letter": rem2[0], "worst_letter": rem2[1]})

    # Item 5 (2AFC): choice="A"
    session.submit_rating({"global_index": 5, "choice": "A"})

    # Item 6 & 7 (Pairwise with unsure)
    session.submit_rating({"global_index": 6, "choice": "unsure"})
    session.submit_rating({"global_index": 7, "choice": "B"})

    # Item 8 (MOS Likert)
    session.submit_rating({"global_index": 8, "likert_by_letter": {"A": 5, "B": 3}})

    # Item 9 (ABX): target is BF16 which is in slot A
    session.record_flag_flawed(9, flawed=True, reason="minor compression artifact")
    session.record_note(9, notes="checked frame 24")
    final_st = session.submit_rating({"global_index": 9, "choice": "A", "notes": "checked frame 24"})
    assert final_st["interstitial"] == "session_complete"

    # Simulate a crash with a partial trailing line in answers.jsonl and verify resume truncates it cleanly
    with open(ans_path, "a", encoding="utf-8") as f:
        f.write('{"event":"item_complete","broken_json":')

    resumed = BlindPanelSession(
        config_path=str(cfg_path),
        sealed_map_path=str(sealed_path),
        answers_path=str(ans_path),
    )
    assert resumed.get_public_state()["interstitial"] == "session_complete"

    # Export blind CSV & JSONL and verify zero arm names appear
    csv_text = resumed.export_blind_csv()
    for arm in ["BF16", "Q1", "Q3", "NANO"]:
        assert arm not in csv_text

    # Unblind and verify summary stats, Q2, ABX, and 100% repeat concordance on C1
    sealed_map = json.loads(sealed_path.read_text(encoding="utf-8"))
    events, _ = load_jsonl_safe(str(ans_path))
    unblinded = unblind_session(sealed_map, events)

    assert unblinded["warmup_completed_items"] == 1
    assert unblinded["completed_rows"] == 9
    assert unblinded["repeat_concordance"]["concordance"] == 1.0
    assert unblinded["abx_summary"]["correct"] == 1
    assert unblinded["pairwise_ties"] == 1
    assert unblinded["summary_by_arm"]["BF16"]["q2_action_completion"]["yes"] == 2


def test_cli_entrypoints_and_http_endpoints_and_validation_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI main() functions, single-block shorthand schema, HTTP API endpoints, byte-range serving,
    and tamper detection on sealed_map."""
    import threading
    import urllib.error
    import urllib.request

    from blindpanel.cosmos3_study import main as cosmos_main
    from blindpanel.schema import letters_for_count, validate_question2_spec
    from blindpanel.server import create_http_server
    from blindpanel.server import main as server_main

    # 1. Test schema helper edge cases
    with pytest.raises(ValueError, match="between 1 and 26"):
        letters_for_count(0)
    with pytest.raises(ValueError, match="must be a JSON object"):
        validate_question2_spec("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing non-empty 'prompt'"):
        validate_question2_spec({"prompt": ""})
    with pytest.raises(ValueError, match="at least 2 choices"):
        validate_question2_spec({"prompt": "ok?", "options": [{"id": "a", "label": "A"}]})
    with pytest.raises(ValueError, match="duplicate option id"):
        validate_question2_spec({
            "prompt": "ok?",
            "options": [{"id": "a", "label": "A"}, {"id": "a", "label": "A2"}],
        })

    # 2. Single-block shorthand config normalization
    arms = ["BF16", "Q1", "Q3"]
    clips = _make_dummy_clips(tmp_path, arms, ["M1-s1111"])
    shorthand_cfg = {
        "schema_version": "blindpanel-config-v1",
        "study_id": "shorthand-study",
        "rows": [
            {
                "cell_id": "M1-s1111",
                "prompt_id": "M1",
                "prompt_type": "manipulation",
                "arms": clips["M1-s1111"],
            }
        ],
        "preflight_flagged_rows": [
            {
                "prompt_id": "P1*",
                "prompt_type": "manipulation",
                "prompt": "Legacy 4-step prompt",
                "required_action": "4 steps",
                "estimated_seconds": "4.0-5.0 s",
                "estimated_max_seconds": 4.5,
                "flagged": True,
            }
        ],
    }
    cfg_path = tmp_path / "shorthand.json"
    cfg_path.write_text(json.dumps(shorthand_cfg), encoding="utf-8")
    sealed_path = tmp_path / "sealed.bin"
    ans_path = tmp_path / "ans.jsonl"

    _ = BlindPanelSession(
        config_path=str(cfg_path),
        sealed_map_path=str(sealed_path),
        answers_path=str(ans_path),
    )
    # Re-instantiate to exercise loading existing sealed_map from disk
    session2 = BlindPanelSession(
        config_path=str(cfg_path),
        sealed_map_path=str(sealed_path),
        answers_path=str(ans_path),
    )

    httpd = create_http_server(session2, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    try:
        # Exercise static & API GET endpoints
        for route in ("/", "/styles.css", "/app.js", "/api/state", "/api/operator/preflight", "/api/operator/audit", "/api/export/csv", "/api/export/jsonl"):
            with urllib.request.urlopen(base_url + route) as resp:
                assert resp.status == 200

        # Exercise Range request on /clip/0/A.mp4
        req = urllib.request.Request(base_url + "/clip/0/A.mp4", headers={"Range": "bytes=0-3"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 206
            assert len(resp.read()) == 4

        # Exercise POST /api/operator/override, /api/session/start, /api/flag_flawed, /api/note, /api/submit, /api/undo
        def post_json(route: str, payload: dict) -> dict:
            r = urllib.request.Request(
                base_url + route,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(r) as resp:
                return json.loads(resp.read().decode("utf-8"))

        post_json("/api/operator/override", {"reason": "Publisher override for test"})
        post_json("/api/session/start", {"mode": "live"})
        post_json("/api/flag_flawed", {"global_index": 0, "flawed": True, "reason": "test flag"})
        post_json("/api/note", {"global_index": 0, "notes": "note via HTTP"})
        post_json(
            "/api/submit",
            {
                "global_index": 0,
                "best_letter": "A",
                "worst_letter": "C",
                "q2_by_letter": {"A": "yes", "B": "partly", "C": "no"},
            },
        )
        post_json("/api/undo", {"mode": "live"})
        post_json(
            "/api/submit",
            {
                "global_index": 0,
                "best_letter": "B",
                "worst_letter": "C",
                "q2_by_letter": {"A": "partly", "B": "yes", "C": "no"},
            },
        )
    finally:
        httpd.shutdown()
        httpd.server_close()

    # 3. Test unblind CLI main()
    out_unblinded = tmp_path / "unblinded_out.json"
    rc = unblind_main(["--sealed-map", str(sealed_path), "--answers", str(ans_path), "--out", str(out_unblinded)])
    assert rc == 0
    assert out_unblinded.is_file()

    # 4. Test tamper detection in unblind_session when map_hash is modified
    tampered_map = json.loads(sealed_path.read_text(encoding="utf-8"))
    tampered_map["rows"][0]["map_hash"] = "f" * 64
    with pytest.raises(ValueError, match="Tamper check failed"):
        unblind_session(tampered_map, [])

    # 5. Test server_main non-localhost refusal and bad config return code
    assert server_main(["--config", str(cfg_path), "--host", "0.0.0.0"]) == 2
    assert server_main(["--config", str(tmp_path / "nonexistent.json")]) == 1

    # 6. Test cosmos3_study CLI main() without file verification
    out_cosmos = tmp_path / "cosmos_out.json"
    monkeypatch.setattr("sys.argv", ["cosmos3_study", "--clips-dir", "/tmp/clips", "--out", str(out_cosmos)])
    assert cosmos_main() == 0
    assert out_cosmos.is_file()

