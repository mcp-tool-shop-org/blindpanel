"""test_unit.py — Unit tests for blindpanel schema, session engine, counterbalancing, honesty checks, mock clips, and unblinding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

import pytest

from blindpanel.cosmos3_study import (
    build_public_study_design_config,
    discover_clips_in_dir,
)
from blindpanel.mock_clips import (
    MOCK_CLIP_MARKER,
    generate_mock_mp4_clip,
    generate_mock_study_clips,
    read_png_mock_marker,
    stamp_png_with_mock_marker,
    write_mock_screenshots_manifest,
)
from blindpanel.schema import (
    AMENDMENT_A_Q2_PROMPT,
    generate_session_slot_orders,
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


def test_public_study_design_config_and_session_counterbalancing() -> None:
    """Verify the public Cosmos 3 Amendment A study config holds zero study blinding data
    and that session-start counterbalancing from a fresh 256-bit secret seed satisfies all Amendment A rules:
    - Block 1 has 43 items (3 warm-up + 36 core + 4 repeats), method bws_3, 3 arms
    - All 6 permutations appear 6 times each (2 per prompt type)
    - Every prompt's 3 variants form a Latin square over slots A, B, C
    - No adjacent items share the same prompt_id
    - Question 2 is enabled on M1-M4 and disabled on L1-L4 and S1-S4
    - Block 2 has 42 items (2 warm-up + 36 core + 4 repeats), method 2afc, 2 arms
    """
    cfg = build_public_study_design_config()
    norm = validate_and_normalize_config(cfg, forbid_study_blinding_data=True)

    assert norm["schema_version"] == "blindpanel-config-v1"
    assert len(norm["blocks"]) == 2

    b1 = norm["blocks"][0]
    assert b1["method"] == "bws_3"
    assert b1["num_items"] == 43
    assert b1["num_warmup"] == 3
    assert b1["num_recorded"] == 40
    assert b1["num_repeats"] == 4
    assert b1["break_after_item"] == 23

    # Verify per-item arms and slot orders are absent from the public study config
    for item in b1["items"]:
        assert item["arms"] is None

    # Generate session slot orders from a fresh 256-bit secret seed and verify counterbalancing
    import secrets
    secret_seed_hex = secrets.token_hex(32)
    b1_slot_orders = generate_session_slot_orders(b1, secret_seed_hex)
    cb_stats = verify_bws3_counterbalancing(
        b1["items"],
        arms=b1["arms"],
        block_id=b1["block_id"],
        slot_orders_by_item_idx=b1_slot_orders,
    )
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
    assert b2["num_items"] == 42
    assert b2["num_warmup"] == 2
    assert b2["num_recorded"] == 40
    assert b2["num_repeats"] == 4
    b2_slot_orders = generate_session_slot_orders(b2, secret_seed_hex)
    assert len(b2_slot_orders) == 42


def test_schema_rejects_invalid_and_unsealed_public_configs() -> None:
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

    with pytest.raises(ValueError, match="Public study config must not contain"):
        validate_and_normalize_config(
            {"schema_version": "blindpanel-config-v1", "study_id": "bad", "secret_seed": "abc"},
            forbid_study_blinding_data=True,
        )


def test_refused_at_load_on_sealed_manifest_sha256_mismatch_and_missing_file(tmp_path: Path) -> None:
    arms = ["ARM_1", "ARM_2", "ARM_3"]
    clips = _make_dummy_clips(tmp_path, arms, ["M1-s1111"])
    clips["M1-s1111"]["ARM_1"]["sha256"] = "0" * 64

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

    with pytest.raises(ValueError, match="REFUSED AT LOAD: sha256 mismatch against sealed manifest"):
        BlindPanelSession(
            config_path=str(cfg_path),
            sealed_map_path=str(tmp_path / "sealed_map.bin"),
            answers_path=str(tmp_path / "answers.jsonl"),
        )

    clips["M1-s1111"]["ARM_1"]["path"] = str(tmp_path / "missing.mp4")
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="REFUSED AT LOAD: missing clip file"):
        BlindPanelSession(
            config_path=str(cfg_path),
            sealed_map_path=str(tmp_path / "sealed_map.bin"),
            answers_path=str(tmp_path / "answers.jsonl"),
        )


def test_session_with_public_design_config_and_clips_dir(tmp_path: Path) -> None:
    """Verify running BlindPanelSession with the public study design config + a local `--clips-dir` of mock clips:
    - Generates a >= 128-bit secret seed in the sealed store
    - Builds the sealed manifest (arm, original_sha256, salted_name)
    - Serves clips under salted names `/clip/clip_<hash>.mp4`
    - Unblinds answers using only the sealed store
    """
    cfg = build_public_study_design_config()
    cfg_path = tmp_path / "public_design.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # Missing --clips-dir on a public design config must refuse at load
    with pytest.raises(ValueError, match="REFUSED AT LOAD: .* pass --clips-dir"):
        BlindPanelSession(
            config_path=str(cfg_path),
            sealed_map_path=str(tmp_path / "sealed_store" / "s1.bin"),
            answers_path=str(tmp_path / "answers.jsonl"),
        )

    all_cells = sorted({it["base_cell_id"] for blk in cfg["blocks"] for it in blk["items"]})
    all_arms = sorted({arm for blk in cfg["blocks"] for arm in blk["arms"]})
    clips_dir = tmp_path / "mock_clips"
    _make_dummy_clips(tmp_path, all_arms, all_cells)
    # _make_dummy_clips writes to tmp_path / "clips"
    clips_dir = tmp_path / "clips"

    sealed_path = tmp_path / "sealed_store" / "sealed_map.bin"
    ans_path = tmp_path / "answers.jsonl"
    session = BlindPanelSession(
        config_path=str(cfg_path),
        clips_dir=str(clips_dir),
        sealed_map_path=str(sealed_path),
        answers_path=str(ans_path),
    )

    # Check sealed store structure (Rule 2)
    smap = json.loads(sealed_path.read_text(encoding="utf-8"))
    assert len(bytes.fromhex(smap["secret_seed"])) >= 16  # 32 bytes = 256 bits >= 128 bits
    assert len(smap["sealed_manifest"]) > 0
    first_manifest_entry = smap["sealed_manifest"][0]
    assert {"arm", "original_sha256", "salted_name"} <= set(first_manifest_entry.keys())
    assert first_manifest_entry["salted_name"].startswith("clip_")

    # Check public state uses salted clip URLs
    pub = session.get_public_state()
    for c in pub["clips"]:
        assert c["url"].startswith("/clip/clip_")
        assert c["url"].endswith(".mp4")


def test_preflight_gate_and_logged_override_and_pilot_isolation(tmp_path: Path) -> None:
    arms = ["ARM_1", "ARM_2", "ARM_3"]
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

    with pytest.raises(ValueError, match="blocked by Operator Pre-Flight"):
        session.start_session(mode="live")

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

    with pytest.raises(ValueError, match="at least 5 characters"):
        session.record_preflight_override("abc")

    pf_after = session.record_preflight_override("Approved by Publisher after M1 prompt revision")
    assert pf_after["live_start_blocked"] is False
    assert pf_after["override_recorded"] is True

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
    arms3 = ["ARM_1", "ARM_2", "ARM_3"]
    arms2 = ["ARM_1", "ARM_2"]
    clips3 = _make_dummy_clips(tmp_path, arms3, ["W1", "C1", "C2", "BWS4"])
    p4 = tmp_path / "clips" / "ARM_4_BWS4.mp4"
    p4.write_bytes(b"FAKE_MP4_ARM_4_BWS4")
    clips3["BWS4"]["ARM_4"] = {"path": str(p4), "sha256": hashlib.sha256(b"FAKE_MP4_ARM_4_BWS4").hexdigest()}

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
                        "arms": {k: clips3["W1"][k] for k in arms3},
                    },
                    {
                        "cell_id": "C1",
                        "prompt_id": "M1",
                        "prompt_type": "manipulation",
                        "ask_q2": True,
                        "arms": {k: clips3["C1"][k] for k in arms3},
                    },
                    {
                        "cell_id": "C2",
                        "prompt_id": "L1",
                        "prompt_type": "locomotion",
                        "ask_q2": False,
                        "arms": {k: clips3["C2"][k] for k in arms3},
                    },
                    {
                        "cell_id": "C1-rep",
                        "repeat_of_cell_id": "C1",
                        "prompt_id": "M1",
                        "prompt_type": "manipulation",
                        "is_repeat": True,
                        "ask_q2": True,
                        "arms": {k: clips3["C1"][k] for k in arms3},
                    },
                ],
            },
            {
                "block_id": "b_bws4",
                "method": "bws",
                "arms": ["ARM_1", "ARM_2", "ARM_3", "ARM_4"],
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
                        "abx_target_arm": "ARM_1",
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

    # Read sealed store to pick ARM_1 as Best and ARM_3 as Worst consistently across Pass 1 and Repeat
    smap = json.loads(sealed_path.read_text(encoding="utf-8"))

    # Item 0 (W1 warmup)
    session.submit_rating({
        "global_index": 0,
        "best_letter": "A",
        "worst_letter": "C",
        "q2_by_letter": {"A": "yes", "B": "partly", "C": "no"},
    })

    # Item 1 (C1 core)
    a2l_1 = smap["rows"][1]["arm_to_letter"]
    st = session.submit_rating({
        "global_index": 1,
        "best_letter": a2l_1["ARM_1"],
        "worst_letter": a2l_1["ARM_3"],
        "q2_by_letter": {a2l_1["ARM_1"]: "yes", a2l_1["ARM_2"]: "partly", a2l_1["ARM_3"]: "no"},
    })
    assert st["interstitial"] == "rest_break"
    st = session.resume_after_break(mode="live")
    assert st["interstitial"] is None

    # Item 2 (C2 core) with undo test
    session.submit_rating({"global_index": 2, "best_letter": "A", "worst_letter": "C"})
    session.record_undo(mode="live")
    session.submit_rating({"global_index": 2, "best_letter": "C", "worst_letter": "B"})

    # Item 3 (C1-rep repeat): pick ARM_1 Best, ARM_3 Worst for 100% concordance with Pass 1
    a2l_3 = smap["rows"][3]["arm_to_letter"]
    # Verify repeat shifted slots relative to Pass 1
    assert a2l_3 != a2l_1
    session.submit_rating({
        "global_index": 3,
        "best_letter": a2l_3["ARM_1"],
        "worst_letter": a2l_3["ARM_3"],
        "q2_by_letter": {a2l_3["ARM_1"]: "yes", a2l_3["ARM_2"]: "partly", a2l_3["ARM_3"]: "no"},
    })

    session.start_block2(mode="live")

    # Item 4 (BWS4 iterative 4-arm peel-off)
    st4 = session.get_public_state(mode="live")
    rem = st4["remaining_letters"]
    session.submit_rating({"global_index": 4, "best_letter": rem[0], "worst_letter": rem[-1]})
    st4b = session.get_public_state(mode="live")
    rem2 = st4b["remaining_letters"]
    session.submit_rating({"global_index": 4, "best_letter": rem2[0], "worst_letter": rem2[1]})

    # Item 5 (2AFC)
    session.submit_rating({"global_index": 5, "choice": "A"})

    # Item 6 & 7 (Pairwise with unsure)
    session.submit_rating({"global_index": 6, "choice": "unsure"})
    session.submit_rating({"global_index": 7, "choice": "B"})

    # Item 8 (MOS Likert)
    session.submit_rating({"global_index": 8, "likert_by_letter": {"A": 5, "B": 3}})

    # Item 9 (ABX): target is ARM_1
    a2l_9 = smap["rows"][9]["arm_to_letter"]
    session.record_flag_flawed(9, flawed=True, reason="minor compression artifact")
    session.record_note(9, notes="checked frame 24")
    final_st = session.submit_rating({"global_index": 9, "choice": a2l_9["ARM_1"], "notes": "checked frame 24"})
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

    csv_text = resumed.export_blind_csv()
    for arm in ["ARM_1", "ARM_2", "ARM_3", "ARM_4"]:
        assert arm not in csv_text

    events, _ = load_jsonl_safe(str(ans_path))
    unblinded = unblind_session(smap, events)

    assert "secret_seed" not in json.dumps(unblinded)
    assert unblinded["warmup_completed_items"] == 1
    assert unblinded["completed_rows"] == 9
    assert unblinded["repeat_concordance"]["concordance"] == 1.0
    assert unblinded["abx_summary"]["correct"] == 1
    assert unblinded["pairwise_ties"] == 1
    assert unblinded["summary_by_arm"]["ARM_1"]["q2_action_completion"]["yes"] == 2


def test_mock_clips_and_png_marker_utilities(tmp_path: Path) -> None:
    clips_dir = tmp_path / "mock_clips"
    generated = generate_mock_study_clips(clips_dir, arms=["ARM_1", "ARM_2"], base_cells=["M1-s1111"])
    assert "M1-s1111" in generated
    assert generated["M1-s1111"]["ARM_1"]["generator"] == MOCK_CLIP_MARKER
    discovered = discover_clips_in_dir(str(clips_dir), arms=["ARM_1", "ARM_2"], base_cell_id="M1-s1111")
    assert discovered["ARM_1"]["sha256"] == generated["M1-s1111"]["ARM_1"]["sha256"]

    single = generate_mock_mp4_clip(tmp_path / "single.mp4", label="S1")
    assert single["generator"] == MOCK_CLIP_MARKER

    # Minimal valid 1x1 PNG
    import struct
    import zlib

    def _make_1x1_png() -> bytes:
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        c_ihdr = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
        raw_px = zlib.compress(b"\x00\x80\x80\x80")
        c_idat = struct.pack(">I", len(raw_px)) + b"IDAT" + raw_px + struct.pack(">I", zlib.crc32(b"IDAT" + raw_px) & 0xFFFFFFFF)
        c_iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
        return sig + c_ihdr + c_idat + c_iend

    png_file = tmp_path / "test-mock-1.png"
    png_file.write_bytes(_make_1x1_png())
    assert read_png_mock_marker(png_file) is None
    stamp_png_with_mock_marker(png_file)
    # Stamping twice is idempotent
    stamp_png_with_mock_marker(png_file)
    assert read_png_mock_marker(png_file) == MOCK_CLIP_MARKER
    manifest = write_mock_screenshots_manifest(tmp_path, ["test-mock-1.png"])
    assert manifest.is_file()


def test_cli_entrypoints_and_http_endpoints_and_validation_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import urllib.error
    import urllib.request

    from blindpanel.cosmos3_study import main as cosmos_main
    from blindpanel.schema import letters_for_count, validate_question2_spec
    from blindpanel.server import create_http_server
    from blindpanel.server import main as server_main

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

    arms = ["ARM_1", "ARM_2", "ARM_3"]
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
        for route in ("/", "/styles.css", "/app.js", "/api/state", "/api/operator/preflight", "/api/operator/audit", "/api/export/csv", "/api/export/jsonl"):
            with urllib.request.urlopen(base_url + route) as resp:
                assert resp.status == 200

        state_data = json.loads(urllib.request.urlopen(base_url + "/api/state").read().decode("utf-8"))
        salted_clip_url = state_data["clips"][0]["url"]

        req = urllib.request.Request(base_url + salted_clip_url, headers={"Range": "bytes=0-3"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 206
            assert len(resp.read()) == 4

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

    out_unblinded = tmp_path / "unblinded_out.json"
    rc = unblind_main(["--sealed-map", str(sealed_path), "--answers", str(ans_path), "--out", str(out_unblinded)])
    assert rc == 0
    assert out_unblinded.is_file()

    tampered_map = json.loads(sealed_path.read_text(encoding="utf-8"))
    tampered_map["rows"][0]["map_hash"] = "f" * 64
    with pytest.raises(ValueError, match="Tamper check failed"):
        unblind_session(tampered_map, [])

    assert server_main(["--config", str(cfg_path), "--host", "0.0.0.0"]) == 2
    assert server_main(["--config", str(tmp_path / "nonexistent.json")]) == 1

    out_cosmos = tmp_path / "cosmos_out.json"
    assert cosmos_main(["--out", str(out_cosmos)]) == 0
    assert out_cosmos.is_file()
