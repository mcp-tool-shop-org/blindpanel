"""server.py — Local-only HTTP server and session engine for blindpanel.

Enforces:
- Versioned schema validation at load (`blindpanel-config-v1`)
- SHA-256 verification of all clips at load (`REFUSED AT LOAD` on missing file or sha256 mismatch)
- Optional ffprobe clip geometry/frame/fps check when `verify_clip_shape=True`
- Cryptographically salted `sealed_map_*.bin` stored outside the browser reach (HTTP 404 on any sealed_map request)
- Strictly blind `/api/state` and `/clip/<global_index>/<letter>.mp4` endpoints (never leaks arm names, paths,
  sha256, warm-up flags, repeat flags, or expected durations to rater views)
- Operator Pre-Flight check & logged override gate (blocks live start if any item is flagged unless an operator
  override with a non-empty reason is recorded to the audit log)
- Pilot mode (`pilot_answers_<date>.jsonl`) isolated from live `answers.jsonl`, running on the real clips
- Crash-safe append-only JSONL event log with automatic trailing partial-line truncation on resume
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import random
import secrets
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qs, urlparse

from blindpanel.schema import letters_for_count, validate_and_normalize_config
from blindpanel.unblind import canonical_map_hash, load_jsonl_safe, reconstruct_session_answers


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def probe_clip_shape(path: str) -> Dict[str, Any]:
    """Probe width, height, nb_frames, and r_frame_rate using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate",
        "-of",
        "json",
        path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=15)
    data = json.loads(res.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError(f"ffprobe found no video stream in '{path}'")
    st = streams[0]
    fps_raw = str(st.get("r_frame_rate", "24/1"))
    if "/" in fps_raw:
        num, den = fps_raw.split("/", 1)
        fps_val = float(num) / float(den) if float(den) != 0 else 0.0
    else:
        fps_val = float(fps_raw)
    return {
        "width": int(st.get("width", 0)),
        "height": int(st.get("height", 0)),
        "frames": int(st.get("nb_frames", 0)),
        "fps": f"{fps_val:.4f}",
    }


class BlindPanelSession:
    """Thread-safe state manager for live and pilot blindpanel sessions."""

    def __init__(
        self,
        config_path: str,
        sealed_map_path: str,
        answers_path: str,
        pilot_answers_path: Optional[str] = None,
        audit_log_path: Optional[str] = None,
        verify_clip_shape: bool = False,
    ) -> None:
        self.lock = threading.Lock()
        self.config_path = os.path.abspath(config_path)
        self.sealed_map_path = os.path.abspath(sealed_map_path)
        self.answers_path = os.path.abspath(answers_path)

        date_tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
        ans_dir = os.path.dirname(self.answers_path)
        if pilot_answers_path:
            self.pilot_answers_path = os.path.abspath(pilot_answers_path)
        else:
            self.pilot_answers_path = os.path.join(ans_dir, f"pilot_answers_{date_tag}.jsonl")

        if audit_log_path:
            self.audit_log_path = os.path.abspath(audit_log_path)
        else:
            self.audit_log_path = os.path.join(ans_dir, "operator_audit.jsonl")

        self.verify_clip_shape = verify_clip_shape

        with open(self.config_path, "rb") as f:
            raw_bytes = f.read()
        self.config_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        raw_cfg = json.loads(raw_bytes.decode("utf-8"))
        self.config = validate_and_normalize_config(raw_cfg)

        self.study_id: str = self.config["study_id"]
        self.title: str = self.config["title"]
        self.rng_seed: int = self.config["rng_seed"]
        self.clip_spec: Dict[str, Any] = self.config["clip_spec"]
        self.blocks: List[Dict[str, Any]] = self.config["blocks"]
        self.flat_items: List[Dict[str, Any]] = self.config["flat_items"]

        self.verified_clip_count = 0
        self._verify_all_clips()
        self.sealed_map = self._init_or_load_sealed_map()

        # Runtime state for Live ("live") and Pilot ("pilot") modes
        self.events_by_mode: Dict[str, List[Dict[str, Any]]] = {"live": [], "pilot": []}
        self.item_states_by_mode: Dict[str, Dict[int, Dict[str, Any]]] = {"live": {}, "pilot": {}}
        self.cursor_by_mode: Dict[str, int] = {"live": 0, "pilot": 0}
        self.interstitial_by_mode: Dict[str, Optional[str]] = {"live": "session_start", "pilot": "session_start"}
        self.block2_declined_by_mode: Dict[str, bool] = {"live": False, "pilot": False}

        # Load audit events (operator pre-flight overrides, pilot resets, etc.)
        self.audit_events, _ = load_jsonl_safe(self.audit_log_path)

        # Resume live and pilot logs safely
        self._load_or_init_mode("live", self.answers_path)
        self._load_or_init_mode("pilot", self.pilot_answers_path)

    def _resolve_clip_path(self, rel_or_abs: str) -> str:
        if os.path.isabs(rel_or_abs) and os.path.exists(rel_or_abs):
            return os.path.abspath(rel_or_abs)
        cand = os.path.abspath(os.path.join(os.path.dirname(self.config_path), rel_or_abs))
        if os.path.exists(cand):
            return cand
        return os.path.abspath(rel_or_abs)

    def _verify_all_clips(self) -> None:
        """Verify every clip exists on disk, matches its SHA-256 hash, and (optionally) matches clip_spec."""
        checked_cache: Dict[str, str] = {}
        expected_w = int(self.clip_spec.get("width", 832))
        expected_h = int(self.clip_spec.get("height", 480))
        expected_frames = int(self.clip_spec.get("frames", 61))
        expected_fps = str(self.clip_spec.get("fps", "24.0000"))

        for item in self.flat_items:
            g_idx = item["global_index"]
            cid = item["cell_id"]
            for arm_name, spec in item["arms"].items():
                resolved = self._resolve_clip_path(spec["path"])
                if not os.path.isfile(resolved):
                    raise ValueError(
                        f"REFUSED AT LOAD: missing clip file for item {g_idx} ({cid}) arm '{arm_name}': {resolved}"
                    )
                expected_sha = spec["sha256"].lower()
                if resolved not in checked_cache:
                    actual_sha = sha256_file(resolved).lower()
                    if actual_sha != expected_sha:
                        raise ValueError(
                            f"REFUSED AT LOAD: sha256 mismatch for item {g_idx} ({cid}) arm '{arm_name}' "
                            f"({resolved}): expected {expected_sha}, got {actual_sha}"
                        )
                    if self.verify_clip_shape:
                        shape = probe_clip_shape(resolved)
                        if (
                            shape["width"] != expected_w
                            or shape["height"] != expected_h
                            or shape["frames"] != expected_frames
                            or shape["fps"] != expected_fps
                        ):
                            raise ValueError(
                                f"REFUSED AT LOAD: clip shape mismatch for '{resolved}': "
                                f"expected {expected_w}x{expected_h}, {expected_frames}f @ {expected_fps}fps; "
                                f"got {shape['width']}x{shape['height']}, {shape['frames']}f @ {shape['fps']}fps"
                            )
                    checked_cache[resolved] = actual_sha
                else:
                    if checked_cache[resolved] != expected_sha:
                        raise ValueError(
                            f"REFUSED AT LOAD: sha256 mismatch for item {g_idx} ({cid}) arm '{arm_name}': "
                            f"expected {expected_sha}, got {checked_cache[resolved]}"
                        )
                spec["resolved_path"] = resolved
        self.verified_clip_count = len(checked_cache)

    def _init_or_load_sealed_map(self) -> Dict[str, Any]:
        if os.path.exists(self.sealed_map_path):
            with open(self.sealed_map_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("config_sha256") == self.config_sha256 and len(existing.get("rows", [])) == len(
                self.flat_items
            ):
                # Hydrate resolved_paths in existing sealed map
                for idx, rmeta in enumerate(existing["rows"]):
                    item = self.flat_items[idx]
                    l2a = rmeta["letter_to_arm"]
                    rmeta["letter_to_path"] = {
                        let: item["arms"][arm]["resolved_path"] for let, arm in l2a.items()
                    }
                return existing

        rng = random.Random(self.rng_seed)
        sealed_rows: List[Dict[str, Any]] = []
        all_study_arms: List[str] = []
        for blk in self.blocks:
            for a in blk["arms"]:
                if a not in all_study_arms:
                    all_study_arms.append(a)

        for item in self.flat_items:
            g_idx = item["global_index"]
            blk_arms = list(self.blocks[item["block_index"]]["arms"])
            letters = letters_for_count(len(blk_arms))

            if item.get("slot_arms"):
                shuffled_arms = list(item["slot_arms"])
            else:
                shuffled_arms = list(blk_arms)
                rng.shuffle(shuffled_arms)

            letter_to_arm = {letters[i]: shuffled_arms[i] for i in range(len(letters))}
            arm_to_letter = {shuffled_arms[i]: letters[i] for i in range(len(letters))}
            salt = secrets.token_hex(16)
            mhash = canonical_map_hash(letter_to_arm, salt)

            abx_target = item.get("abx_target_arm")
            if item["method"] == "abx" and not abx_target:
                abx_target = rng.choice(shuffled_arms)

            letter_to_path = {
                let: item["arms"][arm]["resolved_path"] for let, arm in letter_to_arm.items()
            }
            if item["method"] == "abx" and abx_target:
                letter_to_path["X"] = item["arms"][abx_target]["resolved_path"]

            sealed_rows.append({
                "global_index": g_idx,
                "row_index": g_idx,
                "block_id": item["block_id"],
                "block_index": item["block_index"],
                "item_index": item["item_index"],
                "item_number_in_block": item["item_number_in_block"],
                "cell_id": item["cell_id"],
                "prompt_id": item["prompt_id"],
                "prompt_type": item["prompt_type"],
                "seed": item["seed"],
                "prompt": item["prompt"],
                "method": item["method"],
                "is_warmup": item["is_warmup"],
                "is_repeat": item["is_repeat"],
                "repeat_of_cell_id": item["repeat_of_cell_id"],
                "ask_q2": item["ask_q2"],
                "abx_target_arm": abx_target,
                "salt": salt,
                "map_hash": mhash,
                "letter_to_arm": letter_to_arm,
                "arm_to_letter": arm_to_letter,
                "letter_to_path": letter_to_path,
            })

        smap = {
            "schema": "blindpanel-sealed-map-v1",
            "study_id": self.study_id,
            "arena": self.study_id,
            "title": self.title,
            "config_sha256": self.config_sha256,
            "rng_seed": self.rng_seed,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "arms": all_study_arms,
            "rows": sealed_rows,
        }
        sealed_dir = os.path.dirname(self.sealed_map_path)
        if sealed_dir:
            os.makedirs(sealed_dir, exist_ok=True)
        # Do not store resolved local paths permanently if not needed, or keep in memory and write clean JSON
        disk_copy = json.loads(json.dumps(smap))
        for r in disk_copy["rows"]:
            r.pop("letter_to_path", None)
        with open(self.sealed_map_path, "w", encoding="utf-8") as f:
            json.dump(disk_copy, f, indent=2)
        return smap

    def _load_or_init_mode(self, mode: str, path: str) -> None:
        events, truncated = load_jsonl_safe(path)
        if truncated > 0:
            # Rewrite clean lines so future appends are well-formed JSONL
            with open(path, "w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev, separators=(",", ":")) + "\n")
        self.events_by_mode[mode] = events
        self._rebuild_mode_state(mode)

    def _append_event(self, mode: str, event: Dict[str, Any]) -> None:
        event["ts_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event["mode"] = mode
        path = self.answers_path if mode == "live" else self.pilot_answers_path
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
            f.flush()
        self.events_by_mode[mode].append(event)
        self._rebuild_mode_state(mode)

    def _rebuild_mode_state(self, mode: str) -> None:
        events = self.events_by_mode[mode]
        states = reconstruct_session_answers(self.sealed_map, events)
        self.item_states_by_mode[mode] = states

        session_started = any(ev.get("event") in ("session_start", "item_complete", "step") for ev in events)
        block2_declined = False
        last_break_ack: Set[int] = set()
        block2_started = False

        for ev in events:
            et = ev.get("event")
            if et == "break_resume":
                last_break_ack.add(int(ev.get("global_index", -1)))
            elif et == "block2_start":
                block2_started = True
                block2_declined = False
            elif et == "block2_decline":
                block2_declined = True

        self.block2_declined_by_mode[mode] = block2_declined

        # Determine cursor (first incomplete item)
        cursor = len(self.flat_items) - 1
        all_done = True
        for idx, item in enumerate(self.flat_items):
            if item["block_index"] > 0 and block2_declined:
                continue
            if not states[idx]["completed"]:
                cursor = idx
                all_done = False
                break

        self.cursor_by_mode[mode] = cursor

        if not session_started:
            self.interstitial_by_mode[mode] = "session_start"
            return

        if all_done:
            self.interstitial_by_mode[mode] = "session_complete"
            return

        # Check if we just completed an item that has a mid-block break or block transition
        if cursor > 0:
            prev_idx = cursor - 1
            prev_item = self.flat_items[prev_idx]
            cur_item = self.flat_items[cursor]
            if states[prev_idx]["completed"]:
                if prev_item["break_after"] and prev_idx not in last_break_ack:
                    self.interstitial_by_mode[mode] = "rest_break"
                    return
                if prev_item["block_index"] != cur_item["block_index"] and not block2_started:
                    self.interstitial_by_mode[mode] = "block_break"
                    return

        self.interstitial_by_mode[mode] = None

    # ------------------------------------------------------------------
    # Pre-Flight & Audit methods
    # ------------------------------------------------------------------
    def get_preflight_status(self) -> Dict[str, Any]:
        """Build operator pre-flight report. Names arms, shows required actions, estimated seconds vs clip length,
        and flags any item whose required action exceeds clip duration."""
        seen_prompts: Dict[str, Dict[str, Any]] = {}
        for item in self.flat_items:
            if item["is_warmup"]:
                continue
            pid = item["prompt_id"]
            if pid not in seen_prompts:
                seen_prompts[pid] = {
                    "prompt_id": pid,
                    "prompt_type": item["prompt_type"],
                    "prompt": item["prompt"],
                    "required_action": item["required_action"],
                    "estimated_seconds": item["estimated_seconds"],
                    "estimated_max_seconds": item["estimated_max_seconds"],
                    "clip_seconds": item["clip_seconds"],
                    "ask_q2": item["ask_q2"],
                    "flagged": item["preflight_flagged"],
                    "status": "flagged" if item["preflight_flagged"] else "pass",
                }
        rows = list(seen_prompts.values())
        for extra in self.config.get("preflight_flagged_rows", []):
            rows.append({
                "prompt_id": str(extra.get("prompt_id", "FLAG")),
                "prompt_type": str(extra.get("prompt_type", "manipulation")),
                "prompt": str(extra.get("prompt", "")),
                "required_action": str(extra.get("required_action", "")),
                "estimated_seconds": str(extra.get("estimated_seconds", "> 2.5 s")),
                "estimated_max_seconds": float(extra.get("estimated_max_seconds", 4.0)),
                "clip_seconds": float(extra.get("clip_seconds", self.clip_spec.get("seconds", 2.54))),
                "ask_q2": bool(extra.get("ask_q2", False)),
                "flagged": bool(extra.get("flagged", True)),
                "status": "flagged" if extra.get("flagged", True) else "pass",
            })

        flagged_rows = [r for r in rows if r["flagged"]]
        overrides = [ev for ev in self.audit_events if ev.get("event") == "preflight_override"]
        latest_override = overrides[-1] if overrides else None
        live_blocked = len(flagged_rows) > 0 and latest_override is None

        return {
            "study_id": self.study_id,
            "title": self.title,
            "config_sha256": self.config_sha256,
            "verified_clip_count": self.verified_clip_count,
            "clip_spec": self.clip_spec,
            "blocks_summary": [
                {
                    "block_id": b["block_id"],
                    "title": b["title"],
                    "method": b["method"],
                    "arms": b["arms"],
                    "optional": b["optional"],
                    "num_items": b["num_items"],
                    "num_warmup": b["num_warmup"],
                    "num_recorded": b["num_recorded"],
                    "num_repeats": b["num_repeats"],
                }
                for b in self.blocks
            ],
            "rows": rows,
            "flagged_count": len(flagged_rows),
            "flagged_prompt_ids": [r["prompt_id"] for r in flagged_rows],
            "override_recorded": latest_override is not None,
            "latest_override": latest_override,
            "live_start_blocked": live_blocked,
        }

    def record_preflight_override(self, operator_note: str) -> Dict[str, Any]:
        with self.lock:
            note = (operator_note or "").strip()
            if len(note) < 5:
                raise ValueError("Pre-flight override requires a non-empty reason (at least 5 characters).")
            ev = {
                "event": "preflight_override",
                "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "config_sha256": self.config_sha256,
                "reason": note,
            }
            out_dir = os.path.dirname(self.audit_log_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, separators=(",", ":")) + "\n")
                f.flush()
            self.audit_events.append(ev)
            return self.get_preflight_status()

    def get_operator_audit(self) -> Dict[str, Any]:
        with self.lock:
            pf = self.get_preflight_status()
            live_states = self.item_states_by_mode["live"]
            pilot_states = self.item_states_by_mode["pilot"]
            live_completed = sum(1 for st in live_states.values() if st["completed"] and not st["is_warmup"])
            live_warmup = sum(1 for st in live_states.values() if st["completed"] and st["is_warmup"])
            pilot_completed = sum(1 for st in pilot_states.values() if st["completed"])
            flawed_items = [
                {
                    "global_index": st["global_index"],
                    "block_id": st["block_id"],
                    "item_number_in_block": self.flat_items[st["global_index"]]["item_number_in_block"],
                    "cell_id": st["cell_id"],
                    "flaw_reason": st["flaw_reason"],
                    "notes": st["notes"],
                }
                for st in live_states.values()
                if st["flawed"] or st["notes"]
            ]
            return {
                "study_id": self.study_id,
                "title": self.title,
                "config_sha256": self.config_sha256,
                "sealed_map_filename": os.path.basename(self.sealed_map_path),
                "sealed_map_sealed": True,
                "verified_clip_count": self.verified_clip_count,
                "clip_spec": self.clip_spec,
                "preflight": pf,
                "live_session": {
                    "answers_file": os.path.basename(self.answers_path),
                    "events_count": len(self.events_by_mode["live"]),
                    "recorded_completed": live_completed,
                    "warmup_completed": live_warmup,
                    "total_items": len(self.flat_items),
                    "flawed_or_noted_items": flawed_items,
                },
                "pilot_session": {
                    "pilot_answers_file": os.path.basename(self.pilot_answers_path),
                    "events_count": len(self.events_by_mode["pilot"]),
                    "completed_items": pilot_completed,
                },
                "audit_events": self.audit_events,
            }

    # ------------------------------------------------------------------
    # Strictly Blind Rater State & Actions
    # ------------------------------------------------------------------
    def get_public_state(self, mode: str = "live", requested_global_index: Optional[int] = None) -> Dict[str, Any]:
        """Return strictly blind rater state.

        Contains ZERO arm names, file paths, sha256s, warm-up indicators, repeat flags,
        expected durations, or sealed map mappings.
        """
        with self.lock:
            mode = "pilot" if mode == "pilot" else "live"
            pf = self.get_preflight_status()
            states = self.item_states_by_mode[mode]
            cursor = self.cursor_by_mode[mode]
            interstitial = self.interstitial_by_mode[mode]

            if requested_global_index is not None and 0 <= requested_global_index < len(self.flat_items):
                active_idx = requested_global_index
            else:
                active_idx = cursor

            item = self.flat_items[active_idx]
            st = states[active_idx]
            blk = self.blocks[item["block_index"]]
            srow = self.sealed_map["rows"][active_idx]

            letters = list(st["letters"])
            clips = [
                {
                    "letter": let,
                    "url": f"/clip/{active_idx}/{let}.mp4",
                    "rank": st["ranks_by_letter"].get(let),
                    "remaining": let in st["remaining"],
                    "q2_answer": st["q2_by_letter"].get(let),
                    "likert_score": st["likert_by_letter"].get(let),
                }
                for let in letters
            ]
            reference_clips = None
            if item["method"] == "abx":
                reference_clips = {
                    "A": f"/clip/{active_idx}/A.mp4",
                    "B": f"/clip/{active_idx}/B.mp4",
                    "X": f"/clip/{active_idx}/X.mp4",
                }

            # Count completed items in current block
            block_items = blk["items"]
            block_completed = sum(
                1 for bit in block_items if states[bit["global_index"]]["completed"]
            )

            # Check whether undo is available
            can_undo = any(
                ev.get("event") in ("item_complete", "step") for ev in self.events_by_mode[mode]
            )

            # Public Q2 specification (only when ask_q2 is True for this row)
            pub_q2 = None
            if item["ask_q2"] and item["question2_spec"]:
                pub_q2 = {
                    "id": item["question2_spec"]["id"],
                    "prompt": item["question2_spec"]["prompt"],
                    "per_clip": item["question2_spec"]["per_clip"],
                    "options": item["question2_spec"]["options"],
                }

            return {
                "study_id": self.study_id,
                "title": self.title,
                "mode": mode,
                "live_start_blocked": (mode == "live" and pf["live_start_blocked"]),
                "interstitial": interstitial,
                "block_index": item["block_index"],
                "block_number": item["block_index"] + 1,
                "total_blocks": len(self.blocks),
                "block_title": blk["title"],
                "block_optional": blk["optional"],
                "global_index": active_idx,
                "cursor_global_index": cursor,
                "item_number": item["item_number_in_block"],
                "block_total_items": item["block_total_items"],
                "block_completed_items": block_completed,
                "counter_label": f"Item {item['item_number_in_block']} / {item['block_total_items']}",
                "progress_aria_label": f"Item {item['item_number_in_block']} of {item['block_total_items']}",
                "prompt_id": item["prompt_id"],
                "prompt": item["prompt"],
                "rater_cue": item["rater_cue"],
                "cue_strip_text": f"Look at: {item['rater_cue']}",
                "method": item["method"],
                "allow_unsure": item["allow_unsure"],
                "allow_skip": item["allow_skip"],
                "require_full_play": item["require_full_play"],
                "ask_q2": item["ask_q2"],
                "question2": pub_q2,
                "likert_scale": item["likert_scale"],
                "k": st["k"],
                "letters": letters,
                "remaining_letters": list(st["remaining"]),
                "ranks_by_letter": dict(st["ranks_by_letter"]),
                "choice_outcome": st["choice_outcome"],
                "q2_by_letter": dict(st["q2_by_letter"]),
                "likert_by_letter": dict(st["likert_by_letter"]),
                "completed": st["completed"],
                "flawed": st["flawed"],
                "flaw_reason": st["flaw_reason"],
                "notes": st["notes"],
                "can_undo": can_undo,
                "map_hash": srow["map_hash"],
                "clip_width": int(self.clip_spec.get("width", 832)),
                "clip_height": int(self.clip_spec.get("height", 480)),
                "clips": clips,
                "reference_clips": reference_clips,
            }

    def start_session(self, mode: str = "live") -> Dict[str, Any]:
        with self.lock:
            mode = "pilot" if mode == "pilot" else "live"
            if mode == "live":
                pf = self.get_preflight_status()
                if pf["live_start_blocked"]:
                    raise ValueError(
                        "Live session start is blocked by Operator Pre-Flight check: "
                        f"flagged prompts {pf['flagged_prompt_ids']} exceed clip length without a logged override."
                    )
            self._append_event(mode, {"event": "session_start"})
        return self.get_public_state(mode=mode)

    def resume_after_break(self, mode: str = "live") -> Dict[str, Any]:
        with self.lock:
            mode = "pilot" if mode == "pilot" else "live"
            cursor = self.cursor_by_mode[mode]
            prev_idx = max(0, cursor - 1)
            self._append_event(mode, {"event": "break_resume", "global_index": prev_idx})
        return self.get_public_state(mode=mode)

    def start_block2(self, mode: str = "live") -> Dict[str, Any]:
        with self.lock:
            mode = "pilot" if mode == "pilot" else "live"
            self._append_event(mode, {"event": "block2_start"})
        return self.get_public_state(mode=mode)

    def decline_block2(self, mode: str = "live") -> Dict[str, Any]:
        with self.lock:
            mode = "pilot" if mode == "pilot" else "live"
            self._append_event(mode, {"event": "block2_decline"})
        return self.get_public_state(mode=mode)

    def submit_rating(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and record a rating submission for the active item."""
        with self.lock:
            mode = "pilot" if payload.get("mode") == "pilot" else "live"
            if mode == "live" and self.get_preflight_status()["live_start_blocked"]:
                raise ValueError("Live session is blocked until Operator Pre-Flight override is recorded.")

            g_idx = int(payload.get("global_index", self.cursor_by_mode[mode]))
            if g_idx < 0 or g_idx >= len(self.flat_items):
                raise ValueError(f"Invalid global_index {g_idx}")

            item = self.flat_items[g_idx]
            st = self.item_states_by_mode[mode][g_idx]
            method = item["method"]
            letters = st["letters"]

            # Validate Q2 when required on this row
            q2_by_letter: Dict[str, str] = {}
            if item["ask_q2"] and item["question2_spec"]:
                raw_q2 = payload.get("q2_by_letter")
                if not isinstance(raw_q2, dict):
                    raise ValueError("Question 2 responses ('q2_by_letter') are required for this item.")
                valid_q2_ids = {opt["id"] for opt in item["question2_spec"]["options"]}
                for let in letters:
                    ans = str(raw_q2.get(let, "")).strip().lower()
                    if ans not in valid_q2_ids:
                        raise ValueError(
                            f"Question 2 answer required for Clip {let} (one of {sorted(valid_q2_ids)})."
                        )
                    q2_by_letter[let] = ans

            notes = str(payload.get("notes", "")).strip()
            flawed = bool(payload.get("flawed", st["flawed"]))

            if method == "bws_3":
                best_l = str(payload.get("best_letter", "")).strip().upper()
                worst_l = str(payload.get("worst_letter", "")).strip().upper()
                if best_l not in letters or worst_l not in letters:
                    raise ValueError(f"best_letter and worst_letter must be in {letters}")
                if best_l == worst_l:
                    raise ValueError("Best and Worst cannot be the same clip")
                mid_l = [let for let in letters if let not in (best_l, worst_l)][0]
                ranks_by_letter = {best_l: 1, mid_l: 2, worst_l: 3}
                self._append_event(
                    mode,
                    {
                        "event": "item_complete",
                        "global_index": g_idx,
                        "row_index": g_idx,
                        "block_id": item["block_id"],
                        "item_index": item["item_index"],
                        "cell_id": item["cell_id"],
                        "method": method,
                        "best_letter": best_l,
                        "middle_letter": mid_l,
                        "worst_letter": worst_l,
                        "ranks_by_letter": ranks_by_letter,
                        "q2_by_letter": q2_by_letter,
                        "flawed": flawed,
                        "notes": notes,
                    },
                )

            elif method == "bws":
                # Iterative BWS peel-off step
                rem = st["remaining"]
                best_l = str(payload.get("best_letter", "")).strip().upper()
                worst_l = str(payload.get("worst_letter", "")).strip().upper()
                if best_l not in rem or worst_l not in rem:
                    raise ValueError(f"Selections must be among remaining clips {rem}")
                if best_l == worst_l:
                    raise ValueError("Best and Worst cannot be the same clip")
                step_idx = (st["k"] - len(rem)) // 2
                self._append_event(
                    mode,
                    {
                        "event": "step",
                        "global_index": g_idx,
                        "row_index": g_idx,
                        "block_id": item["block_id"],
                        "item_index": item["item_index"],
                        "cell_id": item["cell_id"],
                        "step_index": step_idx,
                        "best_letter": best_l,
                        "worst_letter": worst_l,
                        "q2_by_letter": q2_by_letter,
                    },
                )

            elif method == "2afc":
                choice = str(payload.get("choice", payload.get("choice_letter", ""))).strip().upper()
                if choice not in letters:
                    raise ValueError(f"2AFC choice must be one of {letters}")
                other = [let for let in letters if let != choice][0]
                self._append_event(
                    mode,
                    {
                        "event": "item_complete",
                        "global_index": g_idx,
                        "row_index": g_idx,
                        "block_id": item["block_id"],
                        "item_index": item["item_index"],
                        "cell_id": item["cell_id"],
                        "method": method,
                        "choice": choice,
                        "ranks_by_letter": {choice: 1, other: 2},
                        "q2_by_letter": q2_by_letter,
                        "flawed": flawed,
                        "notes": notes,
                    },
                )

            elif method == "pairwise":
                raw_choice = str(payload.get("choice", payload.get("choice_letter", ""))).strip()
                choice_up = raw_choice.upper()
                if choice_up in letters:
                    other = [let for let in letters if let != choice_up][0]
                    ranks = {choice_up: 1, other: 2}
                    outcome = choice_up
                elif raw_choice.lower() == "unsure" and item["allow_unsure"]:
                    ranks = {let: 1 for let in letters}
                    outcome = "unsure"
                else:
                    allowed = list(letters) + (["unsure"] if item["allow_unsure"] else [])
                    raise ValueError(f"Pairwise choice must be one of {allowed}")
                self._append_event(
                    mode,
                    {
                        "event": "item_complete",
                        "global_index": g_idx,
                        "row_index": g_idx,
                        "block_id": item["block_id"],
                        "item_index": item["item_index"],
                        "cell_id": item["cell_id"],
                        "method": method,
                        "choice": outcome,
                        "ranks_by_letter": ranks,
                        "q2_by_letter": q2_by_letter,
                        "flawed": flawed,
                        "notes": notes,
                    },
                )

            elif method == "mos_likert":
                raw_likert = payload.get("likert_by_letter")
                valid_scores = {float(pt["score"]) for pt in (item["likert_scale"] or [])}
                likert_by_letter: Dict[str, float] = {}
                if isinstance(raw_likert, dict):
                    for let in letters:
                        if let not in raw_likert:
                            raise ValueError(f"MOS/Likert score required for Clip {let}")
                        sc = float(raw_likert[let])
                        if valid_scores and sc not in valid_scores:
                            raise ValueError(f"Invalid Likert score {sc}; allowed: {sorted(valid_scores)}")
                        likert_by_letter[let] = sc
                elif "score" in payload and len(letters) == 1:
                    sc = float(payload["score"])
                    if valid_scores and sc not in valid_scores:
                        raise ValueError(f"Invalid Likert score {sc}; allowed: {sorted(valid_scores)}")
                    likert_by_letter[letters[0]] = sc
                else:
                    raise ValueError("MOS/Likert submission requires 'likert_by_letter' or 'score'")

                # Derive relative ranks from Likert scores (higher score = better rank)
                sorted_lets = sorted(letters, key=lambda let: -likert_by_letter[let])
                ranks = {let: idx + 1 for idx, let in enumerate(sorted_lets)}
                self._append_event(
                    mode,
                    {
                        "event": "item_complete",
                        "global_index": g_idx,
                        "row_index": g_idx,
                        "block_id": item["block_id"],
                        "item_index": item["item_index"],
                        "cell_id": item["cell_id"],
                        "method": method,
                        "likert_by_letter": likert_by_letter,
                        "ranks_by_letter": ranks,
                        "q2_by_letter": q2_by_letter,
                        "flawed": flawed,
                        "notes": notes,
                    },
                )

            elif method == "abx":
                choice = str(payload.get("choice", payload.get("choice_letter", ""))).strip().upper()
                if choice not in ("A", "B"):
                    raise ValueError("ABX choice must be 'A' or 'B' (which reference X matches)")
                self._append_event(
                    mode,
                    {
                        "event": "item_complete",
                        "global_index": g_idx,
                        "row_index": g_idx,
                        "block_id": item["block_id"],
                        "item_index": item["item_index"],
                        "cell_id": item["cell_id"],
                        "method": method,
                        "choice": choice,
                        "q2_by_letter": q2_by_letter,
                        "flawed": flawed,
                        "notes": notes,
                    },
                )

        return self.get_public_state(mode=mode)

    def record_undo(self, mode: str = "live", global_index: Optional[int] = None) -> Dict[str, Any]:
        with self.lock:
            mode = "pilot" if mode == "pilot" else "live"
            ev: Dict[str, Any] = {"event": "undo"}
            if global_index is not None:
                ev["global_index"] = int(global_index)
            self._append_event(mode, ev)
        return self.get_public_state(mode=mode)

    def record_flag_flawed(
        self,
        global_index: int,
        flawed: bool = True,
        reason: str = "",
        mode: str = "live",
    ) -> Dict[str, Any]:
        with self.lock:
            mode = "pilot" if mode == "pilot" else "live"
            self._append_event(
                mode,
                {
                    "event": "flag_flawed",
                    "global_index": int(global_index),
                    "flawed": bool(flawed),
                    "reason": str(reason).strip(),
                },
            )
        return self.get_public_state(mode=mode, requested_global_index=global_index)

    def record_note(self, global_index: int, notes: str, mode: str = "live") -> Dict[str, Any]:
        with self.lock:
            mode = "pilot" if mode == "pilot" else "live"
            self._append_event(
                mode,
                {
                    "event": "note",
                    "global_index": int(global_index),
                    "notes": str(notes).strip(),
                },
            )
        return self.get_public_state(mode=mode, requested_global_index=global_index)

    def export_blind_csv(self, mode: str = "live") -> str:
        """Export blind ratings CSV (anonymized clip letters only, never arm names)."""
        with self.lock:
            mode = "pilot" if mode == "pilot" else "live"
            states = self.item_states_by_mode[mode]
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow([
                "global_index",
                "block_id",
                "item_number",
                "prompt_id",
                "method",
                "completed",
                "ranks_by_letter",
                "choice_outcome",
                "q2_by_letter",
                "likert_by_letter",
                "flawed",
                "notes",
                "map_hash",
            ])
            for idx, item in enumerate(self.flat_items):
                st = states[idx]
                srow = self.sealed_map["rows"][idx]
                writer.writerow([
                    idx,
                    item["block_id"],
                    item["item_number_in_block"],
                    item["prompt_id"],
                    item["method"],
                    st["completed"],
                    json.dumps(st["ranks_by_letter"], sort_keys=True),
                    st["choice_outcome"] or "",
                    json.dumps(st["q2_by_letter"], sort_keys=True),
                    json.dumps(st["likert_by_letter"], sort_keys=True),
                    st["flawed"],
                    st["notes"],
                    srow["map_hash"],
                ])
            return buf.getvalue()

    def export_blind_jsonl(self, mode: str = "live") -> str:
        with self.lock:
            mode = "pilot" if mode == "pilot" else "live"
            lines = [json.dumps(ev, separators=(",", ":")) for ev in self.events_by_mode[mode]]
            return "\n".join(lines) + ("\n" if lines else "")

    def get_clip_file_path(self, global_index: int, letter: str) -> Optional[str]:
        with self.lock:
            if global_index < 0 or global_index >= len(self.sealed_map["rows"]):
                return None
            srow = self.sealed_map["rows"][global_index]
            return srow.get("letter_to_path", {}).get(letter.upper())


class BlindPanelRequestHandler(BaseHTTPRequestHandler):
    session: BlindPanelSession
    static_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str, status: int = HTTPStatus.OK, filename: Optional[str] = None) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str, content_type: str) -> None:
        if not os.path.isfile(path):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            try:
                spec = range_header.split("=", 1)[1].strip()
                start_s, end_s = spec.split("-", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else file_size - 1
                end = min(end, file_size - 1)
                if start > end or start < 0:
                    raise ValueError("Invalid byte range")
                length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return
            except Exception:
                pass

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        mode = qs.get("mode", ["live"])[0]

        # Explicitly block any attempt to read sealed_map or unblinded files over HTTP
        lowered = path.lower()
        if "sealed_map" in lowered or "sealed_store" in lowered or lowered.endswith(".sealed.jsonl") or lowered.endswith(".bin"):
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        if path in ("/", "/index.html"):
            self._send_file(os.path.join(self.static_dir, "index.html"), "text/html; charset=utf-8")
            return
        if path == "/styles.css":
            self._send_file(os.path.join(self.static_dir, "styles.css"), "text/css; charset=utf-8")
            return
        if path == "/app.js":
            self._send_file(os.path.join(self.static_dir, "app.js"), "application/javascript; charset=utf-8")
            return

        if path == "/api/state":
            req_idx = int(qs["global_index"][0]) if "global_index" in qs else None
            self._send_json(self.session.get_public_state(mode=mode, requested_global_index=req_idx))
            return

        if path == "/api/operator/preflight":
            self._send_json(self.session.get_preflight_status())
            return

        if path == "/api/operator/audit":
            self._send_json(self.session.get_operator_audit())
            return

        if path == "/api/export/csv":
            csv_data = self.session.export_blind_csv(mode=mode)
            fname = f"{self.session.study_id}_{mode}_blind.csv"
            self._send_text(csv_data, "text/csv; charset=utf-8", filename=fname)
            return

        if path == "/api/export/jsonl":
            jsonl_data = self.session.export_blind_jsonl(mode=mode)
            fname = f"{self.session.study_id}_{mode}_answers.jsonl"
            self._send_text(jsonl_data, "application/x-ndjson; charset=utf-8", filename=fname)
            return

        if path.startswith("/clip/") and path.endswith(".mp4"):
            parts = path.strip("/").split("/")
            if len(parts) == 3:
                try:
                    g_idx = int(parts[1])
                    letter = parts[2][:-4].upper()
                    clip_path = self.session.get_clip_file_path(g_idx, letter)
                    if clip_path:
                        self._send_file(clip_path, "video/mp4")
                        return
                except ValueError:
                    pass
            self.send_error(HTTPStatus.NOT_FOUND, "Clip not found")
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        mode = str(payload.get("mode", "live"))
        try:
            if path == "/api/session/start":
                self._send_json(self.session.start_session(mode=mode))
            elif path == "/api/session/resume_break":
                self._send_json(self.session.resume_after_break(mode=mode))
            elif path == "/api/session/start_block2":
                self._send_json(self.session.start_block2(mode=mode))
            elif path == "/api/session/decline_block2":
                self._send_json(self.session.decline_block2(mode=mode))
            elif path in ("/api/submit", "/api/step"):
                self._send_json(self.session.submit_rating(payload))
            elif path == "/api/undo":
                g_idx = payload.get("global_index", payload.get("row_index"))
                self._send_json(self.session.record_undo(mode=mode, global_index=g_idx))
            elif path == "/api/flag_flawed":
                g_idx = int(payload.get("global_index", payload.get("row_index", 0)))
                self._send_json(
                    self.session.record_flag_flawed(
                        global_index=g_idx,
                        flawed=bool(payload.get("flawed", True)),
                        reason=str(payload.get("reason", "")),
                        mode=mode,
                    )
                )
            elif path == "/api/note":
                g_idx = int(payload.get("global_index", payload.get("row_index", 0)))
                self._send_json(
                    self.session.record_note(
                        global_index=g_idx,
                        notes=str(payload.get("notes", "")),
                        mode=mode,
                    )
                )
            elif path == "/api/operator/override":
                reason = str(payload.get("reason", ""))
                self._send_json(self.session.record_preflight_override(reason))
            else:
                self._send_json({"error": "Unknown endpoint"}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def create_http_server(
    session: BlindPanelSession,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    class BoundHandler(BlindPanelRequestHandler):
        pass

    BoundHandler.session = session
    return ThreadingHTTPServer((host, port), BoundHandler)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Start the local blindpanel evaluation server.")
    parser.add_argument("--config", required=True, help="Path to study config JSON")
    parser.add_argument(
        "--sealed-map",
        default=None,
        help="Path to write/read sealed_map_<date>.bin (default: sealed_store/sealed_map_<date>.bin)",
    )
    parser.add_argument(
        "--answers",
        default="answers.jsonl",
        help="Path to write/read live answers.jsonl",
    )
    parser.add_argument(
        "--pilot-answers",
        default=None,
        help="Path to write/read pilot_answers_<date>.jsonl",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (must be 127.0.0.1 / localhost)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (0 for random free port)")
    parser.add_argument(
        "--verify-clip-shape",
        action="store_true",
        help="Also verify width/height/frames/fps with ffprobe at load",
    )
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost"):
        print("REFUSED: blindpanel binds only to 127.0.0.1 or localhost.", file=sys.stderr)
        return 2

    date_tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    sealed_map_path = args.sealed_map or os.path.join("sealed_store", f"sealed_map_{date_tag}.bin")

    try:
        session = BlindPanelSession(
            config_path=args.config,
            sealed_map_path=sealed_map_path,
            answers_path=args.answers,
            pilot_answers_path=args.pilot_answers,
            verify_clip_shape=args.verify_clip_shape,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    httpd = create_http_server(session, host=args.host, port=args.port)
    actual_port = httpd.server_address[1]
    print(f"blindpanel listening on http://{args.host}:{actual_port}")
    print(f"  Rater View      : http://{args.host}:{actual_port}/?view=rater")
    print(f"  Pilot View      : http://{args.host}:{actual_port}/?view=pilot")
    print(f"  Pre-Flight Gate : http://{args.host}:{actual_port}/?view=preflight")
    print(f"  Operator Audit  : http://{args.host}:{actual_port}/?view=audit")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
