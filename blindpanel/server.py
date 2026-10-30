"""server.py — Local-only HTTP server and sealed session engine for blindpanel.

Enforces Rules 1–6 (Card tg, Card sk, and Card tm):
- Committed study configs contain ONLY cell ids, prompts, rater cues, block and method settings, and timing rules.
- At session start, a NEW cryptographic secret seed is generated (`secrets.token_bytes(32)`, 256 bits >= 128 bits).
  It is NEVER printed, logged, or written outside the gitignored `sealed_store/`.
- Optional `--arm-map <path>` maps config pseudonyms (`ARM_1..ARM_4`) to local clip filename prefixes. It is read
  once at session start into the sealed store and is never served, logged, or rendered on any route or page.
- From that secret seed come:
  1. Per-item slot orders and Latin-square / 6-permutation counterbalancing (`generate_session_slot_orders`)
  2. Per-session salted clip names (`clip_<salted_token>.mp4`)
- Clips are served under those salted names (`/clip/<salted_name>`) from a **sealed manifest**
  (`arm`, `original_sha256`, `salted_name`) stored inside the sealed store, which is NEVER sent to the browser.
- Every clip is verified against the sealed manifest's `original_sha256` at load (`REFUSED AT LOAD` on mismatch).
- Separate rater and operator surfaces (Card tm item 1):
  - Why separate ports: Hosting operator endpoints (`/api/operator/*`) and exports (`/api/export/*`, which carry
    `cell_id`s with `-warmup` and `-rep2` suffixes) on the same port as the rater surface exposes study internals
    to any accidental link, script, or URL navigation in the rater's browser. Binding the operator surface to a
    separate localhost port—started only when `--operator-port` is explicitly passed and gated by a per-start
    cryptographic token printed solely to the operator's terminal—guarantees by construction that the rater port
    has zero code paths for operator metadata or session exports (`404 Not Found` with and without a token).
- Test hooks out of the production build (Card tm item 3):
  - `?unlock=1`, `window.__blindpanel`, and test helpers exist ONLY when the server is started with the explicit
    `--enable-test-hooks` flag (`enable_test_hooks=True`), which the browser cannot set. In production mode
    (the default), `/app.js` contains zero references to them.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import hmac
import io
import json
import os
import secrets
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlparse

from blindpanel.cosmos3_study import discover_clips_in_dir
from blindpanel.schema import (
    generate_session_slot_orders,
    letters_for_count,
    validate_and_normalize_config,
    verify_bws3_counterbalancing,
)
from blindpanel.unblind import (
    canonical_map_hash,
    canonical_sealed_manifest_sha256,
    load_jsonl_safe,
    reconstruct_session_answers,
)

# Explicit route tables for the Rater port and the separate Operator port (used by handlers and crawl tests)
RATER_GET_ROUTES: Tuple[str, ...] = (
    "/",
    "/index.html",
    "/styles.css",
    "/app.js",
    "/api/state",
)

RATER_POST_ROUTES: Tuple[str, ...] = (
    "/api/session/start",
    "/api/session/resume_break",
    "/api/session/start_block2",
    "/api/session/decline_block2",
    "/api/submit",
    "/api/step",
    "/api/undo",
    "/api/flag_flawed",
    "/api/note",
)

OPERATOR_GET_ROUTES: Tuple[str, ...] = (
    "/api/operator/preflight",
    "/api/operator/audit",
    "/api/export/csv",
    "/api/export/jsonl",
)

OPERATOR_POST_ROUTES: Tuple[str, ...] = (
    "/api/operator/override",
)

_TEST_HOOKS_SLOT_MARKER = "// __BLINDPANEL_TEST_HOOKS_SLOT__"

_TEST_HOOKS_JS_SNIPPET = """
  if (params.get("unlock") === "1") {
    firstPlayCompleted = true;
  }
  window.__blindpanel = {
    unlockVotingForTest: () => {
      firstPlayCompleted = true;
      refreshSelectionAndLayoutInPlace();
    },
    setSelectionsForTest: (best, worst, q2Map, notesText) => {
      firstPlayCompleted = true;
      draftBest = best;
      draftWorst = worst;
      if (q2Map) draftQ2 = Object.assign({}, q2Map);
      if (notesText != null) {
        draftNotes = notesText;
        const inp = document.getElementById("notes-input");
        if (inp) inp.value = notesText;
      }
      refreshSelectionAndLayoutInPlace();
    },
    setLayoutMode: (mode, flipLetter) => {
      manualLayoutOverride = mode;
      if (flipLetter) activeFlipLetter = flipLetter;
      preserveVideoTimeAcrossRerender(() => renderActiveRatingStage(serverState));
    },
  };
"""


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


def _derive_salted_clip_name(secret_seed_hex: str, global_index: int, arm_name: str, role_tag: str = "slot") -> str:
    """Derive a per-session salted clip filename from the session secret seed (never reveals arm, index, or letter)."""
    digest = hmac.new(
        bytes.fromhex(secret_seed_hex),
        f"blindpanel-clip:{global_index}:{arm_name}:{role_tag}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:24]
    return f"clip_{digest}.mp4"


def _load_arm_map_at_session_start(arm_map_path: Optional[str]) -> Optional[Dict[str, str]]:
    """Read `--arm-map <path>` once at session start. Returns a dict mapping config arm pseudonyms
    to local clip filename prefixes."""
    if not arm_map_path:
        return None
    abs_map_path = os.path.abspath(arm_map_path)
    if not os.path.isfile(abs_map_path):
        raise ValueError(f"REFUSED AT LOAD: arm-map file not found: {abs_map_path}")
    with open(abs_map_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    if not isinstance(raw_map, dict) or not raw_map:
        raise ValueError("REFUSED AT LOAD: arm-map file must be a non-empty JSON object mapping arm pseudonyms to prefixes")
    cleaned: Dict[str, str] = {}
    for k, v in raw_map.items():
        ks = str(k).strip()
        vs = str(v).strip()
        if not ks or not vs:
            raise ValueError("REFUSED AT LOAD: arm-map keys and values must be non-empty strings")
        cleaned[ks] = vs
    return cleaned


class BlindPanelSession:
    """Thread-safe state manager for live and pilot blindpanel sessions."""

    def __init__(
        self,
        config_path: str,
        sealed_map_path: str,
        answers_path: str,
        clips_dir: Optional[str] = None,
        arm_map_path: Optional[str] = None,
        pilot_answers_path: Optional[str] = None,
        audit_log_path: Optional[str] = None,
        verify_clip_shape: bool = False,
    ) -> None:
        self.lock = threading.Lock()
        self.config_path = os.path.abspath(config_path)
        self.sealed_map_path = os.path.abspath(sealed_map_path)
        self.answers_path = os.path.abspath(answers_path)
        self.clips_dir = os.path.abspath(clips_dir) if clips_dir else None

        # Read `--arm-map` once at session start; stored only inside the sealed store and never exposed on any route
        loaded_arm_map = _load_arm_map_at_session_start(arm_map_path)

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
        self.clip_spec: Dict[str, Any] = self.config["clip_spec"]
        self.blocks: List[Dict[str, Any]] = self.config["blocks"]
        self.flat_items: List[Dict[str, Any]] = self.config["flat_items"]

        # Map salted_name -> resolved local file path (in memory only, never exposed to browser)
        self.salted_clip_routes: Dict[str, str] = {}
        self.verified_clip_count = 0

        self._hydrate_item_clips(arm_map=loaded_arm_map)
        self.sealed_map = self._init_or_load_sealed_store(arm_map=loaded_arm_map)
        self._verify_all_clips_against_sealed_manifest()

        # Runtime state for Live ("live") and Pilot ("pilot") modes
        self.events_by_mode: Dict[str, List[Dict[str, Any]]] = {"live": [], "pilot": []}
        self.item_states_by_mode: Dict[str, Dict[int, Dict[str, Any]]] = {"live": {}, "pilot": {}}
        self.cursor_by_mode: Dict[str, int] = {"live": 0, "pilot": 0}
        self.interstitial_by_mode: Dict[str, Optional[str]] = {"live": "session_start", "pilot": "session_start"}
        self.block2_declined_by_mode: Dict[str, bool] = {"live": False, "pilot": False}

        # Load audit events (operator pre-flight overrides, etc.)
        self.audit_events, _ = load_jsonl_safe(self.audit_log_path)

        # Resume live and pilot logs safely
        self._load_or_init_mode("live", self.answers_path)
        self._load_or_init_mode("pilot", self.pilot_answers_path)

    def _resolve_clip_path(self, rel_or_abs: str) -> str:
        if os.path.isabs(rel_or_abs) and os.path.exists(rel_or_abs):
            return os.path.abspath(rel_or_abs)
        if self.clips_dir:
            cand_clips = os.path.abspath(os.path.join(self.clips_dir, rel_or_abs))
            if os.path.exists(cand_clips):
                return cand_clips
        cand = os.path.abspath(os.path.join(os.path.dirname(self.config_path), rel_or_abs))
        if os.path.exists(cand):
            return cand
        return os.path.abspath(rel_or_abs)

    def _hydrate_item_clips(self, arm_map: Optional[Dict[str, str]] = None) -> None:
        """Resolve clip files for each item either from explicit item['arms'] or by discovering clips in `clips_dir`."""
        discovery_cache: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Dict[str, str]]] = {}
        for item in self.flat_items:
            blk = self.blocks[item["block_index"]]
            blk_arms = list(blk["arms"])
            if item["arms"] is not None:
                for _arm_name, spec in item["arms"].items():
                    resolved = self._resolve_clip_path(spec["path"])
                    spec["resolved_path"] = resolved
            else:
                if not self.clips_dir:
                    raise ValueError(
                        f"REFUSED AT LOAD: config '{os.path.basename(self.config_path)}' is a study design config "
                        "without per-item clip paths; pass --clips-dir <dir> to locate the session clips."
                    )
                cache_key = (item["base_cell_id"], tuple(blk_arms))
                if cache_key not in discovery_cache:
                    try:
                        discovery_cache[cache_key] = discover_clips_in_dir(
                            self.clips_dir,
                            arms=blk_arms,
                            base_cell_id=item["base_cell_id"],
                            arm_map=arm_map,
                        )
                    except FileNotFoundError as exc:
                        raise ValueError(f"REFUSED AT LOAD: missing clip file: {exc}") from exc
                discovered = discovery_cache[cache_key]
                item["arms"] = {
                    arm_name: {
                        "path": discovered[arm_name]["path"],
                        "resolved_path": discovered[arm_name]["path"],
                        "sha256": discovered[arm_name]["sha256"].lower(),
                    }
                    for arm_name in blk_arms
                }

    def _init_or_load_sealed_store(self, arm_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate or resume the sealed store (`sealed_map_*.bin`) and its sealed manifest.

        Rule 2 & Card tm item 4:
        - At session start, a NEW secret seed is generated (`secrets.token_bytes(32)`, 256 bits >= 128 bits).
          It is never printed, logged, or written outside the sealed store (gitignored).
        - If `--arm-map` was provided at session start, the mapping is stored ONLY inside `sealed_map["arm_map"]`
          in the gitignored sealed store and never served or logged.
        - From `secret_seed` come the per-item slot orders, the counterbalancing, and the per-session salted clip names.
        - Clips are served under those salted names from a sealed manifest (`arm`, `original_sha256`, `salted_name`),
          which is never sent to the browser.
        """
        if os.path.exists(self.sealed_map_path):
            with open(self.sealed_map_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if (
                existing.get("config_sha256") == self.config_sha256
                and len(existing.get("rows", [])) == len(self.flat_items)
                and isinstance(existing.get("sealed_manifest"), list)
            ):
                return existing

        # Generate a fresh 256-bit (>= 128-bit) cryptographic secret seed via `secrets`
        secret_seed_bytes = secrets.token_bytes(32)
        secret_seed_hex = secret_seed_bytes.hex()

        all_study_arms: List[str] = []
        block_slot_orders: Dict[int, Dict[int, List[str]]] = {}
        for blk in self.blocks:
            for a in blk["arms"]:
                if a not in all_study_arms:
                    all_study_arms.append(a)
            slot_orders = generate_session_slot_orders(blk, secret_seed_hex)
            if blk.get("enforce_counterbalance_bws3", False):
                verify_bws3_counterbalancing(
                    blk["items"],
                    arms=blk["arms"],
                    block_id=blk["block_id"],
                    slot_orders_by_item_idx=slot_orders,
                )
            block_slot_orders[blk["block_index"]] = slot_orders

        sealed_rows: List[Dict[str, Any]] = []
        sealed_manifest: List[Dict[str, Any]] = []

        for item in self.flat_items:
            g_idx = item["global_index"]
            b_idx = item["block_index"]
            i_idx = item["item_index"]
            blk_arms = list(self.blocks[b_idx]["arms"])
            letters = letters_for_count(len(blk_arms))

            shuffled_arms = list(block_slot_orders[b_idx][i_idx])
            letter_to_arm = {letters[i]: shuffled_arms[i] for i in range(len(letters))}
            arm_to_letter = {shuffled_arms[i]: letters[i] for i in range(len(letters))}

            row_salt = hmac.new(
                secret_seed_bytes,
                f"blindpanel-row-salt:{g_idx}".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()[:32]
            mhash = canonical_map_hash(letter_to_arm, row_salt)

            abx_target = item.get("abx_target_arm")
            if item["method"] == "abx" and not abx_target:
                pick_byte = hmac.new(
                    secret_seed_bytes,
                    f"blindpanel-abx:{g_idx}".encode("utf-8"),
                    hashlib.sha256,
                ).digest()[0]
                abx_target = shuffled_arms[pick_byte % len(shuffled_arms)]

            letter_to_salted_name: Dict[str, str] = {}
            for let, arm_name in letter_to_arm.items():
                salted_name = _derive_salted_clip_name(secret_seed_hex, g_idx, arm_name, role_tag=f"slot-{let}")
                orig_sha = item["arms"][arm_name]["sha256"].lower()
                letter_to_salted_name[let] = salted_name
                sealed_manifest.append({
                    "global_index": g_idx,
                    "cell_id": item["cell_id"],
                    "letter": let,
                    "arm": arm_name,
                    "original_sha256": orig_sha,
                    "salted_name": salted_name,
                })

            if item["method"] == "abx" and abx_target:
                salted_x = _derive_salted_clip_name(secret_seed_hex, g_idx, abx_target, role_tag="slot-X")
                orig_sha_x = item["arms"][abx_target]["sha256"].lower()
                letter_to_salted_name["X"] = salted_x
                sealed_manifest.append({
                    "global_index": g_idx,
                    "cell_id": item["cell_id"],
                    "letter": "X",
                    "arm": abx_target,
                    "original_sha256": orig_sha_x,
                    "salted_name": salted_x,
                })

            sealed_rows.append({
                "global_index": g_idx,
                "row_index": g_idx,
                "block_id": item["block_id"],
                "block_index": b_idx,
                "item_index": i_idx,
                "item_number_in_block": item["item_number_in_block"],
                "cell_id": item["cell_id"],
                "base_cell_id": item["base_cell_id"],
                "prompt_id": item["prompt_id"],
                "prompt_type": item["prompt_type"],
                "variant_id": item["variant_id"],
                "prompt": item["prompt"],
                "method": item["method"],
                "is_warmup": item["is_warmup"],
                "is_repeat": item["is_repeat"],
                "repeat_of_cell_id": item["repeat_of_cell_id"],
                "ask_q2": item["ask_q2"],
                "abx_target_arm": abx_target,
                "salt": row_salt,
                "map_hash": mhash,
                "letter_to_arm": letter_to_arm,
                "arm_to_letter": arm_to_letter,
                "letter_to_salted_name": letter_to_salted_name,
            })

        manifest_sha256 = canonical_sealed_manifest_sha256(sealed_manifest)
        smap: Dict[str, Any] = {
            "schema": "blindpanel-sealed-map-v1",
            "study_id": self.study_id,
            "arena": self.study_id,
            "title": self.title,
            "config_sha256": self.config_sha256,
            "sealed_manifest_sha256": manifest_sha256,
            "secret_seed": secret_seed_hex,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "arms": all_study_arms,
            "sealed_manifest": sealed_manifest,
            "rows": sealed_rows,
        }
        if arm_map is not None:
            smap["arm_map"] = dict(arm_map)

        sealed_dir = os.path.dirname(self.sealed_map_path)
        if sealed_dir:
            os.makedirs(sealed_dir, exist_ok=True)
        with open(self.sealed_map_path, "w", encoding="utf-8") as f:
            json.dump(smap, f, indent=2)
        return smap

    def _verify_all_clips_against_sealed_manifest(self) -> None:
        """Verify every clip exists on disk, matches its `original_sha256` in the sealed manifest,
        and (optionally) matches `clip_spec` geometry/frame count via ffprobe."""
        checked_cache: Dict[str, str] = {}
        expected_w = int(self.clip_spec.get("width", 832))
        expected_h = int(self.clip_spec.get("height", 480))
        expected_frames = int(self.clip_spec.get("frames", 61))
        expected_fps = str(self.clip_spec.get("fps", "24.0000"))

        manifest_entries = self.sealed_map.get("sealed_manifest", [])
        if not manifest_entries:
            raise ValueError("REFUSED AT LOAD: sealed_manifest is empty")

        self.salted_clip_routes.clear()
        for entry in manifest_entries:
            g_idx = int(entry["global_index"])
            arm_name = str(entry["arm"])
            salted_name = str(entry["salted_name"])
            expected_sha = str(entry["original_sha256"]).lower()

            item = self.flat_items[g_idx]
            cid = item["cell_id"]
            if arm_name not in item["arms"]:
                raise ValueError(f"REFUSED AT LOAD: arm '{arm_name}' not found on item {g_idx} ({cid})")
            resolved = item["arms"][arm_name]["resolved_path"]
            if not os.path.isfile(resolved):
                raise ValueError(
                    f"REFUSED AT LOAD: missing clip file for item {g_idx} ({cid}) arm '{arm_name}': {resolved}"
                )

            if resolved not in checked_cache:
                actual_sha = sha256_file(resolved).lower()
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"REFUSED AT LOAD: sha256 mismatch against sealed manifest for item {g_idx} ({cid}) "
                        f"arm '{arm_name}': expected {expected_sha}, got {actual_sha}"
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
                        f"REFUSED AT LOAD: sha256 mismatch against sealed manifest for item {g_idx} ({cid}) "
                        f"arm '{arm_name}': expected {expected_sha}, got {checked_cache[resolved]}"
                    )

            self.salted_clip_routes[salted_name] = resolved

        self.verified_clip_count = len(checked_cache)

    def _load_or_init_mode(self, mode: str, path: str) -> None:
        events, truncated = load_jsonl_safe(path)
        if truncated > 0:
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
    # Pre-Flight & Audit methods (Operator surface only)
    # ------------------------------------------------------------------
    def get_preflight_status(self) -> Dict[str, Any]:
        """Build operator pre-flight report. Served ONLY on the separate operator port."""
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
        cell_ids, expected durations, or sealed manifest mappings. Clips are referenced solely by
        their per-session salted clip names (`/clip/<salted_name>`).
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
            l2salted = srow["letter_to_salted_name"]

            letters = list(st["letters"])
            clips = [
                {
                    "letter": let,
                    "url": f"/clip/{l2salted[let]}",
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
                    "A": f"/clip/{l2salted['A']}",
                    "B": f"/clip/{l2salted['B']}",
                    "X": f"/clip/{l2salted['X']}",
                }

            block_items = blk["items"]
            block_completed = sum(
                1 for bit in block_items if states[bit["global_index"]]["completed"]
            )

            # Figures for the interstitial cards (display follow-ups): the block before the active one, and
            # every block whose items are all rated. Totals are the same `num_items` the session counter uses
            # (warm-ups and repeats included, never broken out).
            prev_block_total_items = (
                int(self.blocks[item["block_index"] - 1]["num_items"]) if item["block_index"] > 0 else None
            )
            completed_blocks = [
                {"block_number": b_idx + 1, "total_items": int(b["num_items"])}
                for b_idx, b in enumerate(self.blocks)
                if b["items"] and all(states[bit["global_index"]]["completed"] for bit in b["items"])
            ]

            can_undo = any(
                ev.get("event") in ("item_complete", "step") for ev in self.events_by_mode[mode]
            )

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
                "prev_block_total_items": prev_block_total_items,
                "completed_blocks": completed_blocks,
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
        """Export blind ratings CSV (Operator port only)."""
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
        """Export raw session event log JSONL (Operator port only; carries cell_id)."""
        with self.lock:
            mode = "pilot" if mode == "pilot" else "live"
            lines = [json.dumps(ev, separators=(",", ":")) for ev in self.events_by_mode[mode]]
            return "\n".join(lines) + ("\n" if lines else "")

    def get_clip_by_salted_name(self, salted_name: str) -> Optional[str]:
        with self.lock:
            return self.salted_clip_routes.get(salted_name)


def _render_rater_app_js(static_dir: str, enable_test_hooks: bool = False) -> str:
    """Return the rater JS bundle. In production (`enable_test_hooks=False`), strips the test-hook slot
    so zero references to `?unlock=1` or `window.__blindpanel` exist in the served JS."""
    raw_js_path = os.path.join(static_dir, "app.js")
    with open(raw_js_path, "r", encoding="utf-8") as f:
        src = f.read()
    if enable_test_hooks:
        return src.replace(_TEST_HOOKS_SLOT_MARKER, _TEST_HOOKS_JS_SNIPPET)
    return src.replace(_TEST_HOOKS_SLOT_MARKER, "")


class _BaseBlindPanelHTTPHandler(BaseHTTPRequestHandler):
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

    def _send_text(
        self,
        text: str,
        content_type: str,
        status: int = HTTPStatus.OK,
        filename: Optional[str] = None,
    ) -> None:
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


class BlindPanelRequestHandler(_BaseBlindPanelHTTPHandler):
    """Rater-surface HTTP handler.

    Serves ONLY rater static assets (`/`, `/index.html`, `/styles.css`, `/app.js`),
    strictly blind `/api/state`, rater session/rating POST endpoints, and salted clips (`/clip/<salted_name>`).
    Returns `404 Not Found` on all `/api/operator/*`, `/api/export/*`, answers files, and sealed files
    both with and without a token.
    """

    enable_test_hooks: bool = False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        mode = qs.get("mode", ["live"])[0]

        # Explicitly block any attempt to read sealed_map, sealed_manifest, answers, operator, or export routes
        lowered = path.lower()
        if (
            "sealed_map" in lowered
            or "sealed_manifest" in lowered
            or "sealed_store" in lowered
            or "arm_map" in lowered
            or "arm-map" in lowered
            or "answers" in lowered
            or lowered.startswith("/api/operator")
            or lowered.startswith("/api/export")
            or lowered.endswith(".jsonl")
            or lowered.endswith(".bin")
        ):
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

        if path in ("/", "/index.html"):
            self._send_file(os.path.join(self.static_dir, "index.html"), "text/html; charset=utf-8")
            return
        if path == "/styles.css":
            self._send_file(os.path.join(self.static_dir, "styles.css"), "text/css; charset=utf-8")
            return
        if path == "/app.js":
            js_body = _render_rater_app_js(self.static_dir, enable_test_hooks=self.enable_test_hooks)
            self._send_text(js_body, "application/javascript; charset=utf-8")
            return

        if path == "/api/state":
            req_idx = int(qs["global_index"][0]) if "global_index" in qs else None
            self._send_json(self.session.get_public_state(mode=mode, requested_global_index=req_idx))
            return

        if path.startswith("/clip/") and path.endswith(".mp4"):
            parts = path.strip("/").split("/")
            if len(parts) == 2:
                salted_name = parts[1]
                clip_path = self.session.get_clip_by_salted_name(salted_name)
                if clip_path:
                    self._send_file(clip_path, "video/mp4")
                    return
            self.send_error(HTTPStatus.NOT_FOUND, "Clip not found")
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        # Rater port serves no /api/operator/* at all: 404 with and without a token
        if path.lower().startswith("/api/operator") or path.lower().startswith("/api/export"):
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return

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
            else:
                self._send_json({"error": "Unknown endpoint"}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


class BlindPanelOperatorRequestHandler(_BaseBlindPanelHTTPHandler):
    """Operator-surface HTTP handler running on a separate localhost port.

    Every request must carry the per-start `operator_token` (via `X-Operator-Token` header,
    `Authorization: Bearer <token>`, or `?token=<token>`). Requests without a valid token are
    refused with `403 Forbidden`.
    """

    operator_token: str = ""

    def _check_operator_auth(self, qs: Dict[str, List[str]]) -> bool:
        expected = self.operator_token
        if not expected:
            return False
        hdr_tok = (self.headers.get("X-Operator-Token") or "").strip()
        if hdr_tok and hmac.compare_digest(hdr_tok, expected):
            return True
        auth_hdr = (self.headers.get("Authorization") or "").strip()
        if auth_hdr.lower().startswith("bearer "):
            bearer_tok = auth_hdr[7:].strip()
            if bearer_tok and hmac.compare_digest(bearer_tok, expected):
                return True
        qs_tok = (qs.get("token", [""])[0] or qs.get("operator_token", [""])[0]).strip()
        if qs_tok and hmac.compare_digest(qs_tok, expected):
            return True
        return False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if not self._check_operator_auth(qs):
            self._send_json({"error": "Forbidden: valid operator token required"}, status=HTTPStatus.FORBIDDEN)
            return

        mode = qs.get("mode", ["live"])[0]
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

        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        if not self._check_operator_auth(qs):
            self._send_json({"error": "Forbidden: valid operator token required"}, status=HTTPStatus.FORBIDDEN)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            if path == "/api/operator/override":
                reason = str(payload.get("reason", ""))
                self._send_json(self.session.record_preflight_override(reason))
            else:
                self._send_json({"error": "Unknown operator endpoint"}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)


def create_http_server(
    session: BlindPanelSession,
    host: str = "127.0.0.1",
    port: int = 8765,
    enable_test_hooks: bool = False,
) -> ThreadingHTTPServer:
    """Create the Rater-surface HTTP server."""
    class BoundRaterHandler(BlindPanelRequestHandler):
        pass

    BoundRaterHandler.session = session
    BoundRaterHandler.enable_test_hooks = bool(enable_test_hooks)
    return ThreadingHTTPServer((host, port), BoundRaterHandler)


def create_operator_http_server(
    session: BlindPanelSession,
    host: str = "127.0.0.1",
    port: int = 0,
    operator_token: Optional[str] = None,
) -> Tuple[ThreadingHTTPServer, str]:
    """Create the separate Operator-surface HTTP server and its per-start cryptographic token."""
    token = operator_token or secrets.token_hex(24)

    class BoundOperatorHandler(BlindPanelOperatorRequestHandler):
        pass

    BoundOperatorHandler.session = session
    BoundOperatorHandler.operator_token = token
    return ThreadingHTTPServer((host, port), BoundOperatorHandler), token


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Start the local blindpanel evaluation server.")
    parser.add_argument("--config", required=True, help="Path to study config JSON")
    parser.add_argument(
        "--clips-dir",
        default=None,
        help="Directory containing clip MP4 files when --config is a public study design config",
    )
    parser.add_argument(
        "--arm-map",
        default=None,
        help="Path to local gitignored JSON file mapping config arm pseudonyms (ARM_1..ARM_4) to clip filename prefixes",
    )
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
    parser.add_argument("--port", type=int, default=8765, help="Rater bind port (0 for random free port)")
    parser.add_argument(
        "--operator-port",
        type=int,
        default=None,
        help="Optional separate localhost port for the operator surface (preflight, audit, exports)",
    )
    parser.add_argument(
        "--enable-test-hooks",
        action="store_true",
        help="Enable ?unlock=1 and window.__blindpanel in served /app.js (test harness only; never on by default)",
    )
    parser.add_argument(
        "--verify-clip-shape",
        action="store_true",
        help="Also verify width/height/frames/fps with ffprobe at load",
    )
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost"):
        print("REFUSED: blindpanel binds only to 127.0.0.1 or localhost.", file=sys.stderr)
        return 2

    if args.operator_port is not None and args.operator_port != 0 and args.operator_port == args.port:
        print("REFUSED: --operator-port must be distinct from the rater --port.", file=sys.stderr)
        return 2

    date_tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    sealed_map_path = args.sealed_map or os.path.join("sealed_store", f"sealed_map_{date_tag}.bin")

    try:
        session = BlindPanelSession(
            config_path=args.config,
            sealed_map_path=sealed_map_path,
            answers_path=args.answers,
            clips_dir=args.clips_dir,
            arm_map_path=args.arm_map,
            pilot_answers_path=args.pilot_answers,
            verify_clip_shape=args.verify_clip_shape,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    httpd = create_http_server(
        session,
        host=args.host,
        port=args.port,
        enable_test_hooks=args.enable_test_hooks,
    )
    actual_port = httpd.server_address[1]
    print(f"blindpanel rater server listening on http://{args.host}:{actual_port}")
    print(f"  Rater View : http://{args.host}:{actual_port}/?view=rater")
    print(f"  Pilot View : http://{args.host}:{actual_port}/?view=pilot")

    op_httpd: Optional[ThreadingHTTPServer] = None
    if args.operator_port is not None:
        op_httpd, op_token = create_operator_http_server(
            session,
            host=args.host,
            port=args.operator_port,
        )
        actual_op_port = op_httpd.server_address[1]
        op_thread = threading.Thread(target=op_httpd.serve_forever, daemon=True)
        op_thread.start()
        print(f"blindpanel operator server listening on http://{args.host}:{actual_op_port}")
        print(f"  Operator Token (terminal only): {op_token}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if op_httpd is not None:
            op_httpd.shutdown()
            op_httpd.server_close()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
