"""schema.py — Versioned configuration schema validator and session-time counterbalancing generator for blindpanel.

Enforces Rule 1 & Rule 2:
- Committed study configs hold ONLY cell ids, prompts, rater cues, block and method settings, and timing rules.
  They never contain per-item slot/arm orders, arm-to-hash/path maps, seeds/salts, or absolute paths.
- At session start, a fresh >= 128-bit cryptographic secret seed (`secrets.token_bytes(32)`) is generated
  (never printed, logged, or written outside the gitignored sealed store).
- From that secret seed come the per-item slot orders, the Latin-square / 6-permutation counterbalancing,
  and the per-session salted clip names.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from typing import Any, Dict, List, Optional, Set, Tuple

SUPPORTED_SCHEMA_VERSIONS: Set[str] = {"blindpanel-config-v1", "1.0"}
SUPPORTED_METHODS: Set[str] = {"bws_3", "bws", "2afc", "pairwise", "mos_likert", "abx"}

# Exact Amendment A Question 2 wording and answers for manipulation prompts (M1-M4)
AMENDMENT_A_Q2_PROMPT = "How far does the robot get with the action within the 2.5 s clip?"
AMENDMENT_A_Q2_OPTIONS: List[Dict[str, Any]] = [
    {"id": "yes", "label": "Yes (completes it)", "score": 1.0, "key": "Y"},
    {"id": "partly", "label": "Partly (clearly under way)", "score": 0.5, "key": "P"},
    {"id": "no", "label": "No (not achieved)", "score": 0.0, "key": "N"},
]

DEFAULT_LIKERT_SCALE: List[Dict[str, Any]] = [
    {"score": 5, "label": "5 — Excellent"},
    {"score": 4, "label": "4 — Good"},
    {"score": 3, "label": "3 — Fair"},
    {"score": 2, "label": "2 — Poor"},
    {"score": 1, "label": "1 — Bad"},
]

LETTERS_POOL = [chr(ord("A") + i) for i in range(26)]


def letters_for_count(k: int) -> List[str]:
    if k < 1 or k > 26:
        raise ValueError(f"Number of slots must be between 1 and 26, got {k}")
    return LETTERS_POOL[:k]


def all_permutations_of_arms(arms: List[str]) -> List[Tuple[str, ...]]:
    return list(itertools.permutations(arms))


def validate_question2_spec(q2: Dict[str, Any], ctx: str = "question2") -> Dict[str, Any]:
    if not isinstance(q2, dict):
        raise ValueError(f"{ctx}: must be a JSON object")
    prompt = q2.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{ctx}: missing non-empty 'prompt'")
    options = q2.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise ValueError(f"{ctx}: 'options' must be a list of at least 2 choices")
    norm_options: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for idx, opt in enumerate(options):
        if not isinstance(opt, dict):
            raise ValueError(f"{ctx}.options[{idx}]: must be an object")
        oid = str(opt.get("id", "")).strip()
        label = str(opt.get("label", "")).strip()
        if not oid or not label:
            raise ValueError(f"{ctx}.options[{idx}]: must provide non-empty 'id' and 'label'")
        if oid in seen_ids:
            raise ValueError(f"{ctx}.options[{idx}]: duplicate option id '{oid}'")
        seen_ids.add(oid)
        score = float(opt.get("score", 0.0))
        norm_options.append({
            "id": oid,
            "label": label,
            "score": score,
            "key": str(opt.get("key", oid[0].upper()))[:1].upper(),
        })
    return {
        "id": str(q2.get("id", "action_completion")),
        "prompt": prompt.strip(),
        "per_clip": bool(q2.get("per_clip", True)),
        "options": norm_options,
    }


def validate_and_normalize_config(
    cfg: Dict[str, Any],
    forbid_study_blinding_data: bool = False,
) -> Dict[str, Any]:
    """Validate a blindpanel config dictionary at load and return a normalized structure.

    Accepts both:
    - Sealed-manifest / public study design configs (items have only cell_id, prompt_id, prompt_type,
      prompt, rater_cue, timing rules, and block settings — zero per-item arms, paths, hashes, or slot orders)
    - Locally generated session configs (gitignored) with resolved clip references.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a JSON object")

    if forbid_study_blinding_data:
        for forbidden_top in ("rng_seed", "session_seed", "salt", "secret_seed"):
            if forbidden_top in cfg:
                raise ValueError(f"Public study config must not contain '{forbidden_top}'")

    schema_version = str(cfg.get("schema_version", "blindpanel-config-v1"))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version '{schema_version}'. Supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    study_id = str(cfg.get("study_id", cfg.get("arena", "blindpanel-study"))).strip()
    if not study_id:
        raise ValueError("Config must specify a non-empty 'study_id' or 'title'")
    title = str(cfg.get("title", study_id)).strip()

    clip_spec = cfg.get("clip_spec") or {
        "width": 832,
        "height": 480,
        "frames": 61,
        "fps": "24.0000",
        "seconds": 2.54,
    }
    max_clip_seconds = float(clip_spec.get("seconds", 2.54))
    clip_filename_template = str(cfg.get("clip_filename_template", "{arm}_{base_cell_id}.mp4"))

    default_q2 = cfg.get("question2_spec")
    norm_default_q2: Optional[Dict[str, Any]] = None
    if default_q2 is not None:
        norm_default_q2 = validate_question2_spec(default_q2, "question2_spec")
    else:
        norm_default_q2 = {
            "id": "action_completion",
            "prompt": AMENDMENT_A_Q2_PROMPT,
            "per_clip": True,
            "options": list(AMENDMENT_A_Q2_OPTIONS),
        }

    raw_blocks = cfg.get("blocks")
    if raw_blocks is None:
        raw_items = cfg.get("items", cfg.get("rows"))
        if not isinstance(raw_items, list) or len(raw_items) == 0:
            raise ValueError("Config must provide non-empty 'blocks' or 'items'/'rows'")
        arms = cfg.get("arms")
        if not arms and isinstance(raw_items[0].get("arms"), dict):
            arms = list(raw_items[0]["arms"].keys())
        method = str(cfg.get("method", "bws_3" if arms and len(arms) == 3 else "bws"))
        raw_blocks = [
            {
                "block_id": str(cfg.get("block_id", "block1")),
                "title": title,
                "optional": False,
                "method": method,
                "arms": arms,
                "allow_unsure": bool(cfg.get("allow_unsure", False)),
                "allow_skip": bool(cfg.get("allow_skip", False)),
                "require_full_play": bool(cfg.get("require_full_play", True)),
                "break_after_item": cfg.get("break_after_item"),
                "items": raw_items,
            }
        ]

    if not isinstance(raw_blocks, list) or len(raw_blocks) == 0:
        raise ValueError("Config 'blocks' must be a non-empty list")

    norm_blocks: List[Dict[str, Any]] = []
    seen_block_ids: Set[str] = set()
    flat_items: List[Dict[str, Any]] = []
    global_idx = 0

    for b_idx, blk in enumerate(raw_blocks):
        if not isinstance(blk, dict):
            raise ValueError(f"blocks[{b_idx}]: must be an object")
        block_id = str(blk.get("block_id", f"block{b_idx + 1}")).strip()
        if not block_id:
            raise ValueError(f"blocks[{b_idx}]: empty 'block_id'")
        if block_id in seen_block_ids:
            raise ValueError(f"Duplicate block_id '{block_id}'")
        seen_block_ids.add(block_id)

        method = str(blk.get("method", "")).strip()
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Block '{block_id}': unsupported method '{method}'. Supported: {sorted(SUPPORTED_METHODS)}"
            )

        arms = blk.get("arms")
        if arms is None and "num_arms" in blk:
            k_count = int(blk["num_arms"])
            arms = [f"ARM_{idx + 1}" for idx in range(k_count)]
        if not isinstance(arms, list) or len(arms) < 1 or not all(isinstance(a, str) and a.strip() for a in arms):
            raise ValueError(f"Block '{block_id}': 'arms' must be a non-empty list of arm name strings")
        arms = [a.strip() for a in arms]
        if len(set(arms)) != len(arms):
            raise ValueError(f"Block '{block_id}': duplicate arm names in {arms}")

        k_arms = len(arms)
        if method == "bws_3" and k_arms != 3:
            raise ValueError(f"Block '{block_id}': method 'bws_3' requires exactly 3 arms, got {k_arms}")
        if method == "bws" and k_arms < 3:
            raise ValueError(f"Block '{block_id}': method 'bws' requires >= 3 arms, got {k_arms}")
        if method in ("2afc", "pairwise", "abx") and k_arms != 2:
            raise ValueError(f"Block '{block_id}': method '{method}' requires 2 arms, got {k_arms}")

        allow_unsure = bool(blk.get("allow_unsure", False))
        allow_skip = bool(blk.get("allow_skip", False))
        require_full_play = bool(blk.get("require_full_play", True))
        optional_block = bool(blk.get("optional", False))
        break_after_item = blk.get("break_after_item")
        if break_after_item is not None:
            break_after_item = int(break_after_item)

        blk_q2 = blk.get("question2_spec")
        norm_blk_q2 = validate_question2_spec(blk_q2, f"Block '{block_id}'.question2_spec") if blk_q2 else norm_default_q2

        likert_scale = blk.get("likert_scale", DEFAULT_LIKERT_SCALE)
        if method == "mos_likert":
            if not isinstance(likert_scale, list) or len(likert_scale) < 2:
                raise ValueError(f"Block '{block_id}': 'mos_likert' requires a 'likert_scale' list of >= 2 points")

        items = blk.get("items", blk.get("rows"))
        if not isinstance(items, list) or len(items) == 0:
            raise ValueError(f"Block '{block_id}': must contain a non-empty 'items' list")

        norm_items: List[Dict[str, Any]] = []
        for i_idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Block '{block_id}' item {i_idx}: must be an object")

            if forbid_study_blinding_data:
                for forbidden_item_key in ("arms", "slot" + "_arms", "arm_order", "sha256", "path"):
                    if forbidden_item_key in item:
                        raise ValueError(
                            f"Block '{block_id}' item {i_idx}: public study config must not contain '{forbidden_item_key}'"
                        )

            item_arms_raw = item.get("arms")
            norm_item_arms: Optional[Dict[str, Dict[str, str]]] = None
            if item_arms_raw is not None:
                if not isinstance(item_arms_raw, dict):
                    raise ValueError(f"Block '{block_id}' item {i_idx}: 'arms' must be a dict when provided")
                if set(item_arms_raw.keys()) != set(arms):
                    missing = sorted(set(arms) - set(item_arms_raw.keys()))
                    extra = sorted(set(item_arms_raw.keys()) - set(arms))
                    raise ValueError(
                        f"Block '{block_id}' item {i_idx}: arms mismatch. Missing={missing}, Extra={extra}"
                    )
                norm_item_arms = {}
                for arm_name in arms:
                    spec = item_arms_raw[arm_name]
                    if not isinstance(spec, dict) or "path" not in spec or "sha256" not in spec:
                        raise ValueError(
                            f"Block '{block_id}' item {i_idx} arm '{arm_name}': must specify 'path' and 'sha256'"
                        )
                    norm_item_arms[arm_name] = {
                        "path": str(spec["path"]),
                        "sha256": str(spec["sha256"]).strip().lower(),
                    }

            prompt_id = str(item.get("prompt_id", f"P{i_idx + 1}")).strip()
            prompt_type = str(item.get("prompt_type", item.get("type", "general"))).strip()
            variant_id = int(item.get("variant_id", item.get("seed", 1)))
            cell_id = str(item.get("cell_id", f"{prompt_id}-v{variant_id}")).strip()
            base_cell_id = str(item.get("base_cell_id", item.get("repeat_of_cell_id") or cell_id)).strip()
            for suffix in ("-warmup", "-rep2", "-b2-warmup", "-b2-rep2", "-b2"):
                if base_cell_id.endswith(suffix):
                    base_cell_id = base_cell_id[: -len(suffix)]
                    break

            prompt_text = str(item.get("prompt", "")).strip()
            rater_cue = str(item.get("rater_cue", item.get("cue", "the main subject through the clip"))).strip()

            is_warmup = bool(item.get("is_warmup", False))
            is_repeat = bool(item.get("is_repeat", False))
            repeat_of_cell_id = item.get("repeat_of_cell_id", base_cell_id if is_repeat else None)

            ask_q2 = bool(item.get("ask_q2", prompt_type == "manipulation" and method == "bws_3"))
            item_q2 = norm_blk_q2 if ask_q2 else None

            abx_target_arm = item.get("abx_target_arm")
            if method == "abx" and abx_target_arm is not None and abx_target_arm not in arms:
                raise ValueError(f"Block '{block_id}' item {i_idx}: 'abx_target_arm' must be one of {arms}")

            est_seconds = item.get("estimated_seconds")
            est_max_seconds = (
                float(item["estimated_max_seconds"])
                if "estimated_max_seconds" in item and item["estimated_max_seconds"] is not None
                else None
            )
            clip_sec = float(item.get("clip_seconds", max_clip_seconds))
            preflight_flagged = bool(
                item.get("preflight_flagged", False)
                or (est_max_seconds is not None and est_max_seconds > clip_sec)
            )

            norm_item = {
                "global_index": global_idx,
                "block_id": block_id,
                "block_index": b_idx,
                "item_index": i_idx,
                "item_number_in_block": i_idx + 1,
                "block_total_items": len(items),
                "cell_id": cell_id,
                "base_cell_id": base_cell_id,
                "prompt_id": prompt_id,
                "prompt_type": prompt_type,
                "variant_id": variant_id,
                "prompt": prompt_text,
                "rater_cue": rater_cue,
                "method": method,
                "is_warmup": is_warmup,
                "is_repeat": is_repeat,
                "repeat_of_cell_id": repeat_of_cell_id,
                "ask_q2": ask_q2,
                "question2_spec": item_q2,
                "allow_unsure": allow_unsure,
                "allow_skip": allow_skip,
                "require_full_play": require_full_play,
                "likert_scale": likert_scale if method == "mos_likert" else None,
                "arms": norm_item_arms,
                "abx_target_arm": abx_target_arm,
                "required_action": str(item.get("required_action", "continuous / none")),
                "estimated_seconds": str(est_seconds) if est_seconds is not None else "n/a (continuous)",
                "estimated_max_seconds": est_max_seconds,
                "clip_seconds": clip_sec,
                "preflight_flagged": preflight_flagged,
                "break_after": (break_after_item is not None and (i_idx + 1) == break_after_item),
            }
            norm_items.append(norm_item)
            flat_items.append(norm_item)
            global_idx += 1

        if blk.get("enforce_no_adjacent_prompt", False):
            verify_no_adjacent_prompts(norm_items, block_id=block_id)

        norm_blocks.append({
            "block_id": block_id,
            "block_index": b_idx,
            "title": str(blk.get("title", f"Block {b_idx + 1}")),
            "optional": optional_block,
            "method": method,
            "arms": arms,
            "allow_unsure": allow_unsure,
            "allow_skip": allow_skip,
            "require_full_play": require_full_play,
            "break_after_item": break_after_item,
            "enforce_no_adjacent_prompt": bool(blk.get("enforce_no_adjacent_prompt", False)),
            "enforce_counterbalance_bws3": bool(blk.get("enforce_counterbalance_bws3", False)),
            "question2_spec": norm_blk_q2,
            "likert_scale": likert_scale if method == "mos_likert" else None,
            "num_items": len(norm_items),
            "num_warmup": sum(1 for it in norm_items if it["is_warmup"]),
            "num_recorded": sum(1 for it in norm_items if not it["is_warmup"]),
            "num_repeats": sum(1 for it in norm_items if it["is_repeat"] and not it["is_warmup"]),
            "items": norm_items,
        })

    extra_preflight = cfg.get("preflight_flagged_rows", [])
    if not isinstance(extra_preflight, list):
        raise ValueError("'preflight_flagged_rows' must be a list if provided")

    return {
        "schema_version": schema_version,
        "study_id": study_id,
        "title": title,
        "clip_spec": clip_spec,
        "clip_filename_template": clip_filename_template,
        "question2_spec": norm_default_q2,
        "blocks": norm_blocks,
        "flat_items": flat_items,
        "preflight_flagged_rows": extra_preflight,
    }


def generate_session_slot_orders(
    block: Dict[str, Any],
    secret_seed_hex: str,
) -> Dict[int, List[str]]:
    """Derive counterbalanced per-item slot orders from the session's >= 128-bit secret seed.

    - For a 36-core-triple `bws_3` block (or `enforce_counterbalance_bws3=True`), uses `secret_seed_hex`
      to randomize the arm labeling and prompt-within-type ordering while guaranteeing:
        1. All 6 permutations of the 3 arms appear 6 times each overall and 2 times per prompt_type.
        2. Every prompt's 3 variants form a 3x3 Latin square over slots A, B, C.
        3. Every hidden repeat uses a cyclic slot shift `(A, B, C) -> (B, C, A)` from Pass 1.
    - For a 36-core-pair `2afc` block, uses `secret_seed_hex` to assign balanced Left/Right orders
      (18 AB / 18 BA, 6 per prompt_type) and swaps Left/Right on hidden repeats.
    - For arbitrary custom blocks, shuffles each item's arms from `secret_seed_hex` and shifts repeats.
    """
    block_id = block["block_id"]
    arms = list(block["arms"])
    items = block["items"]
    seed_int = int.from_bytes(
        hashlib.sha256(f"blindpanel-slots:{secret_seed_hex}:{block_id}".encode("utf-8")).digest(),
        "big",
    )
    rng = random.Random(seed_int)

    core_items = [it for it in items if not it["is_warmup"] and not it["is_repeat"]]
    slot_orders_by_item_idx: Dict[int, List[str]] = {}

    if block["method"] == "bws_3" and len(arms) == 3 and len(core_items) == 36:
        # Randomize base arm ordering from the session secret seed
        perm_arms = list(arms)
        rng.shuffle(perm_arms)
        a0, a1, a2 = perm_arms[0], perm_arms[1], perm_arms[2]
        even_latin = [
            [a0, a1, a2],
            [a1, a2, a0],
            [a2, a0, a1],
        ]
        odd_latin = [
            [a0, a2, a1],
            [a2, a1, a0],
            [a1, a0, a2],
        ]
        if rng.random() < 0.5:
            even_latin, odd_latin = odd_latin, even_latin

        # Group core items by prompt_type -> prompt_id -> sorted items
        by_type: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for it in core_items:
            by_type.setdefault(it["prompt_type"], {}).setdefault(it["prompt_id"], []).append(it)

        core_perm_by_base_cell: Dict[str, List[str]] = {}
        for _ptype, pmap in sorted(by_type.items()):
            pids = sorted(pmap.keys())
            rng.shuffle(pids)
            for p_offset, pid in enumerate(pids):
                family = even_latin if (p_offset % 4) < 2 else odd_latin
                shift = p_offset % 2
                p_items = sorted(pmap[pid], key=lambda x: (x["variant_id"], x["cell_id"]))
                variant_indices = list(range(len(p_items)))
                rng.shuffle(variant_indices)
                for idx_in_p, it in enumerate(p_items):
                    v_slot = variant_indices[idx_in_p]
                    perm = list(family[(v_slot + shift) % 3])
                    slot_orders_by_item_idx[it["item_index"]] = perm
                    core_perm_by_base_cell[it["base_cell_id"]] = perm
                    core_perm_by_base_cell[it["cell_id"]] = perm

        # Assign warm-ups and hidden repeats (cyclic shift from Pass 1)
        for it in items:
            i_idx = it["item_index"]
            if i_idx in slot_orders_by_item_idx:
                continue
            ref_cid = it.get("repeat_of_cell_id") or it["base_cell_id"]
            base_perm = core_perm_by_base_cell.get(ref_cid)
            if base_perm is None:
                base_perm = list(perm_arms)
                rng.shuffle(base_perm)
            # Cyclic shift (A, B, C) -> (B, C, A)
            shifted = base_perm[1:] + base_perm[:1]
            slot_orders_by_item_idx[i_idx] = shifted

        return slot_orders_by_item_idx

    if block["method"] in ("2afc", "pairwise") and len(arms) == 2 and len(core_items) == 36:
        by_type_2: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for it in core_items:
            by_type_2.setdefault(it["prompt_type"], {}).setdefault(it["prompt_id"], []).append(it)

        base_patterns = [
            [True, False, True],
            [False, True, False],
            [True, True, False],
            [False, False, True],
        ]
        core_pair_by_base_cell: Dict[str, List[str]] = {}
        for _ptype, pmap in sorted(by_type_2.items()):
            pids = sorted(pmap.keys())
            rng.shuffle(pids)
            pats = list(base_patterns)
            rng.shuffle(pats)
            for p_offset, pid in enumerate(pids):
                pat = list(pats[p_offset % len(pats)])
                p_items = sorted(pmap[pid], key=lambda x: (x["variant_id"], x["cell_id"]))
                for idx_in_p, it in enumerate(p_items):
                    first_left = pat[idx_in_p % len(pat)]
                    order = [arms[0], arms[1]] if first_left else [arms[1], arms[0]]
                    slot_orders_by_item_idx[it["item_index"]] = order
                    core_pair_by_base_cell[it["base_cell_id"]] = order
                    core_pair_by_base_cell[it["cell_id"]] = order

        for it in items:
            i_idx = it["item_index"]
            if i_idx in slot_orders_by_item_idx:
                continue
            ref_cid = it.get("repeat_of_cell_id") or it["base_cell_id"]
            base_order = core_pair_by_base_cell.get(ref_cid)
            if base_order is None:
                base_order = list(arms)
                rng.shuffle(base_order)
            slot_orders_by_item_idx[i_idx] = [base_order[1], base_order[0]]

        return slot_orders_by_item_idx

    # General fallback for arbitrary item counts / methods
    first_pass_by_cell: Dict[str, List[str]] = {}
    for it in items:
        i_idx = it["item_index"]
        if it["is_repeat"]:
            ref_cid = str(it.get("repeat_of_cell_id") or it["base_cell_id"])
            if ref_cid in first_pass_by_cell and len(first_pass_by_cell[ref_cid]) >= 2:
                prev = first_pass_by_cell[ref_cid]
                slot_orders_by_item_idx[i_idx] = prev[1:] + prev[:1]
                continue
        shuffled = list(arms)
        rng.shuffle(shuffled)
        slot_orders_by_item_idx[i_idx] = shuffled
        if not it["is_warmup"]:
            first_pass_by_cell[it["cell_id"]] = shuffled
            first_pass_by_cell[it["base_cell_id"]] = shuffled

    return slot_orders_by_item_idx


def verify_no_adjacent_prompts(items: List[Dict[str, Any]], block_id: str = "") -> None:
    """Verify ITU-T P.913 §11.7.4 constraint: no prompt_id appears on two consecutive trials."""
    for idx in range(1, len(items)):
        prev_p = items[idx - 1]["prompt_id"]
        cur_p = items[idx]["prompt_id"]
        if prev_p == cur_p:
            raise ValueError(
                f"Block '{block_id}': consecutive duplicate prompt_id '{cur_p}' at item indices {idx - 1} and {idx}"
            )


def verify_bws3_counterbalancing(
    items: List[Dict[str, Any]],
    arms: List[str],
    block_id: str = "block1",
    min_repeat_gap: int = 12,
    slot_orders_by_item_idx: Optional[Dict[int, List[str]]] = None,
) -> Dict[str, Any]:
    """Verify Amendment A §1.2 counterbalancing rules on a 3-arm BWS block:
    1. Across the 36 core recorded triples, all 6 permutations of the 3 arms appear 6 times each
       (and 2 times per prompt_type across manipulation, locomotion, scene).
    2. Within every prompt's 3 variants, the 3 assigned permutations form a Latin square over slots A, B, C.
    3. Hidden repeats are separated by at least `min_repeat_gap` intervening trials and use a shifted
       permutation from Pass 1.
    """
    core_items = [it for it in items if not it["is_warmup"] and not it["is_repeat"]]
    repeat_items = [it for it in items if not it["is_warmup"] and it["is_repeat"]]

    if len(core_items) != 36:
        raise ValueError(f"Block '{block_id}': expected 36 core triples for bws_3 counterbalance check, got {len(core_items)}")

    perms_all = all_permutations_of_arms(arms)
    perm_counts: Dict[Tuple[str, ...], int] = {p: 0 for p in perms_all}
    type_perm_counts: Dict[str, Dict[Tuple[str, ...], int]] = {}
    prompt_slots: Dict[str, List[Tuple[str, ...]]] = {}
    first_pass_by_cell: Dict[str, Tuple[int, Tuple[str, ...]]] = {}

    def _get_slots(item: Dict[str, Any]) -> List[str]:
        if slot_orders_by_item_idx is not None and item["item_index"] in slot_orders_by_item_idx:
            return slot_orders_by_item_idx[item["item_index"]]
        return list(item.get("slot" + "_arms") or [])

    for it in core_items:
        ordered_arms = _get_slots(it)
        if not ordered_arms or len(ordered_arms) != 3:
            raise ValueError(f"Block '{block_id}' cell '{it['cell_id']}': missing 3-element slot order")
        perm = tuple(ordered_arms)
        if perm not in perm_counts:
            raise ValueError(f"Block '{block_id}' cell '{it['cell_id']}': invalid permutation {perm}")
        perm_counts[perm] += 1
        ptype = it["prompt_type"]
        type_perm_counts.setdefault(ptype, {p: 0 for p in perms_all})[perm] += 1
        prompt_slots.setdefault(it["prompt_id"], []).append(perm)
        first_pass_by_cell[it["cell_id"]] = (it["item_index"], perm)
        first_pass_by_cell[it["base_cell_id"]] = (it["item_index"], perm)

    for perm, count in perm_counts.items():
        if count != 6:
            raise ValueError(f"Block '{block_id}': permutation {perm} appears {count} times across core triples (expected 6)")
    for ptype, pmap in type_perm_counts.items():
        for perm, count in pmap.items():
            if count != 2:
                raise ValueError(
                    f"Block '{block_id}' type '{ptype}': permutation {perm} appears {count} times (expected 2)"
                )

    for pid, plist in prompt_slots.items():
        if len(plist) != 3:
            raise ValueError(f"Block '{block_id}' prompt '{pid}': expected 3 variants, got {len(plist)}")
        for slot_idx in range(3):
            arms_in_slot = {p[slot_idx] for p in plist}
            if arms_in_slot != set(arms):
                raise ValueError(
                    f"Block '{block_id}' prompt '{pid}': slot {slot_idx} does not form a Latin square ({plist})"
                )

    for rep in repeat_items:
        cid = rep.get("repeat_of_cell_id") or rep["base_cell_id"] or rep["cell_id"]
        if cid not in first_pass_by_cell:
            raise ValueError(f"Block '{block_id}': repeat cell '{cid}' has no matching core pass")
        first_idx, first_perm = first_pass_by_cell[cid]
        intervening = rep["item_index"] - first_idx - 1
        if intervening < min_repeat_gap:
            raise ValueError(
                f"Block '{block_id}' repeat '{cid}': only {intervening} intervening items (< {min_repeat_gap})"
            )
        rep_perm = tuple(_get_slots(rep))
        if rep_perm == first_perm:
            raise ValueError(f"Block '{block_id}' repeat '{cid}': repeat uses identical permutation {rep_perm}")

    return {
        "core_triples": len(core_items),
        "repeat_triples": len(repeat_items),
        "permutation_counts": {"-".join(k): v for k, v in perm_counts.items()},
    }
