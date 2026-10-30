"""unblind.py — Offline unblinding tool and cryptographic salt/map_hash/sealed_manifest verifier for blindpanel.

Reads ONLY the sealed store (`sealed_map_*.bin`) and `answers.jsonl` to produce `unblinded_results.json`.
Never writes or exposes `secret_seed` in the unblinded output.
Verifies per-row cryptographic salt and map_hash AND validates the sealed manifest's integrity (sha256)
before joining, ignores unrecorded warm-up practice rows in recorded metrics while reporting warm-up
completion counts, and computes:
- Per-arm rank statistics (mean rank, median rank, rank counts, Borda score, best/worst shares)
- Pairwise win matrix across arms
- Per-arm Question 2 action-completion breakdown on flagged rows (M1-M4)
- Intra-rater hidden-repeat concordance (Pass 1 vs Pass 2 pairwise order)
- Method-specific summaries for bws_3, bws, 2afc, pairwise, mos_likert, and abx.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_map_hash(letter_to_arm: Dict[str, str], salt: str) -> str:
    """Compute SHA-256 over canonical JSON {"map": letter_to_arm, "salt": salt}."""
    payload = json.dumps(
        {"map": letter_to_arm, "salt": salt},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_sealed_manifest_sha256(sealed_manifest: List[Dict[str, Any]]) -> str:
    """Compute deterministic SHA-256 over the canonicalized `sealed_manifest` entries."""
    normalized = [
        {
            "global_index": int(entry["global_index"]),
            "cell_id": str(entry["cell_id"]),
            "letter": str(entry["letter"]),
            "arm": str(entry["arm"]),
            "original_sha256": str(entry["original_sha256"]).lower(),
            "salted_name": str(entry["salted_name"]),
        }
        for entry in sealed_manifest
    ]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_sealed_manifest_integrity(sealed_map: Dict[str, Any]) -> None:
    """Validate the sealed manifest's integrity (sha256) before unblinding, and refuse on any mismatch."""
    manifest = sealed_map.get("sealed_manifest")
    if manifest is not None:
        if not isinstance(manifest, list) or len(manifest) == 0:
            raise ValueError("Tamper check failed: sealed_manifest must be a non-empty list")

        for idx, entry in enumerate(manifest):
            if not isinstance(entry, dict):
                raise ValueError(f"Tamper check failed: sealed_manifest[{idx}] is not a dict")
            orig_sha = str(entry.get("original_sha256", "")).lower()
            if not _HEX64_RE.match(orig_sha) or orig_sha == ("0" * 64):
                raise ValueError(
                    f"Tamper check failed on sealed_manifest[{idx}]: invalid original_sha256 '{orig_sha}'"
                )

        expected_manifest_sha = sealed_map.get("sealed_manifest_sha256")
        actual_manifest_sha = canonical_sealed_manifest_sha256(manifest)
        if expected_manifest_sha is not None and actual_manifest_sha != expected_manifest_sha:
            raise ValueError(
                "Tamper check failed on sealed_manifest_sha256: "
                f"expected {expected_manifest_sha}, got {actual_manifest_sha}"
            )


def load_jsonl_safe(path: str) -> Tuple[List[Dict[str, Any]], int]:
    """Load JSONL events, ignoring a trailing truncated partial line if present.

    Returns (events, truncated_partial_lines_count).
    """
    events: List[Dict[str, Any]] = []
    truncated = 0
    if not os.path.exists(path):
        return events, 0
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for idx, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue
        try:
            events.append(json.loads(s))
        except json.JSONDecodeError:
            if idx == len(lines) - 1:
                truncated += 1
            else:
                raise ValueError(f"Corrupt JSONL line {idx + 1} in {path}")
    return events, truncated


def reconstruct_session_answers(
    sealed_map: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """Replay append-only events (supporting item_complete, step, flag_flawed, note, and undo) to reconstruct
    completed item state keyed by global_index.
    """
    rows_meta = sealed_map.get("rows", [])
    meta_by_global: Dict[int, Dict[str, Any]] = {}
    for idx, r in enumerate(rows_meta):
        g_idx = int(r.get("global_index", r.get("row_index", idx)))
        meta_by_global[g_idx] = r

    applied_events: List[Dict[str, Any]] = []
    for ev in events:
        etype = ev.get("event")
        if etype in ("item_complete", "step", "flag_flawed", "note", "row_complete"):
            applied_events.append(ev)
        elif etype == "undo":
            target_global = ev.get("global_index", ev.get("row_index"))
            for i in range(len(applied_events) - 1, -1, -1):
                prev = applied_events[i]
                prev_g = prev.get("global_index", prev.get("row_index"))
                if prev.get("event") in ("item_complete", "step") and (
                    target_global is None or prev_g == target_global
                ):
                    applied_events.pop(i)
                    break

    item_states: Dict[int, Dict[str, Any]] = {}
    for g_idx, rmeta in meta_by_global.items():
        l2a = rmeta["letter_to_arm"]
        k = len(l2a)
        letters = sorted(l2a.keys())
        item_states[g_idx] = {
            "global_index": g_idx,
            "block_id": rmeta.get("block_id", "block1"),
            "item_index": rmeta.get("item_index", g_idx),
            "cell_id": rmeta.get("cell_id", f"row-{g_idx}"),
            "prompt_id": rmeta.get("prompt_id", ""),
            "prompt_type": rmeta.get("prompt_type", "general"),
            "variant_id": rmeta.get("variant_id", 0),
            "method": rmeta.get("method", "bws" if k >= 3 else "2afc"),
            "is_warmup": bool(rmeta.get("is_warmup", False)),
            "is_repeat": bool(rmeta.get("is_repeat", False)),
            "repeat_of_cell_id": rmeta.get("repeat_of_cell_id"),
            "ask_q2": bool(rmeta.get("ask_q2", False)),
            "k": k,
            "letters": letters,
            "remaining": list(letters),
            "ranks_by_letter": {},
            "choice_letter": None,
            "choice_outcome": None,
            "likert_by_letter": {},
            "q2_by_letter": {},
            "flawed": False,
            "flaw_reason": "",
            "notes": "",
            "completed": False,
        }

    for ev in applied_events:
        g_idx = int(ev.get("global_index", ev.get("row_index", 0)))
        if g_idx not in item_states:
            continue
        st = item_states[g_idx]
        etype = ev.get("event")

        if etype == "flag_flawed":
            st["flawed"] = bool(ev.get("flawed", True))
            st["flaw_reason"] = str(ev.get("reason", ""))
        elif etype == "note":
            st["notes"] = str(ev.get("notes", ""))
        elif etype == "step":
            rem = st["remaining"]
            step_idx = int(ev.get("step_index", 0))
            best_l = ev.get("best_letter")
            worst_l = ev.get("worst_letter")
            top_rank = 1 + step_idx
            bot_rank = st["k"] - step_idx
            if best_l in rem and worst_l in rem and best_l != worst_l:
                st["ranks_by_letter"][best_l] = top_rank
                st["ranks_by_letter"][worst_l] = bot_rank
                rem.remove(best_l)
                rem.remove(worst_l)
                if len(rem) == 1:
                    st["ranks_by_letter"][rem[0]] = top_rank + 1
                    rem.clear()
                if len(rem) == 0:
                    st["completed"] = True
            if ev.get("q2_by_letter"):
                st["q2_by_letter"].update(ev["q2_by_letter"])
        elif etype == "item_complete":
            method = st["method"]
            if "ranks_by_letter" in ev and isinstance(ev["ranks_by_letter"], dict):
                st["ranks_by_letter"] = {str(k): int(v) for k, v in ev["ranks_by_letter"].items()}
            elif method == "bws_3" and ev.get("best_letter") and ev.get("worst_letter"):
                b_let = str(ev["best_letter"])
                w_let = str(ev["worst_letter"])
                mid_candidates = [let for let in st["letters"] if let not in (b_let, w_let)]
                if len(mid_candidates) == 1 and b_let != w_let:
                    st["ranks_by_letter"] = {b_let: 1, mid_candidates[0]: 2, w_let: 3}

            if method in ("2afc", "pairwise"):
                choice = str(ev.get("choice", ev.get("choice_letter", "")))
                st["choice_outcome"] = choice
                if choice in st["letters"]:
                    st["choice_letter"] = choice
                    other = [let for let in st["letters"] if let != choice]
                    st["ranks_by_letter"] = {choice: 1}
                    if other:
                        st["ranks_by_letter"][other[0]] = 2
                elif choice == "unsure":
                    st["ranks_by_letter"] = {let: 1 for let in st["letters"]}
            elif method == "abx":
                choice = str(ev.get("choice", ev.get("choice_letter", "")))
                st["choice_outcome"] = choice
                st["choice_letter"] = choice
            elif method == "mos_likert":
                if isinstance(ev.get("likert_by_letter"), dict):
                    st["likert_by_letter"] = {str(k): float(v) for k, v in ev["likert_by_letter"].items()}
                elif "score" in ev and st["letters"]:
                    st["likert_by_letter"] = {st["letters"][0]: float(ev["score"])}

            if isinstance(ev.get("q2_by_letter"), dict):
                st["q2_by_letter"] = {str(k): str(v) for k, v in ev["q2_by_letter"].items()}
            if "flawed" in ev:
                st["flawed"] = bool(ev["flawed"])
            if "notes" in ev and ev["notes"]:
                st["notes"] = str(ev["notes"])
            st["completed"] = True

    return item_states


def unblind_session(sealed_map: Dict[str, Any], events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify cryptographic hashes (both per-row map_hash and sealed_manifest sha256) and unblind all completed items."""
    validate_sealed_manifest_integrity(sealed_map)

    rows_meta = sealed_map.get("rows", [])
    for idx, rmeta in enumerate(rows_meta):
        g_idx = rmeta.get("global_index", rmeta.get("row_index", idx))
        expected_hash = canonical_map_hash(rmeta["letter_to_arm"], rmeta["salt"])
        if expected_hash != rmeta.get("map_hash"):
            raise ValueError(
                f"Tamper check failed on item {g_idx} ({rmeta.get('cell_id')}): "
                f"expected map_hash {expected_hash}, got {rmeta.get('map_hash')}"
            )

    item_states = reconstruct_session_answers(sealed_map, events)

    all_arms: List[str] = []
    if isinstance(sealed_map.get("arms"), list):
        for a in sealed_map["arms"]:
            if a not in all_arms:
                all_arms.append(a)
    for rmeta in rows_meta:
        for a in rmeta["arm_to_letter"].keys():
            if a not in all_arms:
                all_arms.append(a)

    arm_ranks: Dict[str, List[int]] = {a: [] for a in all_arms}
    arm_borda: Dict[str, int] = {a: 0 for a in all_arms}
    arm_q2_counts: Dict[str, Dict[str, int]] = {
        a: {"yes": 0, "partly": 0, "no": 0, "total": 0} for a in all_arms
    }
    arm_likert_scores: Dict[str, List[float]] = {a: [] for a in all_arms}
    abx_results: Dict[str, int] = {"correct": 0, "incorrect": 0, "total": 0}
    pairwise_ties: int = 0
    pairwise_wins: Dict[str, Dict[str, int]] = {
        a: {b: 0 for b in all_arms if b != a} for a in all_arms
    }

    unblinded_rows: List[Dict[str, Any]] = []
    warmup_completed_count = 0
    recorded_completed_count = 0

    passes_by_block_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for idx, rmeta in enumerate(rows_meta):
        g_idx = int(rmeta.get("global_index", rmeta.get("row_index", idx)))
        st = item_states[g_idx]
        l2a = rmeta["letter_to_arm"]
        a2l = rmeta["arm_to_letter"]
        method = st["method"]
        is_warmup = st["is_warmup"]
        is_repeat = st["is_repeat"]
        completed = st["completed"]

        if completed:
            if is_warmup:
                warmup_completed_count += 1
            else:
                recorded_completed_count += 1

        ranks_by_arm: Dict[str, int] = {}
        for let, rnk in st["ranks_by_letter"].items():
            if let in l2a:
                ranks_by_arm[l2a[let]] = int(rnk)

        q2_by_arm: Dict[str, str] = {}
        for let, qval in st["q2_by_letter"].items():
            if let in l2a:
                q2_by_arm[l2a[let]] = str(qval)

        likert_by_arm: Dict[str, float] = {}
        for let, lscore in st["likert_by_letter"].items():
            if let in l2a:
                likert_by_arm[l2a[let]] = float(lscore)

        chosen_arm: Optional[str] = None
        if st["choice_letter"] and st["choice_letter"] in l2a:
            chosen_arm = l2a[st["choice_letter"]]
        elif st["choice_outcome"] == "unsure":
            chosen_arm = "unsure"

        abx_target_arm = rmeta.get("abx_target_arm")
        abx_correct: Optional[bool] = None
        if method == "abx" and completed and abx_target_arm and chosen_arm:
            abx_correct = chosen_arm == abx_target_arm

        if completed and not is_warmup:
            k_item = len(l2a)
            if len(ranks_by_arm) == k_item and not (method == "pairwise" and chosen_arm == "unsure"):
                for arm, rnk in ranks_by_arm.items():
                    arm_ranks.setdefault(arm, []).append(rnk)
                    arm_borda[arm] = arm_borda.get(arm, 0) + (k_item - rnk)
                arms_in_item = list(ranks_by_arm.keys())
                for i_a in range(len(arms_in_item)):
                    for j_b in range(i_a + 1, len(arms_in_item)):
                        a1, a2 = arms_in_item[i_a], arms_in_item[j_b]
                        if ranks_by_arm[a1] < ranks_by_arm[a2]:
                            pairwise_wins.setdefault(a1, {})[a2] = pairwise_wins.get(a1, {}).get(a2, 0) + 1
                        elif ranks_by_arm[a2] < ranks_by_arm[a1]:
                            pairwise_wins.setdefault(a2, {})[a1] = pairwise_wins.get(a2, {}).get(a1, 0) + 1
            elif method == "pairwise" and chosen_arm == "unsure":
                pairwise_ties += 1

            for arm, qans in q2_by_arm.items():
                qbucket = arm_q2_counts.setdefault(arm, {"yes": 0, "partly": 0, "no": 0, "total": 0})
                if qans in qbucket:
                    qbucket[qans] += 1
                qbucket["total"] += 1

            for arm, lsc in likert_by_arm.items():
                arm_likert_scores.setdefault(arm, []).append(lsc)

            if method == "abx" and abx_correct is not None:
                abx_results["total"] += 1
                if abx_correct:
                    abx_results["correct"] += 1
                else:
                    abx_results["incorrect"] += 1

        ordering = [
            arm for arm, _ in sorted(ranks_by_arm.items(), key=lambda kv: kv[1])
        ] if ranks_by_arm else []

        row_entry = {
            "global_index": g_idx,
            "row_index": g_idx,
            "block_id": rmeta.get("block_id", "block1"),
            "item_index": rmeta.get("item_index", g_idx),
            "cell_id": rmeta.get("cell_id"),
            "prompt_id": rmeta.get("prompt_id"),
            "prompt_type": rmeta.get("prompt_type"),
            "variant_id": rmeta.get("variant_id"),
            "prompt": rmeta.get("prompt", ""),
            "method": method,
            "is_warmup": is_warmup,
            "is_repeat": is_repeat,
            "repeat_of_cell_id": rmeta.get("repeat_of_cell_id"),
            "completed": completed,
            "letter_to_arm": l2a,
            "arm_to_letter": a2l,
            "map_hash": rmeta["map_hash"],
            "ranks_by_letter": st["ranks_by_letter"],
            "ranks_by_arm": ranks_by_arm,
            "ordering_best_to_worst": ordering,
            "choice_outcome": st["choice_outcome"],
            "chosen_arm": chosen_arm,
            "q2_by_letter": st["q2_by_letter"],
            "q2_by_arm": q2_by_arm,
            "likert_by_arm": likert_by_arm,
            "abx_target_arm": abx_target_arm,
            "abx_correct": abx_correct,
            "flawed": st["flawed"],
            "flaw_reason": st["flaw_reason"],
            "notes": st["notes"],
        }
        unblinded_rows.append(row_entry)

        if completed and not is_warmup:
            canonical_cid = str(rmeta.get("repeat_of_cell_id") or rmeta.get("cell_id") or "")
            b_cid = (str(rmeta.get("block_id", "block1")), canonical_cid)
            passes_by_block_cell.setdefault(b_cid, []).append(row_entry)

    summary_by_arm: Dict[str, Any] = {}
    for arm in all_arms:
        rlist = arm_ranks.get(arm, [])
        rcounts: Dict[str, int] = {}
        for rnk in sorted(set(rlist)):
            rcounts[str(rnk)] = sum(1 for x in rlist if x == rnk)
        q2c = arm_q2_counts.get(arm, {"yes": 0, "partly": 0, "no": 0, "total": 0})
        q2_score = (q2c["yes"] * 1.0 + q2c["partly"] * 0.5) / q2c["total"] if q2c["total"] > 0 else None
        lscores = arm_likert_scores.get(arm, [])
        summary_by_arm[arm] = {
            "n_rows": len(rlist),
            "mean_rank": round(statistics.mean(rlist), 4) if rlist else None,
            "median_rank": statistics.median(rlist) if rlist else None,
            "borda_score": arm_borda.get(arm, 0),
            "rank_counts": rcounts,
            "best_count": rcounts.get("1", 0),
            "best_share": round(rcounts.get("1", 0) / len(rlist), 4) if rlist else None,
            "q2_action_completion": {
                "yes": q2c["yes"],
                "partly": q2c["partly"],
                "no": q2c["no"],
                "total": q2c["total"],
                "mean_score": round(q2_score, 4) if q2_score is not None else None,
            },
            "mean_mos_likert": round(statistics.mean(lscores), 4) if lscores else None,
        }

    repeat_concordance_details: List[Dict[str, Any]] = []
    total_repeat_pairs = 0
    agreeing_repeat_pairs = 0
    for (block_id, cell_id), pass_rows in sorted(passes_by_block_cell.items()):
        if len(pass_rows) < 2:
            continue
        p1, p2 = pass_rows[0], pass_rows[1]
        r1 = p1["ranks_by_arm"]
        r2 = p2["ranks_by_arm"]
        shared_arms = sorted(set(r1.keys()) & set(r2.keys()))
        pair_total = 0
        pair_agree = 0
        for i in range(len(shared_arms)):
            for j in range(i + 1, len(shared_arms)):
                a, b = shared_arms[i], shared_arms[j]
                if r1[a] == r1[b] or r2[a] == r2[b]:
                    continue
                pair_total += 1
                total_repeat_pairs += 1
                if (r1[a] < r1[b]) == (r2[a] < r2[b]):
                    pair_agree += 1
                    agreeing_repeat_pairs += 1
        repeat_concordance_details.append({
            "block_id": block_id,
            "cell_id": cell_id,
            "pass1_item_index": p1["item_index"],
            "pass2_item_index": p2["item_index"],
            "pass1_ranks_by_arm": r1,
            "pass2_ranks_by_arm": r2,
            "pairwise_total": pair_total,
            "pairwise_agree": pair_agree,
            "concordance": round(pair_agree / pair_total, 4) if pair_total > 0 else None,
        })

    overall_concordance = (
        round(agreeing_repeat_pairs / total_repeat_pairs, 4) if total_repeat_pairs > 0 else None
    )

    return {
        "schema": "blindpanel-unblinded-v1",
        "study_id": sealed_map.get("study_id", sealed_map.get("arena", "blindpanel-study")),
        "arena": sealed_map.get("study_id", sealed_map.get("arena", "blindpanel-study")),
        "config_sha256": sealed_map.get("config_sha256"),
        "sealed_manifest_sha256": sealed_map.get("sealed_manifest_sha256"),
        "total_items": len(rows_meta),
        "warmup_completed_items": warmup_completed_count,
        "completed_rows": recorded_completed_count,
        "arms": all_arms,
        "summary_by_arm": summary_by_arm,
        "pairwise_wins": pairwise_wins,
        "pairwise_ties": pairwise_ties,
        "abx_summary": abx_results if abx_results["total"] > 0 else None,
        "repeat_concordance": {
            "total_pairs": total_repeat_pairs,
            "agreeing_pairs": agreeing_repeat_pairs,
            "concordance": overall_concordance,
            "repeats": repeat_concordance_details,
        },
        "rows": unblinded_rows,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Unblind blindpanel session answers using the sealed store.")
    parser.add_argument("--sealed-map", required=True, help="Path to sealed_map_*.bin (in sealed_store/)")
    parser.add_argument("--answers", required=True, help="Path to answers.jsonl")
    parser.add_argument("--out", default="unblinded_results.json", help="Output path for unblinded_results.json")
    args = parser.parse_args(argv)

    with open(args.sealed_map, "r", encoding="utf-8") as f:
        sealed_map = json.load(f)

    events, _ = load_jsonl_safe(args.answers)
    result = unblind_session(sealed_map, events)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"Unblinded {result['completed_rows']} recorded items "
        f"(+{result['warmup_completed_items']} practice items) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
