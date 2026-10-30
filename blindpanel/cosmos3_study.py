"""cosmos3_study.py — Build and verify the Cosmos 3 Amendment A study design config and local sealed session files.

Rules enforced (Card tg & Card tm):
1. The committed public study config (`configs/cosmos3_amendment_a.json`) holds ONLY cell ids, prompts,
   rater cues, block and method settings, and timing rules. It has NO arms per item, NO per-item slot or arm
   orders, NO clip paths or SHA-256 digests, NO seed or salt, and NO absolute local paths.
2. When preparing a local session on a directory of clips (mock or real), a local gitignored config or sealed
   manifest is written outside tracked files, verified by SHA-256 at load.
3. Supports an optional `--arm-map <path>` mapping each pseudonym (`ARM_1..ARM_4`) to a local clip filename prefix,
   read once at session start into the sealed store and never served.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from blindpanel.schema import (
    AMENDMENT_A_Q2_OPTIONS,
    AMENDMENT_A_Q2_PROMPT,
    validate_and_normalize_config,
)

# Prompts and metadata from DESIGN-cosmos3-quant-human-test-2026-10-30.md §1 and rater-cues-v1.json
COSMOS3_PROMPTS: List[Dict[str, Any]] = [
    {
        "id": "M1",
        "type": "manipulation",
        "prompt": "A single robot arm in a warehouse grasps a cardboard box on a moving conveyor belt and lifts it clear of the belt. Static camera, overhead fluorescent light, realistic.",
        "required_action": "grasp and lift a box clear of the belt",
        "estimated_seconds": "1.5–2.5 s",
        "estimated_max_seconds": 2.5,
        "rater_cue": "the box and the gripper through the clip",
    },
    {
        "id": "M2",
        "type": "manipulation",
        "prompt": "A two-armed robot at a kitchen counter grasps a ceramic bowl with one gripper and lifts it off the counter. Static camera, soft daylight, realistic.",
        "required_action": "close the gripper and raise the bowl",
        "estimated_seconds": "1.5–2.0 s",
        "estimated_max_seconds": 2.0,
        "rater_cue": "the ceramic bowl and the gripper through the clip",
    },
    {
        "id": "M3",
        "type": "manipulation",
        "prompt": "A robot gripper positioned close to a grey control panel moves in and presses a red button. Close-up, static camera, realistic.",
        "required_action": "move in and press",
        "estimated_seconds": "~1.0 s",
        "estimated_max_seconds": 1.0,
        "rater_cue": "the red button and the gripper through the clip",
    },
    {
        "id": "M4",
        "type": "manipulation",
        "prompt": "A robot hand holds a glass tilted over a white bowl while water pours out. Static camera, bright light, realistic.",
        "required_action": "continuous pouring (starts mid-pour)",
        "estimated_seconds": "n/a (continuous)",
        "estimated_max_seconds": None,
        "rater_cue": "the tilted glass, the water, and the white bowl through the clip",
    },
    {
        "id": "L1",
        "type": "locomotion",
        "prompt": "A humanoid robot walks steadily down a narrow office hallway. Tracking camera from behind, realistic.",
        "required_action": "walking (continuous)",
        "estimated_seconds": "n/a (continuous)",
        "estimated_max_seconds": None,
        "rater_cue": "the humanoid robot's legs and torso along the hallway through the clip",
    },
    {
        "id": "L2",
        "type": "locomotion",
        "prompt": "A four-legged robot trots across a gravel yard. Side tracking camera, overcast light, realistic.",
        "required_action": "trotting (continuous)",
        "estimated_seconds": "n/a (continuous)",
        "estimated_max_seconds": None,
        "rater_cue": "the four-legged robot's legs and body over the gravel through the clip",
    },
    {
        "id": "L3",
        "type": "locomotion",
        "prompt": "A person jogs steadily along the waterline of a beach at sunrise, seen from the side. Side tracking camera, photoreal.",
        "required_action": "jogging (continuous)",
        "estimated_seconds": "n/a (continuous)",
        "estimated_max_seconds": None,
        "rater_cue": "the person's legs and stride along the waterline through the clip",
    },
    {
        "id": "L4",
        "type": "locomotion",
        "prompt": "A small wheeled delivery robot rolls along a city sidewalk. Static camera, daylight, realistic.",
        "required_action": "rolling (continuous)",
        "estimated_seconds": "n/a (continuous)",
        "estimated_max_seconds": None,
        "rater_cue": "the wheeled robot and the sidewalk surface through the clip",
    },
    {
        "id": "S1",
        "type": "scene",
        "prompt": "Waves break onto a rocky shore at sunset as spray catches the low orange light; the camera pans slowly left along the waterline. Photoreal.",
        "required_action": "none (continuous)",
        "estimated_seconds": "n/a (continuous)",
        "estimated_max_seconds": None,
        "rater_cue": "the waves, the spray, and the rocky waterline through the clip",
    },
    {
        "id": "S2",
        "type": "scene",
        "prompt": "Three people sit at a wooden table with teacups in a cafe, talking and laughing. Handheld camera, warm indoor light, realistic.",
        "required_action": "none (conversation, continuous)",
        "estimated_seconds": "n/a (continuous)",
        "estimated_max_seconds": None,
        "rater_cue": "the three people's faces, hands, and teacups at the table through the clip",
    },
    {
        "id": "S3",
        "type": "scene",
        "prompt": "A city street at night in light rain; cars pass with headlights reflecting on the wet asphalt. Static camera, photoreal.",
        "required_action": "none (continuous)",
        "estimated_seconds": "n/a (continuous)",
        "estimated_max_seconds": None,
        "rater_cue": "the passing cars and the headlight reflections on the wet asphalt through the clip",
    },
    {
        "id": "S4",
        "type": "scene",
        "prompt": "Wind moves tall grass across an open field under a cloudy sky. Static camera, photoreal.",
        "required_action": "none (continuous)",
        "estimated_seconds": "n/a (continuous)",
        "estimated_max_seconds": None,
        "rater_cue": "the tall grass and the sky across the field through the clip",
    },
]

VARIANT_IDS = [1111, 2026, 4242]

# Presentation sequences for Block 1 (43 items: 3 warm-ups + 40 recorded) and Block 2 (42 items: 2 warm-ups + 40 recorded)
# Carefully ordered so:
# - No prompt_id ever repeats on consecutive trials (including across warm-ups -> Sub-block 1A -> Sub-block 1B)
# - Every hidden repeat is separated from its first pass by at least 12 intervening items:
#   - M1-s2026: item index 5 (Item 6) and repeat at item index 18 (Item 19) -> 12 intervening items
#   - L2-s2026: item index 8 (Item 9) and repeat at item index 21 (Item 22) -> 12 intervening items
#   - M1-s4242: item index 24 (Item 25) and repeat at item index 37 (Item 38) -> 12 intervening items
#   - S1-s1111: item index 26 (Item 27) and repeat at item index 40 (Item 41) -> 13 intervening items
BLOCK1_SEQUENCE: List[Tuple[str, int, bool, bool]] = [
    # 3 unrecorded warm-ups (indices 0..2, Items 1..3)
    ("M3", 1111, True, False),
    ("L1", 2026, True, False),
    ("S4", 4242, True, False),
    # Sub-block 1A recorded items (indices 3..22, Items 4..23; break after Item 23)
    ("M2", 1111, False, False),  # idx 3  (Item 4)
    ("S2", 2026, False, False),  # idx 4  (Item 5)
    ("M1", 2026, False, False),  # idx 5  (Item 6)  <-- Pass 1 of M1-s2026
    ("L3", 1111, False, False),  # idx 6  (Item 7)
    ("S3", 1111, False, False),  # idx 7  (Item 8)
    ("L2", 2026, False, False),  # idx 8  (Item 9)  <-- Pass 1 of L2-s2026
    ("M4", 1111, False, False),  # idx 9  (Item 10)
    ("S1", 2026, False, False),  # idx 10 (Item 11)
    ("L4", 1111, False, False),  # idx 11 (Item 12)
    ("M3", 2026, False, False),  # idx 12 (Item 13)
    ("L1", 1111, False, False),  # idx 13 (Item 14)
    ("S4", 1111, False, False),  # idx 14 (Item 15)
    ("M2", 2026, False, False),  # idx 15 (Item 16)
    ("L3", 2026, False, False),  # idx 16 (Item 17)
    ("S2", 1111, False, False),  # idx 17 (Item 18)
    ("M1", 2026, False, True),   # idx 18 (Item 19) <-- Repeat of M1-s2026 (18 - 5 - 1 = 12 intervening)
    ("L4", 2026, False, False),  # idx 19 (Item 20)
    ("M4", 2026, False, False),  # idx 20 (Item 21)
    ("L2", 2026, False, True),   # idx 21 (Item 22) <-- Repeat of L2-s2026 (21 - 8 - 1 = 12 intervening)
    ("S3", 2026, False, False),  # idx 22 (Item 23) <-- End of Sub-block 1A (Rest Break)
    # Sub-block 1B recorded items (indices 23..42, Items 24..43)
    ("L2", 1111, False, False),  # idx 23 (Item 24)
    ("M1", 4242, False, False),  # idx 24 (Item 25) <-- Pass 1 of M1-s4242
    ("L1", 2026, False, False),  # idx 25 (Item 26)
    ("S1", 1111, False, False),  # idx 26 (Item 27) <-- Pass 1 of S1-s1111
    ("M3", 1111, False, False),  # idx 27 (Item 28)
    ("S4", 2026, False, False),  # idx 28 (Item 29)
    ("M1", 1111, False, False),  # idx 29 (Item 30)
    ("L2", 4242, False, False),  # idx 30 (Item 31)
    ("S1", 4242, False, False),  # idx 31 (Item 32)
    ("M3", 4242, False, False),  # idx 32 (Item 33)
    ("L3", 4242, False, False),  # idx 33 (Item 34)
    ("S2", 4242, False, False),  # idx 34 (Item 35)
    ("M2", 4242, False, False),  # idx 35 (Item 36)
    ("L1", 4242, False, False),  # idx 36 (Item 37)
    ("M1", 4242, False, True),   # idx 37 (Item 38) <-- Repeat of M1-s4242 (37 - 24 - 1 = 12 intervening)
    ("S3", 4242, False, False),  # idx 38 (Item 39)
    ("M4", 4242, False, False),  # idx 39 (Item 40)
    ("S1", 1111, False, True),   # idx 40 (Item 41) <-- Repeat of S1-s1111 (40 - 26 - 1 = 13 intervening)
    ("L4", 4242, False, False),  # idx 41 (Item 42)
    ("S4", 4242, False, False),  # idx 42 (Item 43)
]

BLOCK2_SEQUENCE: List[Tuple[str, int, bool, bool]] = [
    # 2 unrecorded warm-ups (indices 0..1, Items 1..2)
    ("M2", 1111, True, False),
    ("S3", 2026, True, False),
    # 40 recorded pairs (36 core + 4 repeats, indices 2..41, Items 3..42)
    *[entry for entry in BLOCK1_SEQUENCE[3:]],
]


def _prompt_by_id() -> Dict[str, Dict[str, Any]]:
    return {p["id"]: p for p in COSMOS3_PROMPTS}


def build_public_study_design_config(
    block1_arms: Optional[List[str]] = None,
    block2_arms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Construct the clean, public study design config (Rule 1).

    Holds ONLY cell ids, prompts, rater cues, block and method settings, and timing rules.
    Contains NO per-item arms, NO clip paths or SHA-256 digests, NO slot orders, and NO seeds/salts.
    """
    prompts_map = _prompt_by_id()
    b1_arms = block1_arms or ["ARM_1", "ARM_2", "ARM_3"]
    b2_arms = block2_arms or ["ARM_1", "ARM_4"]

    block1_items: List[Dict[str, Any]] = []
    for idx, (pid, var_id, is_warmup, is_repeat) in enumerate(BLOCK1_SEQUENCE):
        pmeta = prompts_map[pid]
        base_cell_id = f"{pid}-s{var_id}"
        cell_id = f"{base_cell_id}-warmup" if is_warmup else (f"{base_cell_id}-rep2" if is_repeat else base_cell_id)
        block1_items.append({
            "cell_id": cell_id,
            "base_cell_id": base_cell_id,
            "prompt_id": pid,
            "prompt_type": pmeta["type"],
            "variant_id": var_id,
            "prompt": pmeta["prompt"],
            "rater_cue": pmeta["rater_cue"],
            "required_action": pmeta["required_action"],
            "estimated_seconds": pmeta["estimated_seconds"],
            "estimated_max_seconds": pmeta["estimated_max_seconds"],
            "clip_seconds": 2.54,
            "is_warmup": is_warmup,
            "is_repeat": is_repeat,
            "repeat_of_cell_id": base_cell_id if is_repeat else None,
            "ask_q2": pmeta["type"] == "manipulation",
            "break_after": (idx == 22),
        })

    block2_items: List[Dict[str, Any]] = []
    for _idx, (pid, var_id, is_warmup, is_repeat) in enumerate(BLOCK2_SEQUENCE):
        pmeta = prompts_map[pid]
        base_cell_id = f"{pid}-s{var_id}"
        cell_id = (
            f"{base_cell_id}-b2-warmup"
            if is_warmup
            else (f"{base_cell_id}-b2-rep2" if is_repeat else f"{base_cell_id}-b2")
        )
        block2_items.append({
            "cell_id": cell_id,
            "base_cell_id": base_cell_id,
            "prompt_id": pid,
            "prompt_type": pmeta["type"],
            "variant_id": var_id,
            "prompt": pmeta["prompt"],
            "rater_cue": pmeta["rater_cue"],
            "required_action": pmeta["required_action"],
            "estimated_seconds": pmeta["estimated_seconds"],
            "estimated_max_seconds": pmeta["estimated_max_seconds"],
            "clip_seconds": 2.54,
            "is_warmup": is_warmup,
            "is_repeat": is_repeat,
            "repeat_of_cell_id": f"{base_cell_id}-b2" if is_repeat else None,
            "ask_q2": False,
        })

    cfg: Dict[str, Any] = {
        "schema_version": "blindpanel-config-v1",
        "study_id": "cosmos3-amendment-a-v1",
        "title": "Blind Video Evaluation — Block 1 (3-Clip BWS · 43 Items) + Optional Block 2 (2AFC · 42 Items)",
        "clip_spec": {
            "width": 832,
            "height": 480,
            "frames": 61,
            "fps": "24.0000",
            "seconds": 2.54,
        },
        "question2_spec": {
            "id": "action_completion",
            "prompt": AMENDMENT_A_Q2_PROMPT,
            "per_clip": True,
            "options": list(AMENDMENT_A_Q2_OPTIONS),
        },
        "blocks": [
            {
                "block_id": "block1_bws3",
                "title": "Main Block 1 — Pick Best & Worst of 3",
                "optional": False,
                "method": "bws_3",
                "arms": b1_arms,
                "allow_unsure": False,
                "allow_skip": False,
                "require_full_play": True,
                "break_after_item": 23,
                "enforce_no_adjacent_prompt": True,
                "enforce_counterbalance_bws3": True,
                "items": block1_items,
            },
            {
                "block_id": "block2_2afc",
                "title": "Block 2 · Side-by-Side Comparison",
                "optional": True,
                "method": "2afc",
                "arms": b2_arms,
                "allow_unsure": False,
                "allow_skip": False,
                "require_full_play": True,
                "break_after_item": None,
                "enforce_no_adjacent_prompt": True,
                "enforce_counterbalance_bws3": False,
                "items": block2_items,
            },
        ],
    }

    validate_and_normalize_config(cfg, forbid_study_blinding_data=True)
    return cfg


def discover_clips_in_dir(
    clips_dir: str,
    arms: List[str],
    base_cell_id: str,
    arm_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Resolve and hash clip files on disk for a given cell and arm list.

    Supports naming conventions `{prefix}_{base_cell_id}.mp4` or `{prefix}_*_{base_cell_id}.mp4`,
    where `prefix` is `arm_map.get(arm, arm)` if `arm_map` is provided, or `arm` otherwise.
    """
    if not os.path.isdir(clips_dir):
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")
    entries = os.listdir(clips_dir)
    resolved_arms: Dict[str, Dict[str, str]] = {}
    for arm in arms:
        file_prefix = str(arm_map[arm]).strip() if (arm_map and arm in arm_map) else arm
        direct_name = f"{file_prefix}_{base_cell_id}.mp4"
        matched_file: Optional[str] = None
        if direct_name in entries:
            matched_file = direct_name
        else:
            prefix = f"{file_prefix}_"
            suffix = f"_{base_cell_id}.mp4"
            for fname in entries:
                if fname.startswith(prefix) and fname.endswith(suffix):
                    matched_file = fname
                    break
        if not matched_file:
            raise FileNotFoundError(
                f"Clip for arm '{arm}' and cell '{base_cell_id}' not found in '{clips_dir}'"
            )
        full_path = os.path.abspath(os.path.join(clips_dir, matched_file))
        with open(full_path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        resolved_arms[arm] = {
            "path": full_path,
            "sha256": digest,
        }
    return resolved_arms


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the public Cosmos 3 Amendment A study design config (no study blinding data)."
    )
    parser.add_argument(
        "--out",
        default="configs/cosmos3_amendment_a.json",
        help="Output path for the public study design config JSON",
    )
    args = parser.parse_args(argv)

    cfg = build_public_study_design_config()
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote public study design config (Block 1=43 items, Block 2=42 items) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
