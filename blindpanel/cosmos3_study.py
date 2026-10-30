"""cosmos3_study.py — Build and verify the first study config for Cosmos 3 Amendment A from the 144 real B200 clips.

Study layout (`DESIGN-amendment-A-2026-10-30.md`):
- Block 1 (Primary Main Super 3-Clip BWS · 3 Arms: BF16, Q1, Q3):
  - 3 unrecorded warm-up items at start (`M3-s1111`, `L1-s2026`, `S4-s4242` under practice salts)
  - 36 core recorded `(prompt, seed)` triples (12 prompts M1-M4, L1-L4, S1-S4 × 3 seeds 1111, 2026, 4242)
  - 4 hidden repeat triples (`M1-s2026`, `L2-s2026` in Sub-block 1A; `M1-s4242`, `S1-s1111` in Sub-block 1B)
  - Total Block 1 items = 43 rater-facing items (40 recorded rows), mid-block rest break after Item 23 / 43.
  - Counterbalancing: all 6 permutations of `(BF16, Q1, Q3)` appear 6 times each across the 36 core triples
    (2 per prompt type), forming a 3×3 Latin square over slots `(A, B, C)` within each prompt's 3 seeds.
  - Presentation order: no prompt_id appears twice in a row (`prompt_t != prompt_{t-1}`), and each hidden repeat
    is separated by >= 12 intervening items with a cyclic slot shift `(A, B, C) -> (B, C, A)`.
  - Question 2 (`How far does the robot get with the action within the 2.5 s clip?` ->
    `Yes (completes it)` / `Partly (clearly under way)` / `No (not achieved)`) asked on M1–M4 rows only.
- Block 2 (Optional Nano vs BF16 2AFC Sensitivity Check · 2 Arms: BF16, NANO):
  - 2 unrecorded warm-up pairs at start (`M2-s1111`, `S3-s2026`)
  - 36 core recorded `(prompt, seed)` pairs (18 Left / 18 Right, 6 Left / 6 Right per prompt type)
  - 4 hidden repeat pairs (`M1-s2026`, `M1-s4242`, `L2-s2026`, `S1-s1111` with Left/Right swapped from Pass 1)
  - Total Block 2 items = 42 rater-facing items (40 recorded pairs).
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

DEFAULT_CLIPS_DIR = "E:/AI/rnd-local/projects/cosmos3/b200-same-card/artifacts/arm4_NANO/x/mp4"

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

SEEDS = [1111, 2026, 4242]

# 6 permutations of (BF16, Q1, Q3), split into two cyclic Latin-square families:
# Even family E0, E1, E2:
#   E0 = (BF16, Q1, Q3)
#   E1 = (Q1, Q3, BF16)
#   E2 = (Q3, BF16, Q1)
# Odd family O0, O1, O2:
#   O0 = (BF16, Q3, Q1)
#   O1 = (Q3, Q1, BF16)
#   O2 = (Q1, BF16, Q3)
# Within each prompt type (4 prompts x 3 seeds = 12 cells):
#   Prompts 0 and 1 get Even family across seeds (1111->E0, 2026->E1, 4242->E2) and (1111->E1, 2026->E2, 4242->E0)
#   Prompts 2 and 3 get Odd family across seeds  (1111->O0, 2026->O1, 4242->O2) and (1111->O1, 2026->O2, 4242->O0)
# Every prompt's 3 seeds form a 3x3 Latin square over slots A, B, C, and each of the 6 permutations
# appears 2 times per prompt type (6 times total across the 36 core triples).
EVEN_LATIN: List[List[str]] = [
    ["BF16", "Q1", "Q3"],
    ["Q1", "Q3", "BF16"],
    ["Q3", "BF16", "Q1"],
]
ODD_LATIN: List[List[str]] = [
    ["BF16", "Q3", "Q1"],
    ["Q3", "Q1", "BF16"],
    ["Q1", "BF16", "Q3"],
]

# Pre-verified SHA-256 table for all 144 clips on the B200 Same-Card run
B200_144_SHA256: Dict[str, str] = {
    "BF16_cell_L1-s1111.mp4": "f01d33fc959a1aa6e5ef9a840f5bed22c4c789263b1c928124a6edf38984d206",
    "BF16_cell_L1-s2026.mp4": "91123fd635bdf7efa78d73773aa7cfa5435cae00ae65e8824c0e42f5fc67c448",
    "BF16_cell_L1-s4242.mp4": "b11cea45bfaf6682dbcab256b1c0c48b45693896709b51d75eaccba5ad58c7e9",
    "BF16_cell_L2-s1111.mp4": "a90b4e9895631284ca232ee950142ecfd7393291af8b0e24423e201946c2a840",
    "BF16_cell_L2-s2026.mp4": "9a5f75073e708c06402e7979b550119487eee6818120e529fc23c001cff52052",
    "BF16_cell_L2-s4242.mp4": "d7fcc6ebc68b5af3858e3bbda8cc83fec6ecac8a8f018dc1d677d113ede6a781",
    "BF16_cell_L3-s1111.mp4": "e61e0b52fb5a31f0ea6385be488f075799573b5c2b0dd605e08a80738d9df409",
    "BF16_cell_L3-s2026.mp4": "22c77732ab63a4e7ca6456df8570a932968b3089d9d697c405f5265fb0d9650a",
    "BF16_cell_L3-s4242.mp4": "0578d03387c0413b12ff0e6781f722b3be9f8b7b70f44bd22ee81f832d2fb4e1",
    "BF16_cell_L4-s1111.mp4": "e09960d3ee72386b9b1916f7ab89bf653160e947e6f728d46963391c5222a9f8",
    "BF16_cell_L4-s2026.mp4": "752c9a4929c0d59c15271b2b84c945a30672249ac236097754fbf700daaa6d08",
    "BF16_cell_L4-s4242.mp4": "d50ef73c90fe2c41be0a985b4327aa4666d63cd18b39f55276c678f0a402c5dc",
    "BF16_cell_M1-s1111.mp4": "26738e93925931849239a9ee0aa53f88a98c89c7b7046b946fea21df4b259a5b",
    "BF16_cell_M1-s2026.mp4": "f3b0839ef61e5668a580f0753986f319a0394ed4677081d3c70ce64d11585964",
    "BF16_cell_M1-s4242.mp4": "0c79728b8ae3c7c77e26cbad851cf5a4c8910bb36790c266ab160a72ce68d6ef",
    "BF16_cell_M2-s1111.mp4": "d20195fe82636d57e8c37e250ef6f6ac06c340339b6b33fd79e72921ee53a44a",
    "BF16_cell_M2-s2026.mp4": "a26ec6a42f789351edc6773de199629850caa2366fe461a14501078e67c17404",
    "BF16_cell_M2-s4242.mp4": "f062994b361d4bb4f92b9d5c82a4b2be2a46d3afa25836931f1d6a6de5434644",
    "BF16_cell_M3-s1111.mp4": "77931b4f678fd8c612b263b99552c43f0cebae8e5f0f39520699cb2d3f180e14",
    "BF16_cell_M3-s2026.mp4": "f71f274bf44ca92526fb7db7a126b676e519bac09ceeda6d8882764d8030bfd0",
    "BF16_cell_M3-s4242.mp4": "0c0ac4a2a74bc623c9470e7a0b52ec82fdbc1b8ce1672b93e56ce6995e949735",
    "BF16_cell_M4-s1111.mp4": "5c063ff6cba0987ff8f405c082869e7b73f4b1128c32f63eb699f0a4d598f7c6",
    "BF16_cell_M4-s2026.mp4": "c4f314ffaeda74205992229cd234d073ca69b575332a142072099a014e032b93",
    "BF16_cell_M4-s4242.mp4": "3e52a12f90696bca9c5fc56a4fe13ed0eb865d714814d235dc69c60e4a664dec",
    "BF16_cell_S1-s1111.mp4": "058ccc633041bff91ea68af211e37fa82d62a178d96e865efe95f489df936c7d",
    "BF16_cell_S1-s2026.mp4": "79dc75f141df436678b930f90a93b6953b76f0debcba354afd20c0282efe34fe",
    "BF16_cell_S1-s4242.mp4": "52583e57c452235b7779d46449cf1db30a69de92e787150762bd2a9af602dba8",
    "BF16_cell_S2-s1111.mp4": "0e2807e9f2b8686731d0e2962a3bbb880eb7cf76e0fbe5facd5d1b28bd6244f1",
    "BF16_cell_S2-s2026.mp4": "4db5690b568ca3abaf77ce3e945166cc7d3dcdb19115ccecca7a696a17b3f985",
    "BF16_cell_S2-s4242.mp4": "d1b552795c87855475f8db05493fd8fecd70f181170094d5f85c1bc6ba98a604",
    "BF16_cell_S3-s1111.mp4": "bd846640dd95bce8f8ec0f942973850de45c1d7364a4d6d0ffc6e955d659cdc8",
    "BF16_cell_S3-s2026.mp4": "825b7abf6b6609617f27c35cea81afcca1a9e660ed5c0d4487133575b5122794",
    "BF16_cell_S3-s4242.mp4": "824ff0a957820e640d8e352433dc9ffbbde0ae7b83e4b63e741e5bdd823d5a35",
    "BF16_cell_S4-s1111.mp4": "0b0bc7d0aff78004421853e16492d2eb261fde3cdf7f62ccb9deac5dc7408a77",
    "BF16_cell_S4-s2026.mp4": "8a767d00615a263dc2b98ad36d1c35a3c83f7ffc93bbf3eeb256b68aa353e444",
    "BF16_cell_S4-s4242.mp4": "a38a2b4870c91e2223353ae4c4a16dd4578f4cd8e48149121a63082b74e2fe8d",
    "NANO_cell_L1-s1111.mp4": "89378e34f09d98475252df1ef05c50719a76707bbe1bf4c67b173f55c18537ba",
    "NANO_cell_L1-s2026.mp4": "04b6ba43e85d44c03307a967411905bf58e29509611d8593428daeb8c96182ba",
    "NANO_cell_L1-s4242.mp4": "dc11bb4852cd8a9b8891f403c56572e0aef4eacaf6581f928db7c94e97335ba1",
    "NANO_cell_L2-s1111.mp4": "681090d7756bb45820a9ce3944a73a82640be2da3301045c39611c36bb6f6b83",
    "NANO_cell_L2-s2026.mp4": "79e0734375eaeef25f4a3e7532db436a5cffa50d635d8c5706d6062183965574",
    "NANO_cell_L2-s4242.mp4": "d050b065757bc7aa622eb4b11dad09c504a0ad055f9b6e0d917d7f1d598ef405",
    "NANO_cell_L3-s1111.mp4": "4442cd377bbac518a36dd5b82f8e34618c4f4715c6e1aaea41a851e8f708fea5",
    "NANO_cell_L3-s2026.mp4": "5a4de548755b782d7abbb3bb511bdb18c6aa0d19876f9c91bf6bad1ee44f318c",
    "NANO_cell_L3-s4242.mp4": "7ac736ac505fa07734cd719bab74a4070ab08a96658b1d07a51df9064716b541",
    "NANO_cell_L4-s1111.mp4": "191a2f11517912e6732624430cbec80e2d00e9430cc4d4d11e2c8a9704996e21",
    "NANO_cell_L4-s2026.mp4": "b2f47113700a088c48d443323a330a3ef509145db1bafe7e8938f7671ecb9ba4",
    "NANO_cell_L4-s4242.mp4": "c2a19e7de951423a7c581ff0b4c9ab3e9264d2e538daa497a7c60d82632d238b",
    "NANO_cell_M1-s1111.mp4": "30718308c509512e1985de909c758a7d9d6acf4f4af048ee6c0acfd80fec55bd",
    "NANO_cell_M1-s2026.mp4": "f3baeb57edf29a0c2ebf007ce7f29e03f9140a6de2c279a26919ba21edadcb4b",
    "NANO_cell_M1-s4242.mp4": "7e345a4b11a5d0ef29d0f5582249bebb5dcb710191415cc6dcf4a4488feea2a7",
    "NANO_cell_M2-s1111.mp4": "cc1e807aa100e0024ccf7220b777933c823024b2ea0a89d287768d73fe6179a1",
    "NANO_cell_M2-s2026.mp4": "af5b8f441a32d2699e15ae7186b82250c08edcc34e569b242e2ad72ad1ef0735",
    "NANO_cell_M2-s4242.mp4": "26b20a4a8152998aa3a11eddd4c337739f0fec8346947be007cadc1a5e4239d0",
    "NANO_cell_M3-s1111.mp4": "cdfc036d598953d29938170cd239ff405e155574bb2fc6a610d27527c397876d",
    "NANO_cell_M3-s2026.mp4": "7eb9a868c9cea38082d1015f5783cd53dd7f01ea90df68e5c9be22f268a08ab1",
    "NANO_cell_M3-s4242.mp4": "a89b97cc15cc7c4880c9f66b75370ef05a8e305f3759a99714f8f1ee22933b27",
    "NANO_cell_M4-s1111.mp4": "53b40905303d6f24d28b3f9a607765752738d78f55ef70d985e4aa152f258b75",
    "NANO_cell_M4-s2026.mp4": "548503cfc2d24f8cf9c9dd21c45e0653e932acbc203584404a56ac1dbdbcbdaf",
    "NANO_cell_M4-s4242.mp4": "d5d1a370cb8228630a91f9689a763a33a3511965e51a6e64f53b866cdd45ced8",
    "NANO_cell_S1-s1111.mp4": "e4278be6ff2c6c415bfc3eadf12e6898704e2bf228cd4a7897eb0f447316f16a",
    "NANO_cell_S1-s2026.mp4": "6f0cd57255a5a80475314ecabc8da2ce10203ad34908a46dff231ba9feb885f4",
    "NANO_cell_S1-s4242.mp4": "b5a2f0d853cc18555c12a9f078d4cd52e479c8e664dc42adaca9bb18831cbf29",
    "NANO_cell_S2-s1111.mp4": "f36b76b76b9e349919179f013c488fa77fe3b1baf6ed1b928197465305e2ed3f",
    "NANO_cell_S2-s2026.mp4": "37f772d92bde8203b1e6cd6cdf054214b6d0aeaf28939acdeecfed568ba24278",
    "NANO_cell_S2-s4242.mp4": "8551595aedbd7da10e1dabe24871cd2496dee9cf83098cb719506f827ede27cf",
    "NANO_cell_S3-s1111.mp4": "87907475a9063b33ea6608d9eaa411ec815cfdc5f4ace837f9fb9a049fa21334",
    "NANO_cell_S3-s2026.mp4": "decca60c40c15e28202510331aa52aeee710bf0ce56a117770d4c0d23e4711e1",
    "NANO_cell_S3-s4242.mp4": "f51afd94a00535bb6eec9f803dbbaaefbabe396a288604a84a43486d265b089b",
    "NANO_cell_S4-s1111.mp4": "1408b75a28fdb76dbca7c3d8fb552c4a660abb98abff2acd5fb0d85492521232",
    "NANO_cell_S4-s2026.mp4": "aae1c6458a360d535f78ef88019def799de9806aca6f3dcd193541c32ad262b3",
    "NANO_cell_S4-s4242.mp4": "578ae75ff30f05f7ca4773d4dfcfa8a2eb15830e067f8ff0543f01879e19e1cc",
    "Q1_cell_L1-s1111.mp4": "7f5eb5fb688adc348147829e4c55a392d7049287e786bfd5b33cb543d16d3067",
    "Q1_cell_L1-s2026.mp4": "2fb7f0d28a900596018a4aecc8e74682e6289e18a81f68fc3217dbaa807f3664",
    "Q1_cell_L1-s4242.mp4": "4b8ccbb84f6571f180ad20dab3f7e7159bb164216e4298281aa0f50563036064",
    "Q1_cell_L2-s1111.mp4": "8a27184db954c4157a7646351669c7a6ce086f3a5e3fdcae6386617167e942df",
    "Q1_cell_L2-s2026.mp4": "a4e35855d3770eb7f57f078c636b02cb378042e5d083c755e5c766f8fb0c33bc",
    "Q1_cell_L2-s4242.mp4": "ddf3c1c540041fcd37e7feb9d6dedcffb459f0e57135bcef2108eed2b9b3caea",
    "Q1_cell_L3-s1111.mp4": "daeb153c55e918fcc41e5ea22fa634fda9f4a76e63b02e574600e7c92c32fe0a",
    "Q1_cell_L3-s2026.mp4": "c87550bb0335c1caa4a3db2bac722dec88c1a26f874544b413146b316a928929",
    "Q1_cell_L3-s4242.mp4": "4ecb651aa9ec6d1ab6e0a4677c18b962be250abfa580c027ebe546f931395451",
    "Q1_cell_L4-s1111.mp4": "d28943d5cf881f71358f13a831f0b75ce84e8f513aa0caa5fa758b73c2e94200",
    "Q1_cell_L4-s2026.mp4": "f03e9759d5da005b474e68b1727e757c459daf5810f38a798db37c50b9a99bc5",
    "Q1_cell_L4-s4242.mp4": "60d4c64ef5025f45bbf5aa6caf7a0cd0547b15257c9d93185ed7c4035b22b7a4",
    "Q1_cell_M1-s1111.mp4": "97bad351b5e4fd3d462b3fa32d637a2e6e5dc2b74ed55d24088ace86defbc7a4",
    "Q1_cell_M1-s2026.mp4": "a7c4c44c0fa434c24ed78064357716bc74c3ef7175513b3479933ccafc58f812",
    "Q1_cell_M1-s4242.mp4": "6a8995b48f228598b8122b4a29097fe624ddb05ced1a5707c43a55c9374dd2ab",
    "Q1_cell_M2-s1111.mp4": "6f11ec86c0731875d944495bb80b0770738d21c484583eacef0d32b09cc50131",
    "Q1_cell_M2-s2026.mp4": "7641e3b01845b4434b2167a08b1aa9ad50ba1ac828c510e6b7fce26a301d6280",
    "Q1_cell_M2-s4242.mp4": "1943b9d2e69322ac53e26e7ccd83cd4264a4bf1d3a7c682e38fcee083ec0cdd1",
    "Q1_cell_M3-s1111.mp4": "34e8bcfecbb42647e1efd93f1ced936d87ba1136e7ed309d763ab80f2197dca2",
    "Q1_cell_M3-s2026.mp4": "03d7c6f38f46cc04c602e5cf4f00fc870ab08590a44e86f1d2ff95f3bdbe86ce",
    "Q1_cell_M3-s4242.mp4": "870ac2cf85aba40a350a32ee04f17f8873f145d33cd3bb42d56cb7ae1346d433",
    "Q1_cell_M4-s1111.mp4": "e06ded8401995204936bb42703cce817cc91e17e56d04d4b8a9d54ae076c6891",
    "Q1_cell_M4-s2026.mp4": "510b97249987e99faded6897c5e066dc145d2bcb1cc898537a2b0b805bf8ae5b",
    "Q1_cell_M4-s4242.mp4": "7ada540117313fa6f0ab2a01dad37c95180e074e0d4c3e535a620e2e3e852fa2",
    "Q1_cell_S1-s1111.mp4": "024c6bb0e482e1ff8496505efb3eac067dfdae30af7c0246eb007c951fdce680",
    "Q1_cell_S1-s2026.mp4": "ae3c410a66ad81268a960911803592dd3fea2c5524c5705899e8603d4db8e42a",
    "Q1_cell_S1-s4242.mp4": "44684af326bbf2b430371bb317530d36aad1aabc4afbafe174b4a4dd5b70f16c",
    "Q1_cell_S2-s1111.mp4": "00d65a73069bd5433d4ad5bbccbe2618415d27bb3bb463d9260726d99efb620b",
    "Q1_cell_S2-s2026.mp4": "1eff64e79f97f3dfc1497f5c105d9c560b6b9343adec5ee7ce121b63ec36f500",
    "Q1_cell_S2-s4242.mp4": "7255faaad3a23d2094c1af3dc7b6245469e27470bc277e4280994832c1f13f7b",
    "Q1_cell_S3-s1111.mp4": "085e9bfa5d89a7fa90bfecedcf92c606a08c82c2fdfbc54337199d19472e3608",
    "Q1_cell_S3-s2026.mp4": "7c3ac008056a12a3b620d95efd7020b112a61f0186514727781ceba4f2504890",
    "Q1_cell_S3-s4242.mp4": "c5e2f691417b5399d0d9fac794d5903ef99be5245907e8a721468b2e0b30244b",
    "Q1_cell_S4-s1111.mp4": "60f0ac435e8c5485f16720f6f7dbd80cab089e09beca8af28d3d4473de908389",
    "Q1_cell_S4-s2026.mp4": "4a205cc2f0c96ef3d84391f5f65df658977e0d22a33ea82d3fa42060a5402401",
    "Q1_cell_S4-s4242.mp4": "a6275093c45f2d1fd9b660f9b164dc35eb2934f3d699e1b3313fb59051d701b8",
    "Q3_cell_L1-s1111.mp4": "60f9c8af8c3821897bba94b7e3a186c132dc2869432f00219a5eb141834d7feb",
    "Q3_cell_L1-s2026.mp4": "fadee9779e033dde44b28c4abb9ab952655e25fab8e0c10ac068d6c36cfb4834",
    "Q3_cell_L1-s4242.mp4": "8caeb3567560a8ac540c7a2a3a0de9d0cb7c13362b0cae6da690bebef9df6c52",
    "Q3_cell_L2-s1111.mp4": "a821657aad03c93b74c69bda5d772a11ac2a0404e8bf4f789ccf8442bb88e3d2",
    "Q3_cell_L2-s2026.mp4": "387cfbfdb48e4ede3639dae85aba26d54fef1c24d889b616c2e2a0ce3fb05661",
    "Q3_cell_L2-s4242.mp4": "766c1fc2963468c8b971bd260e624792058c2dcb9a948c9624500de93e0af5ed",
    "Q3_cell_L3-s1111.mp4": "434a87f0f5896a911e309f7e39db38ca6168f489e66b4d8d928a83b7351905d9",
    "Q3_cell_L3-s2026.mp4": "e0d187edc4369ce7362912ff27daa89935901ca79f9dfb2d6a42292c6e0fae00",
    "Q3_cell_L3-s4242.mp4": "b88efea317a63fc38e40e7da59c632f1fc2d0abef57986c666bb428e5d3edb54",
    "Q3_cell_L4-s1111.mp4": "0401bca617561e16ebb9c649efb0a2495c72570edd5b414003c04284ad55de16",
    "Q3_cell_L4-s2026.mp4": "9bebbfebd01bd9af6c78718823bec61dd619c170ea3889b3a3a0a814a1499bb5",
    "Q3_cell_L4-s4242.mp4": "f9f0fd35211542fbf9e61385055e763d379b38d6415c127935d0793f9f905492",
    "Q3_cell_M1-s1111.mp4": "6e59259ff3bbb63e6fcba0e9b012da6cd9a68bc8cbb54b17341ef05dfd9cbad2",
    "Q3_cell_M1-s2026.mp4": "80811ae7abfff484ab3ace6147a5905678ba04b286121e0113ab5ac58cc44922",
    "Q3_cell_M1-s4242.mp4": "2a7c1c5fbdba4bdf6e98a44f37f8751f4a6914a4aa15a8648ec15d0654f6a19b",
    "Q3_cell_M2-s1111.mp4": "566c45e04db516741721fbae418d04304486d31efbb049c21c1ddfa8b0ff7c53",
    "Q3_cell_M2-s2026.mp4": "c74f2252f1d05f835db8eba663f1d2938c4261f551090fd876dcdf53ad3f0fcd",
    "Q3_cell_M2-s4242.mp4": "71c31c1a1c74ce66bdc806c16c60121fe771fea463e0fa8acf7104d33acf58ae",
    "Q3_cell_M3-s1111.mp4": "d392acc76eeff777055fa8fffc4482cfb44ea37a487bfbbb23a6939aa522b6f0",
    "Q3_cell_M3-s2026.mp4": "85eae71092c60a77e8c5bacb2921bbab6be70c9237e9ae3dcebfd3e15a0ffe82",
    "Q3_cell_M3-s4242.mp4": "f5d5a6041fb43f82d710f42d19c3d052d389c205bdfa7121d31a82cd680269e1",
    "Q3_cell_M4-s1111.mp4": "670b6a1deb1180187283102e7245183cf00b09fe6c6cbf18613d89fcda32051b",
    "Q3_cell_M4-s2026.mp4": "abd0535e4323c4531a9f8359acc5abe84c35bc23bca43a943c252d07d8403c15",
    "Q3_cell_M4-s4242.mp4": "f0191020bd7b60750b05e70c1bf38898b8fbf37bd2f0e11ead5a59a9cf1c3223",
    "Q3_cell_S1-s1111.mp4": "2909f975b1252ae1e6f28dea38a1e0761237dd3a0caf3fdb8843e2b6177ed0d1",
    "Q3_cell_S1-s2026.mp4": "9445fb9f9b152cbd2f1a748fd630794700e29a2014a9f9a402d1787e64e0cb94",
    "Q3_cell_S1-s4242.mp4": "c5f0f5d2fc8062f1f9c3e82d4b8604b1ae0dd30340e7da05c7acc84ac507d0ca",
    "Q3_cell_S2-s1111.mp4": "982404bc19c2eb2be2634876268c3397e6fcbcd1f3c941f3916b1b5dbd0fc14a",
    "Q3_cell_S2-s2026.mp4": "4f7b82073b3336ba7dfc652e8e65ddfff7451356704af996e13fdc3f2738177c",
    "Q3_cell_S2-s4242.mp4": "f69200d1a086831ecbb9f5914499124029a7bd1bdf7566455de89e39624492cc",
    "Q3_cell_S3-s1111.mp4": "f2a31d925a3c8d5b8af47617138578b36d897ef7c1ac8f062129aea4ef81f3ab",
    "Q3_cell_S3-s2026.mp4": "c5b137a694a28b5b83b82f0c8cb941747c19aa9dccb4f680396fbddcf71a89a6",
    "Q3_cell_S3-s4242.mp4": "97744a00a5cf3f1b336f2959e7507086ca6dba3167c458cf7e03d8f3d36b43e4",
    "Q3_cell_S4-s1111.mp4": "ec76c473e09431471cca9f8f92337b360c28ac0a8e353b6d0cc3e8c06da0cbc6",
    "Q3_cell_S4-s2026.mp4": "7d2b172062a28dc82bbd5b7f65343467124eab9d3a2b6ebdc142ed5612ae4e28",
    "Q3_cell_S4-s4242.mp4": "f3ed8bc4940505badb1c7c35c05177feb76c9edf4edad84d3bab5878ebc055cc",
}

# Exact presentation sequences for Block 1 (43 items: 3 warm-ups + 40 recorded) and Block 2 (42 items: 2 warm-ups + 40 recorded)
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


def compute_bws3_core_permutations() -> Dict[Tuple[str, int], List[str]]:
    """Assign each of the 36 (prompt_id, seed) triples one of the 6 permutations of (BF16, Q1, Q3)
    so that:
    - Each of the 6 permutations occurs 6 times overall and 2 times per prompt type
    - Each prompt's 3 seeds form a 3x3 Latin square over slots (A, B, C)
    """
    assignments: Dict[Tuple[str, int], List[str]] = {}
    by_type = {
        "manipulation": ["M1", "M2", "M3", "M4"],
        "locomotion": ["L1", "L2", "L3", "L4"],
        "scene": ["S1", "S2", "S3", "S4"],
    }
    for _ptype, pids in by_type.items():
        for p_offset, pid in enumerate(pids):
            family = EVEN_LATIN if p_offset < 2 else ODD_LATIN
            shift = p_offset % 2
            for s_idx, seed in enumerate(SEEDS):
                perm = list(family[(s_idx + shift) % 3])
                assignments[(pid, seed)] = perm
    return assignments


def compute_2afc_core_orders() -> Dict[Tuple[str, int], List[str]]:
    """Assign each of the 36 (prompt_id, seed) pairs to ["BF16", "NANO"] or ["NANO", "BF16"]
    so that BF16 is Left in 18 pairs (6 per prompt type) and Right in 18 pairs (6 per prompt type),
    with 2/1 or 1/2 balance across each prompt's 3 seeds.
    """
    assignments: Dict[Tuple[str, int], List[str]] = {}
    by_type = {
        "manipulation": ["M1", "M2", "M3", "M4"],
        "locomotion": ["L1", "L2", "L3", "L4"],
        "scene": ["S1", "S2", "S3", "S4"],
    }
    # Pattern across 4 prompts x 3 seeds = 12 cells per type (6 AB, 6 BA; each prompt has 2/1 or 1/2):
    # P0: AB, BA, AB (2 AB, 1 BA)
    # P1: BA, AB, BA (1 AB, 2 BA)
    # P2: AB, AB, BA (2 AB, 1 BA)
    # P3: BA, BA, AB (1 AB, 2 BA)
    patterns = [
        [True, False, True],
        [False, True, False],
        [True, True, False],
        [False, False, True],
    ]
    for _ptype, pids in by_type.items():
        for p_offset, pid in enumerate(pids):
            for s_idx, seed in enumerate(SEEDS):
                bf16_left = patterns[p_offset][s_idx]
                assignments[(pid, seed)] = ["BF16", "NANO"] if bf16_left else ["NANO", "BF16"]
    return assignments


def build_cosmos3_amendment_a_config(
    clips_dir: str = DEFAULT_CLIPS_DIR,
    relative_to_dir: Optional[str] = None,
    verify_files_exist: bool = False,
) -> Dict[str, Any]:
    """Construct the complete validated Amendment A config JSON dict referencing the 144 B200 clips."""
    prompts_map = _prompt_by_id()
    b1_perms = compute_bws3_core_permutations()
    b2_orders = compute_2afc_core_orders()

    def clip_entry(arm: str, pid: str, seed: int) -> Dict[str, str]:
        fname = f"{arm}_cell_{pid}-s{seed}.mp4"
        if fname not in B200_144_SHA256:
            raise KeyError(f"Missing hash for {fname}")
        full_path = os.path.join(clips_dir, fname)
        if verify_files_exist:
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Real clip not found on disk: {full_path}")
            actual = hashlib.sha256(open(full_path, "rb").read()).hexdigest()
            if actual != B200_144_SHA256[fname]:
                raise ValueError(f"SHA-256 mismatch on {full_path}: {actual} != {B200_144_SHA256[fname]}")
        if relative_to_dir is not None:
            path_str = os.path.relpath(full_path, relative_to_dir).replace("\\", "/")
        else:
            path_str = full_path.replace("\\", "/")
        return {
            "path": path_str,
            "sha256": B200_144_SHA256[fname],
        }

    # Build Block 1 items (43 items: 3 warm-ups + 36 core + 4 repeats)
    block1_items: List[Dict[str, Any]] = []
    for idx, (pid, seed, is_warmup, is_repeat) in enumerate(BLOCK1_SEQUENCE):
        pmeta = prompts_map[pid]
        cell_id = f"{pid}-s{seed}"
        base_perm = b1_perms[(pid, seed)]
        if is_warmup:
            # Practice permutation shifted by (idx + 1)
            slot_arms = base_perm[1:] + base_perm[:1]
        elif is_repeat:
            # Cyclic shift (A, B, C) -> (B, C, A) from Pass 1 per Amendment A §1.2
            slot_arms = base_perm[1:] + base_perm[:1]
        else:
            slot_arms = list(base_perm)

        block1_items.append({
            "cell_id": f"{cell_id}-warmup" if is_warmup else (f"{cell_id}-rep2" if is_repeat else cell_id),
            "base_cell_id": cell_id,
            "prompt_id": pid,
            "prompt_type": pmeta["type"],
            "seed": seed,
            "prompt": pmeta["prompt"],
            "rater_cue": pmeta["rater_cue"],
            "required_action": pmeta["required_action"],
            "estimated_seconds": pmeta["estimated_seconds"],
            "estimated_max_seconds": pmeta["estimated_max_seconds"],
            "clip_seconds": 2.54,
            "is_warmup": is_warmup,
            "is_repeat": is_repeat,
            "repeat_of_cell_id": cell_id if is_repeat else None,
            "ask_q2": pmeta["type"] == "manipulation",
            "slot_arms": slot_arms,
            "break_after": (idx == 22),  # Mid-block rest break after Item 23 / 43
            "arms": {
                "BF16": clip_entry("BF16", pid, seed),
                "Q1": clip_entry("Q1", pid, seed),
                "Q3": clip_entry("Q3", pid, seed),
            },
        })

    # Build Block 2 items (42 items: 2 warm-ups + 36 core + 4 repeats)
    block2_items: List[Dict[str, Any]] = []
    for _idx, (pid, seed, is_warmup, is_repeat) in enumerate(BLOCK2_SEQUENCE):
        pmeta = prompts_map[pid]
        cell_id = f"{pid}-s{seed}"
        base_order = b2_orders[(pid, seed)]
        if is_warmup or is_repeat:
            # Swap Left/Right on repeat pass per ITU-T P.910 Annex C.4 and Amendment A §2.1
            slot_arms = [base_order[1], base_order[0]]
        else:
            slot_arms = list(base_order)

        block2_items.append({
            "cell_id": f"{cell_id}-b2-warmup" if is_warmup else (f"{cell_id}-b2-rep2" if is_repeat else f"{cell_id}-b2"),
            "base_cell_id": cell_id,
            "prompt_id": pid,
            "prompt_type": pmeta["type"],
            "seed": seed,
            "prompt": pmeta["prompt"],
            "rater_cue": pmeta["rater_cue"],
            "required_action": pmeta["required_action"],
            "estimated_seconds": pmeta["estimated_seconds"],
            "estimated_max_seconds": pmeta["estimated_max_seconds"],
            "clip_seconds": 2.54,
            "is_warmup": is_warmup,
            "is_repeat": is_repeat,
            "repeat_of_cell_id": f"{cell_id}-b2" if is_repeat else None,
            "ask_q2": False,
            "slot_arms": slot_arms,
            "arms": {
                "BF16": clip_entry("BF16", pid, seed),
                "NANO": clip_entry("NANO", pid, seed),
            },
        })

    cfg = {
        "schema_version": "blindpanel-config-v1",
        "study_id": "cosmos3-super-quant-amendment-a-v1",
        "title": "Cosmos 3 Super Quant Test — 3 Arms (Block 1: BF16, Q1, Q3 · 43 Items) + optional Nano block (Block 2: Nano vs BF16 · 42 Items)",
        "rng_seed": 20261030,
        "clip_spec": {
            "width": 832,
            "height": 480,
            "frames": 61,
            "fps": "24.0000",
            "seconds": 2.54,
            "hardware_note": "B200 Same-Card UUID",
        },
        "question2_spec": {
            "id": "action_completion",
            "prompt": AMENDMENT_A_Q2_PROMPT,
            "per_clip": True,
            "options": list(AMENDMENT_A_Q2_OPTIONS),
        },
        "blocks": [
            {
                "block_id": "block1_super_bws3",
                "title": "Main Block 1 — Pick Best & Worst of 3",
                "optional": False,
                "method": "bws_3",
                "arms": ["BF16", "Q1", "Q3"],
                "allow_unsure": False,
                "allow_skip": False,
                "require_full_play": True,
                "break_after_item": 23,
                "enforce_no_adjacent_prompt": True,
                "enforce_counterbalance_bws3": True,
                "items": block1_items,
            },
            {
                "block_id": "block2_nano_2afc",
                "title": "Block 2 · Side-by-Side Comparison",
                "optional": True,
                "method": "2afc",
                "arms": ["BF16", "NANO"],
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

    # Validate before returning
    validate_and_normalize_config(cfg)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Cosmos 3 Amendment A study config JSON.")
    parser.add_argument(
        "--clips-dir",
        default=DEFAULT_CLIPS_DIR,
        help="Directory containing the 144 extracted B200 MP4 clips (BF16_cell_*.mp4, Q1_*, Q3_*, NANO_*)",
    )
    parser.add_argument(
        "--out",
        default="configs/cosmos3_amendment_a.json",
        help="Output path for the generated study config JSON",
    )
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="Verify all 144 clip files exist on disk and match their SHA-256 hashes",
    )
    args = parser.parse_args()

    cfg = build_cosmos3_amendment_a_config(
        clips_dir=args.clips_dir,
        verify_files_exist=args.verify_files,
    )
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote validated Cosmos 3 Amendment A config (144 clips, Block 1=43 items, Block 2=42 items) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
