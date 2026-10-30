# Changelog

All notable changes to `blindpanel` will be documented in this file.

## [Unreleased]

### Fixed
- Session Complete card counted the wrong block: after Block 1 (43 items) with Block 2 declined it read `42 / 42` (the last Block 2 item's total, since the cursor rests on the first unrated item); the card and the block-transition card now use totals the server reports per finished block (`completed_blocks`, `prev_block_total_items`), so Block 1 ends as `43 / 43` and the hardcoded `43` is gone. Totals are the same ones the session counter uses; warm-ups and repeats are never broken out.
- Three native tiles are now the default whenever the window fits them: the `3-Up` toggle and `?layout=3up` gave the scaled 456 px grid even at `3440×1440`. The `<video>` is exactly `832×480` CSS px in native and single-clip layouts (the tile is 834×482 around its 1 px border), the breakpoint is `2554 px`, and the zoom buttons show `1:1` as active there (`Fit` stays as a manual option).
- Wide-screen text scale: every font size and control height is `calc(<px> * var(--ui-scale))`, 1.25 at `>= 2554 px`, so the smallest control text at `3440×1440` is 15 px (was 12 px) while `1440×900` and `1024×768` are unchanged.
- The rest-break and block-transition cards' "recommended rest remaining" timer now counts down from 10:00 instead of showing a fixed `08 : 42`.

### Added
- Mock screenshot set now includes the Block 2 optional card and the `3440×1440` native 3-up and flip screens.
- Tests for the three cards' counts, the native tile width and zoom state at `3440×1440`, the flip default at `1440×900`, and the `>= 14 px` text floor at `3440×1440`.

## [0.1.0] - 2026-10-30

### Added
- Versioned configuration schema validator (`blindpanel-config-v1`) supporting `bws_3` (single-step Best-Worst of 3), `2afc`, iterative `bws`, `pairwise` (with optional `Unsure`), `mos_likert`, and `abx`.
- Sealed session blinding architecture: public study configs (`configs/cosmos3_amendment_a.json`) hold only cell IDs, prompts, rater cues, block/method settings, and timing rules; at session start, a fresh $\ge 128$-bit `secrets` seed generates per-item slot orders, Latin-square / 6-permutation counterbalancing, and per-session salted clip names served from a sealed manifest inside `sealed_store/`.
- Repository leak guard (`blindpanel.leak_guard`) enforced in `scripts/verify.sh`, pytest, and CI over `git ls-files`.
- Synthetic mock-clip generator and PNG metadata marker (`blindpanel.mock_clips`).
- Per-row secondary Question 2 (`ask_q2`) for manipulation prompts (`M1–M4`) with Amendment A wording and options.
- Detail visibility engine: automatic 3-up native `832×480` layout on wide windows ($\ge 2548\,\text{px}$) and synchronized single-large-clip `1:1 Flip` fallback (`A / B / C` via keys `1 / 2 / 3`) on narrower windows, plus manual toggle (`1:1 Flip` / `3-Up`) preserving playback time and selections.
- Operator Pre-Flight screen with clip-duration feasibility checks and logged override audit gate, isolated Pilot Mode (`pilot_answers_<date>.jsonl`), and visible cryptographic integrity strip.
- Rater screen hygiene enforcement (zero study internals in text, attributes, or comments; $\ge 12\,\text{px}$ minimum font size; `#808080` surround; all 4 `BUILD-MUST-DO.md` items).
- Offline unblinding CLI (`blindpanel.unblind`) reading only the sealed store and session answers, with per-row salt/hash verification, warm-up exclusion, Q2 tallies, and hidden-repeat pairwise concordance.
