# Changelog

All notable changes to `blindpanel` will be documented in this file.

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
