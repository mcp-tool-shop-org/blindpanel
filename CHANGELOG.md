# Changelog

All notable changes to `blindpanel` will be documented in this file.

## [0.1.0] - 2026-10-30

### Added
- Versioned configuration schema validator (`blindpanel-config-v1`) supporting `bws_3` (single-step Best-Worst of 3), `2afc`, iterative `bws`, `pairwise` (with optional `Unsure`), `mos_likert`, and `abx`.
- Per-row secondary Question 2 (`ask_q2`) for manipulation prompts (`M1–M4`) with Amendment A wording and options.
- Counterbalancing and sequence validator enforcing 6-permutation balance across 36 core triples, per-prompt Latin squares across seeds, non-adjacent prompt IDs, and hidden repeat spacing ($\ge 12$ intervening items with cyclic slot shift).
- Detail visibility engine: automatic 3-up native `832×480` layout on wide windows ($\ge 2548\,\text{px}$) and synchronized single-large-clip `1:1 Flip` fallback (`A / B / C` via keys `1 / 2 / 3`) on narrower windows, plus manual toggle (`1:1 Flip` / `3-Up`) preserving playback time and selections.
- Operator Pre-Flight screen with clip-duration feasibility checks and logged override audit gate, isolated Pilot Mode (`pilot_answers_<date>.jsonl`), and visible cryptographic integrity strip.
- Rater screen hygiene enforcement (zero study internals in text, attributes, or comments; $\ge 12\,\text{px}$ minimum font size; `#808080` surround; all 4 `BUILD-MUST-DO.md` items).
- First study configuration (`configs/cosmos3_amendment_a.json`) built and verified against the 144 real B200 clips.
- Offline unblinding CLI (`blindpanel.unblind`) with per-row salt/hash verification, warm-up exclusion, Q2 tallies, and hidden-repeat pairwise concordance.
- Unit, security/leak-scan, and headless Chromium E2E test suite plus GitHub Actions CI with Codecov OIDC coverage and JUnit test report uploads.
