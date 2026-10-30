# blindpanel

**Offline, cryptographically sealed, schema-driven human blind-evaluation harness for video and visual generation studies.**

`blindpanel` (`v0.1.0`) is a local-only (`127.0.0.1`), zero-network-dependency evaluation server and offline unblinding toolkit built for rigorous perceptual and physical-plausibility studies (ITU-T BT.500-15, P.910, P.913).

---

## Key Features

1. **Schema-Driven Test Methods (`blindpanel-config-v1`)**:
   - **`bws_3`**: Single-step Best-Worst of 3 (picking **Best** and **Worst** automatically fixes **2nd / Middle**, producing a complete 3-arm ranking and all 3 pairwise comparisons per triple).
   - **`2afc`**: Synchronized side-by-side 2-Alternative Forced Choice (`A` vs `B`).
   - **`bws`**: Iterative Best-Worst Scaling (peel-off for $K \ge 3$ arms).
   - **`pairwise`**: Pairwise comparison with optional `Unsure` tie support (`allow_unsure: true`).
   - **`mos_likert`**: Absolute Category Rating (MOS / Likert 1–5 or custom scale).
   - **`abx`**: A/B/X reference discrimination (`Reference A`, `Reference B`, unknown `X`).
   - **Per-Row Question 2 (`ask_q2`)**: Configurable secondary question per clip on flagged rows (e.g., manipulation rows `M1–M4`: *"How far does the robot get with the action within the 2.5 s clip?"* with options *"Yes (completes it)"*, *"Partly (clearly under way)"*, *"No (not achieved)"*).
   - **Blocks, Warm-ups, Rest Breaks, Hidden Repeats & Counterbalancing**: Validates Latin-square slot balance across seeds, all 6 permutations across 36 core triples, non-adjacent prompt constraints ($\text{prompt}_t \ne \text{prompt}_{t-1}$), and hidden repeat spacing ($\ge 12$ intervening items with cyclic slot shift).

2. **Detail Visibility (`832×480` Native 1:1 & Synchronized Flip Mode)**:
   - On wide displays that fit $3 \times 832\,\text{px}$ plus gutters ($\ge 2548\,\text{px}$, such as a `3440×1440` ultrawide monitor), `blindpanel` defaults to **three native-size (`832×480`) tiles side-by-side**.
   - On narrower windows (`< 2548px`, such as `1440×900` or `1024×768`), `blindpanel` automatically falls back to **synchronized single-large-clip 1:1 Flip mode** (`A / B / C` flip via buttons or keys `1 / 2 / 3`), keeping all `<video>` elements frame-locked in time and playback state, while also allowing one-click switching to the **3-Up** side-by-side grid without losing time position or selected answers.

3. **Study Honesty & Cryptographic Blindness**:
   - **SHA-256 & Clip Shape Verification at Load**: Refuses to start (`REFUSED AT LOAD`) if any clip file is missing, fails its SHA-256 hash check, or mismatches expected geometry/frame count.
   - **Sealed Identity Map (`sealed_map_*.bin`)**: Generated with 128-bit per-row cryptographic salts and `map_hash` commitments. Never served to the browser (any HTTP request matching `sealed_map` or `.bin` returns `404`).
   - **Operator Pre-Flight & Logged Override Gate**: Displays required actions and estimated durations against actual clip duration; blocks live session start if any item is flagged unless a logged operator override reason is recorded to the audit log.
   - **Isolated Pilot Mode**: Writes to `pilot_answers_<date>.jsonl` apart from live `answers.jsonl`, running on the real verified clips.
   - **Rater Screen Hygiene**: Zero study internals (no arm names, no `"repeat"`, `"warm-up"` counts, `"expected"`, or `"half"`) in any visible text, `title`, `aria-*`, `alt`, or HTML comment. Minimum font size $\ge 12\,\text{px}$ on `#808080` calibrated 50% sRGB grey surround.

4. **100% Offline & Zero External UI Libraries**:
   - Uses Python's standard library for the runtime server and vanilla HTML/CSS/JS with system font stacks in the browser. Zero external UI libraries or CDN calls at runtime.

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Generate or Validate the Cosmos 3 Amendment A Study Config

```bash
python -m blindpanel.cosmos3_study \
  --clips-dir /path/to/b200/mp4 \
  --out configs/cosmos3_amendment_a.json \
  --verify-files
```

### 3. Start the Local Evaluation Server

```bash
python -m blindpanel \
  --config configs/cosmos3_amendment_a.json \
  --sealed-map sealed_store/sealed_map_20261030.bin \
  --answers answers.jsonl \
  --host 127.0.0.1 \
  --port 8765
```

- **Rater View**: `http://127.0.0.1:8765/?view=rater`
- **Pilot Mode**: `http://127.0.0.1:8765/?view=pilot`
- **Operator Pre-Flight**: `http://127.0.0.1:8765/?view=preflight`
- **Operator Post-Session Audit**: `http://127.0.0.1:8765/?view=audit`

### 4. Offline Unblinding

```bash
python -m blindpanel.unblind \
  --sealed-map sealed_store/sealed_map_20261030.bin \
  --answers answers.jsonl \
  --out unblinded_results.json
```

---

## Verification & Tests

Run the full lint, unit, security/leak-scan, and headless Chromium E2E test suite:

```bash
bash scripts/verify.sh
```

---

## License & Third-Party Notice

Licensed under the [MIT License](LICENSE).
`blindpanel` bundles zero third-party UI libraries, fonts, or runtime dependencies; all HTML, CSS, JavaScript, and Python runtime code is original MIT-licensed code.
