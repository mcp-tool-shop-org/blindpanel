# blindpanel

**Offline, cryptographically sealed, schema-driven human blind-evaluation harness for video and visual generation studies.**

`blindpanel` (`v0.1.0`) is a local-only (`127.0.0.1`), zero-network-dependency evaluation server and offline unblinding toolkit built for rigorous perceptual and physical-plausibility studies (ITU-T BT.500-15, P.910, P.913).

---

## Key Features

1. **Zero Study Artifacts in Git (Sealed Session Design)**:
   - Committed study configs (`configs/cosmos3_amendment_a.json`) hold **only** cell IDs, prompts, rater cues, block and method settings, and timing rules — never per-item arm orders, clip paths, clip SHA-256 digests, seeds, salts, or absolute paths.
   - At session start, a fresh 256-bit ($\ge 128$-bit) cryptographic secret seed (`secrets.token_bytes(32)`) is generated and stored exclusively inside the gitignored `sealed_store/`.
   - Optional `--arm-map <path>` maps config pseudonyms (`ARM_1..ARM_4`) to local clip filename prefixes, read once at session start into the sealed store and never served.
   - From that secret seed come the per-item slot orders, the Latin-square / 6-permutation counterbalancing, and the per-session salted clip names (`clip_<salted_token>.mp4`).
   - Clips are served under those salted names from a **sealed manifest** (`arm`, `original_sha256`, `salted_name`) inside the sealed store, verified by SHA-256 at load (`REFUSED AT LOAD` on mismatch), and never sent to the browser.

2. **Schema-Driven Test Methods (`blindpanel-config-v1`)**:
   - **`bws_3`**: Single-step Best-Worst of 3 (picking **Best** and **Worst** automatically fixes **2nd / Middle**, producing a complete 3-arm ranking and all 3 pairwise comparisons per triple).
   - **`2afc`**: Synchronized side-by-side 2-Alternative Forced Choice (`A` vs `B`).
   - **`bws`**: Iterative Best-Worst Scaling (peel-off for $K \ge 3$ arms).
   - **`pairwise`**: Pairwise comparison with optional `Unsure` tie support (`allow_unsure: true`).
   - **`mos_likert`**: Absolute Category Rating (MOS / Likert 1–5 or custom scale).
   - **`abx`**: A/B/X reference discrimination (`Reference A`, `Reference B`, unknown `X`).
   - **Per-Row Question 2 (`ask_q2`)**: Configurable secondary question per clip on flagged rows (e.g., manipulation rows `M1–M4`: *"How far does the robot get with the action within the 2.5 s clip?"* with options *"Yes (completes it)"*, *"Partly (clearly under way)"*, *"No (not achieved)"*).
   - **Blocks, Warm-ups, Rest Breaks, Hidden Repeats & Counterbalancing**: Enforces Latin-square slot balance across variants, all 6 permutations across 36 core triples, non-adjacent prompt constraints ($\text{prompt}_t \ne \text{prompt}_{t-1}$), and hidden repeat spacing ($\ge 12$ intervening items with cyclic slot shift).

3. **Detail Visibility (`832×480` Native 1:1 & Synchronized Flip Mode)**:
   - On wide displays that fit $3 \times 832\,\text{px}$ plus gutters ($\ge 2548\,\text{px}$, such as a `3440×1440` ultrawide monitor), `blindpanel` defaults to **three native-size (`832×480`) tiles side-by-side**.
   - On narrower windows (`< 2548px`, such as `1440×900` or `1024×768`), `blindpanel` automatically falls back to **synchronized single-large-clip 1:1 Flip mode** (`A / B / C` flip via buttons or keys `1 / 2 / 3`), keeping all `<video>` elements frame-locked in time and playback state, while also allowing one-click switching to the **3-Up** side-by-side grid without losing time position or selected answers.

4. **Strict Rater/Operator Surface Separation & Leak Guard**:
   - **Separate Rater & Operator Ports**: The rater port serves only blind rater views and returns `404 Not Found` on all `/api/operator/*` and `/api/export/*` routes (with and without a token). The operator surface runs on a separate localhost port started only when `--operator-port` is passed, authenticated by a per-start token printed exclusively to the operator's terminal.
   - **Isolated Pilot Mode**: Writes to `pilot_answers_<date>.jsonl` apart from live `answers.jsonl`, running on the verified clips.
   - **Rater Screen Hygiene**: Zero study internals (no arm names, no `"repeat"`, `"warm-up"` counts, `"expected"`, or `"half"`) in any visible text, `title`, `aria-*`, `alt`, or HTML comment. Minimum font size $\ge 12\,\text{px}$ on `#808080` calibrated 50% sRGB grey surround, with a 700 ms `#808080` grey adaptation curtain between trials.
   - **Repository Leak Guard (`blindpanel.leak_guard`)**: Runs in CI and `scripts/verify.sh` over `git ls-files`, rejecting any per-item slot order, arm-map file, arm label near a hash or path, seed/salt key, absolute local path, or non-mock screenshot.

5. **100% Offline & Zero External UI Libraries**:
   - Uses Python's standard library for the runtime server and vanilla HTML/CSS/JS with system font stacks in the browser. Zero external UI libraries or CDN calls at runtime.

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Start the Local Evaluation Server

```bash
python -m blindpanel \
  --config configs/cosmos3_amendment_a.json \
  --clips-dir ./local_clips \
  --arm-map ./local_arm_map.json \
  --sealed-map sealed_store/sealed_map_session.bin \
  --answers answers.jsonl \
  --host 127.0.0.1 \
  --port 8765 \
  --operator-port 8766
```

- **Rater View**: `localhost:8765/?view=rater`
- **Pilot Mode**: `localhost:8765/?view=pilot`
- **Operator Surface** (separate port `8766`, requires `X-Operator-Token: <token>` printed to terminal): `/api/operator/preflight`, `/api/operator/audit`, `/api/operator/override`, `/api/export/csv`, `/api/export/jsonl`

### 3. Offline Unblinding

```bash
python -m blindpanel.unblind \
  --sealed-map sealed_store/sealed_map_session.bin \
  --answers answers.jsonl \
  --out unblinded_results.json
```

---

## Verification & Tests

Run the full lint, leak-guard, unit, security/leak-scan, and headless Chromium E2E test suite:

```bash
bash scripts/verify.sh
```

---

## License & Third-Party Notice

Licensed under the [MIT License](LICENSE).
`blindpanel` bundles zero third-party UI libraries, fonts, or runtime dependencies; all HTML, CSS, JavaScript, and Python runtime code is original MIT-licensed code.
