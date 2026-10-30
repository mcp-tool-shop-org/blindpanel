# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Threat Model & Blindness Guarantees

1. **Localhost Binding Only**: `blindpanel` refuses to bind to any non-loopback interface (`127.0.0.1` or `localhost` only).
2. **No Study Blinding Data in Git**: Committed study configurations hold only cell IDs, prompts, rater cues, block/method settings, and timing rules. Per-item slot orders, arm-map files, clip paths, clip SHA-256 digests, and session seeds are never committed to the repository (`blindpanel.leak_guard`).
3. **Separate Rater & Operator Ports**: The rater port serves zero `/api/operator/*` or `/api/export/*` routes (`HTTP 404` with and without a token) and ships zero test hooks (`window.__blindpanel`, `?unlock=1`) in production mode. The operator surface binds to a separate localhost port only when `--operator-port` is passed and requires a per-start cryptographic token printed exclusively to the operator's terminal.
4. **Session Secret Seed & Sealed Store Isolation**: At session start, a fresh 256-bit ($\ge 128$-bit) cryptographic secret seed (`secrets.token_bytes(32)`) is generated and stored solely inside the gitignored `sealed_store/`. It is never logged, printed, or served over HTTP (`HTTP 404`).
5. **Sealed Manifest & Salted Clip Names**: Clips are served under per-session salted names (`clip_<salted_token>.mp4`) backed by a sealed manifest (`arm`, `original_sha256`, `salted_name`) inside the sealed store, verified by SHA-256 at load before the server opens a socket.
6. **Cryptographic Commitments**: Every trial row is salted and committed via `SHA-256(canonical_json({"map": letter_to_arm, "salt": salt}))`. Offline unblinding (`blindpanel.unblind`) verifies both `sealed_manifest` integrity (`sealed_manifest_sha256` and per-entry `original_sha256`) and every row's `map_hash` against the sealed store before joining.

## Reporting a Vulnerability

Please report security issues privately via GitHub Security Advisories on `mcp-tool-shop-org/blindpanel`.
