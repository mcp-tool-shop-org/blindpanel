# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Threat Model & Blindness Guarantees

1. **Localhost Binding Only**: `blindpanel` refuses to bind to any non-loopback interface (`127.0.0.1` or `localhost` only).
2. **No Study Blinding Data in Git**: Committed study configurations hold only cell IDs, prompts, rater cues, block/method settings, and timing rules. Per-item slot orders, clip paths, clip SHA-256 digests, and session seeds are never committed to the repository (`blindpanel.leak_guard`).
3. **Session Secret Seed & Sealed Store Isolation**: At session start, a fresh 256-bit ($\ge 128$-bit) cryptographic secret seed (`secrets.token_bytes(32)`) is generated and stored solely inside the gitignored `sealed_store/`. It is never logged, printed, or served over HTTP (`HTTP 404`).
4. **Sealed Manifest & Salted Clip Names**: Clips are served under per-session salted names (`clip_<salted_token>.mp4`) backed by a sealed manifest (`arm`, `original_sha256`, `salted_name`) inside the sealed store, verified by SHA-256 at load before the server opens a socket.
5. **Cryptographic Commitments**: Every trial row is salted and committed via `SHA-256(canonical_json({"map": letter_to_arm, "salt": salt}))`. Offline unblinding (`blindpanel.unblind`) verifies every row's `map_hash` against the sealed store before joining.

## Reporting a Vulnerability

Please report security issues privately via GitHub Security Advisories on `mcp-tool-shop-org/blindpanel`.
