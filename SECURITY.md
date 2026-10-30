# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Threat Model & Blindness Guarantees

1. **Localhost Binding Only**: `blindpanel` refuses to bind to any non-loopback interface (`127.0.0.1` or `localhost` only).
2. **Sealed Map Isolation**: `sealed_map_*.bin` files are stored outside the static web root, excluded via `.gitignore`, and explicitly blocked by the HTTP handler (`HTTP 404`).
3. **Cryptographic Commitments**: Every trial row is salted with 128 bits of CSPRNG entropy (`secrets.token_hex(16)`) and committed via `SHA-256(canonical_json({"map": letter_to_arm, "salt": salt}))`. Offline unblinding verifies every row's `map_hash` before joining.
4. **Clip Verification at Load**: All media files are verified against their configured SHA-256 digests before the server opens a socket.

## Reporting a Vulnerability

Please report security issues privately via GitHub Security Advisories on `mcp-tool-shop-org/blindpanel`.
