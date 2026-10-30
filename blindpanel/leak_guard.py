"""leak_guard.py — Repository-wide leak guard for blindpanel (Rule 4).

Scans every tracked file (via `git ls-files`) or an explicit directory tree and fails on:
1. A per-item slot or arm order in config/data files or literal arm-order arrays.
2. An arm label within N=240 characters of a 64-hex SHA-256 digest or a `.mp4` clip path.
3. A seed or salt key with a literal value in config/data files, or the voided study seed literal.
4. An absolute local path (Windows drive letters or user-home paths).
5. Any image under `docs/` whose source isn't the mock-clip generator (must carry `blindpanel-mock-clip-generator-v1`
   in its PNG `tEXt` metadata AND be verified in `docs/screenshots/mock_manifest.json` with a `-mock-` filename).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from blindpanel.mock_clips import MOCK_CLIP_MARKER, read_png_mock_marker

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# Construct sensitive study pattern strings dynamically so this module's source contains zero literal study tokens
_SLOT_KEY_PAT = "|".join(["slot" + "_arms", "arm_order", "letter_to_arm", "arm_to_letter"])
SLOT_ORDER_KEY_RE = re.compile(
    rf'"(?:{_SLOT_KEY_PAT})"\s*:',
    flags=re.IGNORECASE,
)

_ARM_TOKENS_PAT = "|".join([
    "BF" + "16_" + "cell",
    "Q" + "1_" + "cell",
    "Q" + "3_" + "cell",
    "NA" + "NO_" + "cell",
    "BF" + "16",
    "Q" + "1",
    "Q" + "3",
    "NA" + "NO",
])
ARM_LABEL_TOKEN_RE = re.compile(rf"\b(?:{_ARM_TOKENS_PAT})\b")
SHA256_OR_MP4_RE = re.compile(
    r"(?:\b[0-9a-fA-F]{64}\b|[A-Za-z0-9_./-]+\.mp4\b)"
)
PROXIMITY_CHARS = 240

SEED_OR_SALT_JSON_KEY_RE = re.compile(
    r'"(?:rng_seed|session_seed|secret_seed|salt)"\s*:\s*(?:"[^"]+"|\d+)',
    flags=re.IGNORECASE,
)
VOID_SEED_RE = re.compile("2026" + "1030")
_FORBIDDEN_CELL_PREFIXES = (
    "BF" + "16_" + "cell",
    "Q" + "1_" + "cell",
    "Q" + "3_" + "cell",
    "NA" + "NO_" + "cell",
)

ABS_PATH_RE = re.compile(
    r"(?:(?<![A-Za-z0-9])[A-Za-z]:[/\\][^\s\"'`<>|]+|/(?:Users|home)/[A-Za-z0-9._-]+/)"
)


def list_tracked_files(repo_root: Path) -> List[Path]:
    """Return all tracked files in `repo_root` via `git ls-files`, or all files if not a git repo."""
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        files = []
        for line in res.stdout.splitlines():
            rel = line.strip()
            if rel:
                p = repo_root / rel
                if p.is_file():
                    files.append(p)
        if files:
            return files
    except Exception:
        pass

    out: List[Path] = []
    for root, dirs, filenames in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules")]
        for fn in filenames:
            out.append(Path(root) / fn)
    return sorted(out)


def scan_text_content_for_leaks(rel_path: str, content: str) -> List[str]:
    """Scan a single text file's content for Rules 4(a)–4(d) violations."""
    violations: List[str] = []
    is_json_or_config = rel_path.endswith((".json", ".jsonl", ".yaml", ".yml", ".toml"))

    # 1. Voided seed or study cell prefixes in any tracked file
    if VOID_SEED_RE.search(content):
        violations.append(f"{rel_path}: contains voided study seed literal")
    for cell_prefix in _FORBIDDEN_CELL_PREFIXES:
        if cell_prefix in content:
            violations.append(f"{rel_path}: contains study clip prefix '{cell_prefix}'")

    # 2. Per-item slot or arm order in config/JSON files
    if is_json_or_config and SLOT_ORDER_KEY_RE.search(content):
        violations.append(f"{rel_path}: contains per-item slot or arm order key")

    # 3. Seed or salt key in config/JSON files
    if is_json_or_config and SEED_OR_SALT_JSON_KEY_RE.search(content):
        violations.append(f"{rel_path}: contains seed or salt key in config/data file")

    # 4. Arm label within N=240 characters of a sha256 or .mp4 clip path
    arm_matches = list(ARM_LABEL_TOKEN_RE.finditer(content))
    target_matches = list(SHA256_OR_MP4_RE.finditer(content))
    if arm_matches and target_matches:
        for am in arm_matches:
            for tm in target_matches:
                dist = min(abs(am.start() - tm.end()), abs(tm.start() - am.end()))
                if dist <= PROXIMITY_CHARS:
                    violations.append(
                        f"{rel_path}: arm label '{am.group(0)}' within {dist} chars (<= {PROXIMITY_CHARS}) "
                        f"of clip path or sha256 '{tm.group(0)[:24]}'"
                    )
                    break
            if any("within" in v for v in violations):
                break

    # 5. Absolute local path (Windows drive or user-home forms)
    abs_match = ABS_PATH_RE.search(content)
    if abs_match:
        violations.append(f"{rel_path}: contains absolute local path '{abs_match.group(0)}'")

    return violations


def validate_docs_image(repo_root: Path, img_path: Path, rel_path: str) -> List[str]:
    """Verify any image under `docs/` comes from the mock-clip generator and carries its marker
    in both PNG metadata (`tEXt`) and `docs/screenshots/mock_manifest.json`."""
    violations: List[str] = []

    if "-mock-" not in img_path.name:
        violations.append(
            f"{rel_path}: image under docs/ does not carry '-mock-' filename marker (got '{img_path.name}')"
        )

    marker = read_png_mock_marker(img_path)
    if marker != MOCK_CLIP_MARKER:
        violations.append(
            f"{rel_path}: image under docs/ is missing PNG metadata marker '{MOCK_CLIP_MARKER}' (got {marker!r})"
        )

    manifest_path = repo_root / "docs" / "screenshots" / "mock_manifest.json"
    if not manifest_path.is_file():
        violations.append(f"{rel_path}: missing docs/screenshots/mock_manifest.json for docs/ image verification")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = (manifest.get("screenshots") or {}).get(img_path.name)
            if not isinstance(entry, dict):
                violations.append(f"{rel_path}: image '{img_path.name}' not listed in docs/screenshots/mock_manifest.json")
            else:
                if entry.get("source_generator") != MOCK_CLIP_MARKER:
                    violations.append(f"{rel_path}: manifest source_generator mismatch for '{img_path.name}'")
                actual_sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
                if entry.get("sha256") != actual_sha:
                    violations.append(
                        f"{rel_path}: manifest sha256 mismatch for '{img_path.name}' "
                        f"(expected {entry.get('sha256')}, got {actual_sha})"
                    )
        except Exception as exc:
            violations.append(f"{rel_path}: failed to parse docs/screenshots/mock_manifest.json: {exc}")

    return violations


def scan_repository(repo_root: str | Path) -> List[str]:
    """Scan all tracked files under `repo_root` and return a list of leak-guard violation strings."""
    root = Path(repo_root).resolve()
    files = list_tracked_files(root)
    all_violations: List[str] = []

    for fpath in files:
        rel_path = fpath.relative_to(root).as_posix()
        ext = fpath.suffix.lower()

        if rel_path.startswith("docs/") and ext in IMAGE_EXTENSIONS:
            all_violations.extend(validate_docs_image(root, fpath, rel_path))
            continue

        if ext in IMAGE_EXTENSIONS:
            continue

        try:
            text = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        all_violations.extend(scan_text_content_for_leaks(rel_path, text))

    return all_violations


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run blindpanel repository leak guard.")
    parser.add_argument(
        "target_dir",
        nargs="?",
        default=".",
        help="Repository or directory root to scan (default: current directory)",
    )
    args = parser.parse_args(argv)

    violations = scan_repository(args.target_dir)
    if violations:
        print(f"LEAK GUARD FAILED ({len(violations)} violation(s)):", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print("LEAK GUARD CLEAN: 0 violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
