#!/usr/bin/env bash
set -euo pipefail

echo "=== 1. Ruff Lint Check ==="
python -m ruff check blindpanel tests scripts

echo "=== 2. Pytest Unit, Security/Leak, and Browser E2E Suite ==="
python -m pytest \
  --cov=blindpanel \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml \
  --junitxml=junit.xml

echo "=== Verification Passed ==="
