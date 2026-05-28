#!/usr/bin/env bash
# Export a clean repository zip without generated artifacts.
# Usage: ./scripts/export_clean_repo.sh [output.zip]
set -euo pipefail

OUT="${1:-neural-search-clean.zip}"
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

rm -f "$OUT"
git archive --format=zip --output="$OUT" HEAD

# Verify export doesn't contain generated artifacts
if unzip -l "$OUT" | grep -qE 'node_modules|__pycache__|\.pyc|\.git/|/dist/'; then
    echo "WARNING: Export may contain generated artifacts" >&2
fi

SIZE=$(du -h "$OUT" | cut -f1)
printf "Wrote %s (%s)\n" "$OUT" "$SIZE"
