#!/usr/bin/env bash
# Checks what the built archives in dist/ hold: the wheel marks the package
# typed, and the sdist carries what its tests need, so that distributors can
# run them from it.
set -euo pipefail
wheel=$(python3 -m zipfile -l dist/*.whl)
grep -q "lazy/py.typed" <<<"$wheel" || { echo "wheel lacks py.typed" >&2; exit 1; }
listing=$(tar -tzf dist/*.tar.gz)
for path in tests/lazy/conftest.py docs/models/index.md CHANGELOG.md \
    CITATION.cff; do
  grep -q "/$path$" <<<"$listing" || { echo "sdist lacks $path" >&2; exit 1; }
done
