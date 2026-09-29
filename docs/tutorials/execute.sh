#!/usr/bin/env bash
# Executes the tutorials into notebooks, for the orphan `tutorials` branch.
#
# The tutorials are written here as py:percent scripts, so main tracks only
# their source. They need a GPU, so neither CI nor Read the Docs can run them:
# the executed notebooks, outputs and figures included, live on the orphan
# branch `tutorials`, which the docs build fetches (see docs/conf.py) and the
# Colab and GitHub buttons open. This script writes each executed notebook
# next to its script (ignored by git), ready to be copied there.
#
# Typical usage, from the repository root on a machine with a GPU:
#
#   docs/tutorials/execute.sh                 # every tutorial
#   docs/tutorials/execute.sh introduction    # one
#
# then commit the notebooks to the branch:
#
#   git worktree add ../lazy-tutorials tutorials
#   cp docs/tutorials/*.ipynb ../lazy-tutorials/
#   git -C ../lazy-tutorials commit -am "Re-execute the tutorials"
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
if (($#)); then
  names=("$@")
else
  names=()
  for script in "$here"/*.py; do
    names+=("$(basename "$script" .py)")
  done
fi

for name in "${names[@]}"; do
  notebook="$here/$name.ipynb"
  uv run --no-sync jupytext --to ipynb --output "$notebook" "$here/$name.py"
  uv run --no-sync jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 "$notebook"
  # Progress bars are ipywidgets, which a static page cannot draw: without
  # their live state they render as a bar stuck at 0%. Drop them.
  uv run --no-sync python - "$notebook" <<'EOF'
import sys

import nbformat

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
widget = "application/vnd.jupyter.widget-view+json"
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = [
            out for out in cell.outputs if widget not in out.get("data", {})
        ]
nb.metadata.pop("widgets", None)
nbformat.write(nb, path)
EOF
  echo "wrote $notebook"
done
