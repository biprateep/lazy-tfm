#!/usr/bin/env bash
# Executes the tutorials into notebooks and thumbnails, for the orphan
# `tutorials` branch.
#
# The tutorials are written here as py:percent scripts, so main tracks only
# their source. They need a GPU, so neither CI nor Read the Docs can run them:
# the executed notebooks, outputs and figures included, live on the orphan
# branch `tutorials`, which the docs build fetches (see docs/conf.py) and the
# Colab and GitHub buttons open. This script writes each executed notebook
# next to its script, and its gallery thumbnail to thumbnails/<name>.png
# (both ignored by git), ready to be copied there.
#
# Gallery metadata. docs/conf.py builds the cards of docs/tutorials/index.md
# from the scripts themselves, so a tutorial needs no other registration:
#
#   - title: the first Markdown heading of the script;
#   - description (optional, one line of Markdown): in the jupytext header,
#       # ---
#       # jupyter:
#       #   gallery:
#       #     description: Fit, predict and evaluate redshift PDFs.
#       #   jupytext:
#       #     notebook_metadata_filter: accelerator,colab,gallery
#     (`gallery` must be in the filter, or a jupytext sync drops it);
#   - thumbnail: the first PNG output of the cell tagged `thumbnail`,
#       # %% tags=["thumbnail"]
#     or, with no such cell, the last PNG in the notebook. It is scaled to
#     fit 400 x 300 px (4:3) and padded with white;
#   - order: TUTORIAL_ORDER in docs/conf.py; unlisted tutorials follow,
#     alphabetically.
#
# Typical usage, from the repository root on a machine with a GPU:
#
#   docs/tutorials/execute.sh                 # every tutorial
#   docs/tutorials/execute.sh introduction    # one
#
# then commit the notebooks and thumbnails to the branch:
#
#   git worktree add ../lazy-tutorials tutorials
#   cp docs/tutorials/*.ipynb ../lazy-tutorials/
#   mkdir -p ../lazy-tutorials/thumbnails
#   cp docs/tutorials/thumbnails/*.png ../lazy-tutorials/thumbnails/
#   git -C ../lazy-tutorials add -A
#   git -C ../lazy-tutorials commit -m "Re-execute the tutorials"
#
# To redo only the thumbnails of notebooks already executed, with no GPU,
# pass --thumbnails-only; --notebooks DIR reads the notebooks from DIR and
# writes DIR/thumbnails/ (default: this directory), for example
#
#   docs/tutorials/execute.sh --thumbnails-only --notebooks ../lazy-tutorials
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
execute=1
notebooks="$here"
names=()
while (($#)); do
  case "$1" in
    --thumbnails-only) execute=0 ;;
    --notebooks)
      notebooks="$(cd "$2" && pwd)"
      shift
      ;;
    -*)
      echo "unknown option: $1" >&2
      exit 2
      ;;
    *) names+=("$1") ;;
  esac
  shift
done
if ((!${#names[@]})); then
  for script in "$here"/*.py; do
    names+=("$(basename "$script" .py)")
  done
fi

for name in "${names[@]}"; do
  notebook="$notebooks/$name.ipynb"
  if ((execute)); then
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
  fi
  # The gallery thumbnail: the tagged cell's first figure, else the last one.
  thumbnail="$notebooks/thumbnails/$name.png"
  mkdir -p "$notebooks/thumbnails"
  uv run --no-sync python - "$notebook" "$thumbnail" <<'EOF'
import base64
import io
import sys

import nbformat
from PIL import Image

SIZE = (400, 300)  # 4:3, about the width of a card at three per row, x1.5.

path, out = sys.argv[1:]
nb = nbformat.read(path, as_version=4)
tagged, last = None, None
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    pngs = [
        o["data"]["image/png"]
        for o in cell.outputs
        if "image/png" in o.get("data", {})
    ]
    if pngs:
        last = pngs[-1]
        if tagged is None and "thumbnail" in cell.metadata.get("tags", []):
            tagged = pngs[0]
png = tagged or last
if png is None:
    print(f"{path} has no figure; its card shows a placeholder", file=sys.stderr)
    sys.exit()
image = Image.open(io.BytesIO(base64.b64decode(png))).convert("RGBA")
image.thumbnail(SIZE, Image.Resampling.LANCZOS)
card = Image.new("RGBA", SIZE, "white")
offset = ((SIZE[0] - image.width) // 2, (SIZE[1] - image.height) // 2)
card.alpha_composite(image, offset)
card.convert("RGB").save(out, optimize=True)
print(f"wrote {out}")
EOF
done
