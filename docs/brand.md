# Brand

The name, the logo files and the colors of LAZY.

## The name

LAZY stands for "Lazy but Accurate Z\* for Yinz". It has three spellings,
each for one use.

| Spelling | Use |
| --- | --- |
| LAZY | The project, in prose. |
| `lazy-tfm` | The distribution: `pip install lazy-tfm`, PyPI, version numbers. |
| `lazy` | The import name and the `lazy` command, in code font. |

## The logo

The logo spells "LAZY" as one row of a table. The L, the A and the Y are solid
lines, the training data. The Z is predicted points with error bars, which
grow with the distance from the solid strokes. The square icon is the Z alone.
`docs/logo/make_logo.py` draws every file, transparent, into `docs/_static/`.

| File | Content | Use |
| --- | --- | --- |
| `lazy-logo.svg`, `lazy-logo.png` | The logo in blue (PNG: 1730 by 589 px) | On a light background. |
| `lazy-logo-white.svg`, `lazy-logo-white.png` | The logo in white (PNG: 1730 by 589 px) | On a dark background. |
| `lazy-icon.svg`, `lazy-icon.png` | The Z alone, in blue (PNG: 474 by 474 px) | A square space, such as a social profile. |
| `favicon.svg` | The icon, in a light tint in a dark browser theme | The browser tab. |
| `favicon.png` | The icon at 64 px | The browser tab, where SVG is not read. |
| `apple-touch-icon.png` | The icon at 180 px | The home-screen icon on Apple devices. |

Prefer the SVG files, which stay sharp at any size. This site shows the blue
logo in the light theme and the white one in the dark theme.

## The colors

| Name | Value | Use |
| --- | --- | --- |
| University of Toronto blue | `#1E3765` | The logo and the icon on a light background. |
| White | `#FFFFFF` | The logo on a dark background. |
| Light tint | `#A9BFE8` | The SVG favicon in a dark browser theme. |

## Regenerating the files

Edit `docs/logo/make_logo.py`, then run it from the repository root:

```bash
uv run python docs/logo/make_logo.py
```

The output is deterministic: a second run changes no file.
