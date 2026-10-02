# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # The lazy logo
#
# "LAZY" in monoline script capitals. The L, the A and the Y are solid lines:
# the training data. The Z is predicted points with error bars, each bar
# growing with its distance from the nearest solid stroke, as a model's
# uncertainty grows away from its data.
#
# The letters are drawn here, not set in a font: each is a few smooth curves
# through hand-placed points, read off a script-capital reference and split at
# the letter's sharp corners so the corners stay crisp. Running this script
# rewrites every logo file in `docs/_static/`:
#
#     python docs/logo/make_logo.py
#
# The colour is the University of Toronto's blue, #1E3765.

# %%
import pathlib

import matplotlib as mpl

mpl.use("Agg")

from matplotlib import axes as mpl_axes
from matplotlib import figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

#: The University of Toronto's blue.
UOFT_BLUE = "#1E3765"

#: Where the rendered files go.
OUT = pathlib.Path(__file__).resolve().parents[1] / "_static"

#: Each letter's strokes, as points in the reference's pixel coordinates (y
#: downwards). A letter is split into strokes at its sharp corners.
# fmt: off
LETTERS: dict[str, list[list[tuple[float, float]]]] = {
    "L": [
        # The curl, the top loop and the stem, to the bottom-left corner.
        [(690, 227), (702, 229), (713, 224), (718, 212), (712, 200),
         (698, 195), (683, 199), (672, 211), (665, 230), (658, 250),
         (649, 265), (635, 281)],
        # The foot, flicking out to the right.
        [(635, 281), (652, 272), (670, 266), (685, 265), (694, 269),
         (699, 274)],
    ],
    "A": [
        [(37, 117), (55, 123), (77, 122), (97, 112), (116, 94), (135, 72),
         (153, 56), (168, 49), (176, 48)],
        [(176, 48), (162, 59), (149, 75), (139, 95), (133, 114), (134, 127),
         (142, 131)],
        [(117, 90), (141, 90)],
    ],
    "Z": [
        [(838, 511), (843, 502), (857, 495), (878, 495), (898, 499),
         (914, 494)],
        [(914, 494), (814, 573)],
        [(814, 573), (830, 566), (848, 561), (866, 562), (880, 566),
         (889, 566)],
    ],
    "Y": [
        [(697, 507), (711, 499), (724, 492)],
        [(724, 492), (726, 518), (727, 543), (728, 565)],
        [(775, 491), (761, 505), (746, 530), (731, 562), (716, 584),
         (700, 597), (688, 600), (681, 596), (683, 590), (690, 589)],
    ],
}
# fmt: on

#: How far each letter sits above the shared baseline; the Y's descender
#: drops below it.
BASELINE_SHIFT = {"L": 0, "A": -6, "Z": -2, "Y": 22}

#: Space between neighbouring letters' extents; negative tucks them together.
GAP = -4

#: The Z's error bars on each of its strokes, as a fraction of their full
#: length: the bottom stroke sits under the diagonal, so its bars stay short.
Z_BAR_SCALE = (1.0, 1.0, 0.5)


# %% [markdown]
# ## Geometry


# %%
def spline(
    points: list[tuple[float, float]], n: int = 500, smooth: float = 2.0
) -> np.ndarray:
    """A smooth curve through a stroke's points, shape (n, 2)."""
    pts = np.array(points, float)
    if len(pts) == 2:
        t = np.linspace(0, 1, n)[:, None]
        return pts[0] + t * (pts[1] - pts[0])
    tck, _ = interpolate.splprep(
        pts.T, s=smooth * len(pts), k=min(3, len(pts) - 1)
    )
    return np.column_stack(interpolate.splev(np.linspace(0, 1, n), tck))


def layout(gap: float = GAP) -> dict[str, list[np.ndarray]]:
    """Each letter's curves, side by side, y upwards, on one baseline."""
    placed, x = {}, 0.0
    for name in "LAZY":
        curves = [spline(stroke) for stroke in LETTERS[name]]
        points = np.vstack(curves)
        x0, bottom = points[:, 0].min(), points[:, 1].max()
        placed[name] = [
            np.column_stack(
                [c[:, 0] - x0 + x, bottom - c[:, 1] - BASELINE_SHIFT[name]]
            )
            for c in curves
        ]
        x += points[:, 0].max() - x0 + gap
    return placed


def z_points(
    strokes: list[np.ndarray], spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced points along the Z, and the stroke each lies on."""
    points, stroke_of = [], []
    for k, curve in enumerate(strokes):
        arc = np.r_[0, np.cumsum(np.hypot(*np.diff(curve, axis=0).T))]
        s = np.linspace(0, arc[-1], max(round(arc[-1] / spacing), 1) + 1)
        for p in zip(
            np.interp(s, arc, curve[:, 0]),
            np.interp(s, arc, curve[:, 1]),
            strict=True,
        ):
            # A corner is shared with the previous stroke: one point there.
            if (
                points
                and np.min(np.hypot(*(np.array(points) - p).T)) < 0.7 * spacing
            ):
                continue
            points.append(p)
            stroke_of.append(k)
    return np.array(points), np.array(stroke_of)


# %% [markdown]
# ## Drawing


# %%
def draw(
    ax: mpl_axes.Axes,
    colour: str = UOFT_BLUE,
    *,
    gap: float = GAP,
    lw: float = 4.5,
    spacing: float = 11.0,
    min_bar: float = 1.5,
    max_bar: float = 14.0,
    ms: float = 4.5,
    eb: float = 1.3,
    cap: float = 2.6,
) -> None:
    """Draws the logo on ``ax`` in one colour."""
    placed = layout(gap)
    for name in "LAY":
        for curve in placed[name]:
            ax.plot(
                *curve.T,
                color=colour,
                lw=lw,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
    data = np.vstack([c for name in "LAY" for c in placed[name]])
    points, stroke_of = z_points(placed["Z"], spacing)
    distance = np.min(
        np.hypot(*(points[:, None] - data[None]).transpose(2, 0, 1)), axis=1
    )
    for p, d, k in zip(points, distance, stroke_of, strict=True):
        grow = (d / distance.max()) ** 0.9 * Z_BAR_SCALE[k]
        ax.errorbar(
            *p,
            yerr=min_bar + (max_bar - min_bar) * grow,
            fmt="o",
            ms=ms,
            mew=0.5,
            color=colour,
            elinewidth=eb,
            capsize=cap,
            capthick=eb,
        )
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")


def draw_icon(ax: mpl_axes.Axes, colour: str = UOFT_BLUE) -> None:
    """The square icon: the predicted Z alone, bolder, for small sizes."""
    placed = layout()
    z = np.vstack(placed["Z"])
    centre = (z.min(axis=0) + z.max(axis=0)) / 2
    strokes = [c - centre for c in placed["Z"]]
    points, stroke_of = z_points(strokes, 24.0)
    # No data in the icon: bars grow from the Z's two ends to the middle of
    # its length, which is the middle of the diagonal.
    walked = np.r_[0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]
    for p, t, k in zip(points, walked / walked[-1], stroke_of, strict=True):
        grow = (1 - abs(2 * t - 1)) ** 0.9 * Z_BAR_SCALE[k]
        ax.errorbar(
            *p,
            yerr=4 + 14 * grow,
            fmt="o",
            ms=10,
            mew=1.0,
            color=colour,
            elinewidth=3.0,
            capsize=5.0,
            capthick=3.0,
        )
    half = np.abs(np.vstack(strokes)).max() + 12
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.axis("off")


def save(fig: mpl_figure.Figure, stem: str) -> None:
    """Writes ``stem``.svg and ``stem``.png, transparent, into ``OUT``."""
    for suffix, extra in (
        (".svg", {"metadata": {"Date": None}}),
        (".png", {"dpi": 300}),
    ):
        fig.savefig(
            OUT / f"{stem}{suffix}",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.02,
            **extra,
        )
    plt.close(fig)


# %%
if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["svg.hashsalt"] = "lazy-logo"  # stable SVG ids across runs
    for stem, colour in (
        ("lazy-logo", UOFT_BLUE),
        ("lazy-logo-white", "white"),
    ):
        fig, ax = plt.subplots(figsize=(6, 2.4))
        draw(ax, colour)
        save(fig, stem)
    fig, ax = plt.subplots(figsize=(2, 2))
    draw_icon(ax)
    # The favicon is the icon at 64 by 64 pixels (2 inches at 32 dpi).
    fig.savefig(OUT / "favicon.png", transparent=True, dpi=32)
    save(fig, "lazy-icon")
