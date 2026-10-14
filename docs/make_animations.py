"""Render the README animations and the animated docs logo from real learner runs.

Writes docs/_static/readme/{learner1d,learner2d,learnerND}.gif and
docs/_static/logo_docs.webm. Needs ffmpeg.

Usage: python docs/make_animations.py [learner1d learner2d learnerND logo]
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import adaptive  # noqa: E402

STATIC = Path(__file__).parent / "_static"
DARK, ACCENT, GREY = "#2c3e6b", "#21918c", "#c8c8c8"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#888888",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "figure.facecolor": "white",
    }
)


def peak(x, a=0.01):
    return x + a**2 / (a**2 + x**2)


def ring(xy, a=0.2):
    x, y = xy
    return x + np.exp(-((x**2 + y**2 - 0.75**2) ** 2) / a**4)


def shell(xyz, a=0.3):
    x, y, z = xyz
    return np.exp(-((x**2 + y**2 + z**2 - 0.75**2) ** 2) / a**4)


def sampled(learner, **goal) -> tuple[np.ndarray, np.ndarray]:
    """Run ``learner`` and return its points and values in evaluation order."""
    adaptive.runner.simple(learner, **goal)
    points, values = zip(*learner.data.items())
    return np.array(points, dtype=float), np.array(values, dtype=float)


def counts(start: int, stop: int, nframes: int, power: float = 2) -> np.ndarray:
    """Number of points per frame, slow at first so the start is visible."""
    return (start + (stop - start) * np.linspace(0, 1, nframes) ** power).astype(int)


def write_frames(
    draw: Callable[[plt.Figure, int], None],
    nframes: int,
    figsize: tuple[float, float],
    folder: Path,
    dpi: int = 100,
    transparent: bool = False,
) -> None:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i in range(nframes):
        fig.clear()
        draw(fig, i)
        fig.savefig(folder / f"{i:04d}.png", transparent=transparent)
    plt.close(fig)


def to_gif(folder: Path, target: Path, fps: int = 15, hold: float = 2.5) -> None:
    graph = (
        f"tpad=stop_mode=clone:stop_duration={hold},split[a][b];"
        "[a]palettegen=max_colors=128:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    _ffmpeg(
        "-framerate", fps, "-i", folder / "%04d.png", "-vf", graph, "-loop", 0, target
    )


def _ffmpeg(*args) -> None:
    cmd = ["ffmpeg", "-loglevel", "error", "-y", *map(str, args)]
    subprocess.run(cmd, check=True)


def learner1d(target: Path) -> None:
    xs, ys = sampled(adaptive.Learner1D(peak, bounds=(-1, 1)), npoints_goal=120)
    x_true = np.linspace(-1, 1, 4000)

    def draw(fig, i):
        n = frames[i]
        axes = fig.subplots(1, 2, sharey=True)
        homogeneous = np.linspace(-1, 1, n)
        adaptive_xs = np.sort(xs[:n])
        for ax, x, title in [
            (axes[0], homogeneous, "Homogeneous grid"),
            (axes[1], adaptive_xs, "Adaptive"),
        ]:
            ax.plot(x_true, peak(x_true), color=GREY, lw=1, zorder=1)
            ax.plot(x, peak(x), color=DARK, lw=1.6, zorder=2)
            ax.scatter(x, peak(x), s=14, color=ACCENT, zorder=3, linewidths=0)
            ax.set_title(title)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.1, 2.1)
        fig.suptitle(f"{n} points", fontsize=12, color="#555555", y=0.06)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.2, wspace=0.08)

    frames = counts(3, len(xs), 90, power=1.5)
    _render_gif(draw, len(frames), (8, 3.6), target)


def _triangles(ax, points, values, vmin, vmax, edge_color):
    tri = mtri.Triangulation(points[:, 0], points[:, 1])
    ax.tripcolor(tri, values, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.triplot(tri, color=edge_color, lw=0.35, alpha=0.7)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def learner2d(target: Path) -> None:
    points, values = sampled(
        adaptive.Learner2D(ring, bounds=((-1, 1), (-1, 1))), npoints_goal=1600
    )
    vmin, vmax = values.min(), values.max()

    def draw(fig, i):
        n = frames[i]
        side = max(2, int(round(n**0.5)))
        grid = np.linspace(-1, 1, side)
        homogeneous = np.array([(x, y) for x in grid for y in grid])
        axes = fig.subplots(1, 2)
        _triangles(axes[0], homogeneous, ring(homogeneous.T), vmin, vmax, "white")
        _triangles(axes[1], points[:n], values[:n], vmin, vmax, "white")
        axes[0].set_title("Homogeneous grid")
        axes[1].set_title("Adaptive")
        fig.suptitle(
            f"{side * side} vs {n} points", fontsize=12, color="#555555", y=0.05
        )
        fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1, wspace=0.05)

    frames = counts(9, len(points), 60)
    _render_gif(draw, len(frames), (7, 3.75), target, dpi=90)


def learnerND(target: Path) -> None:
    points, values = sampled(
        adaptive.LearnerND(shell, bounds=[(-1, 1)] * 3), npoints_goal=2500
    )
    # Hide the octant facing the camera to show the inside of the shell.
    visible = ~((points[:, 0] > 0) & (points[:, 1] < 0) & (points[:, 2] > 0))
    colors = plt.get_cmap("viridis")(values)
    colors[:, 3] = 0.15 + 0.85 * values
    sizes = 2 + 10 * values
    nframes = 90

    def draw(fig, i):
        n = frames[i]
        ax = fig.add_subplot(projection="3d")
        mask = visible.copy()
        mask[n:] = False
        ax.scatter(
            *points[mask].T,
            c=colors[mask],
            s=sizes[mask],
            depthshade=False,
            linewidths=0,
        )
        ax.view_init(elev=20, azim=-60 + 360 * i / nframes)
        for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            set_lim(-1, 1)
        ax.set_box_aspect((1, 1, 1), zoom=1.25)
        ax.set_axis_off()
        fig.text(
            0.5,
            0.04,
            f"LearnerND, {n} points",
            ha="center",
            color="#555555",
            fontsize=12,
        )
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    frames = counts(20, len(points), nframes // 2)
    frames = np.concatenate([frames, np.full(nframes - len(frames), len(points))])
    _render_gif(draw, nframes, (4.5, 4.5), target, hold=0, dpi=80)


def _render_gif(draw, nframes, figsize, target, hold=2.5, dpi=100):
    with tempfile.TemporaryDirectory() as folder:
        write_frames(draw, nframes, figsize, Path(folder), dpi=dpi)
        to_gif(Path(folder), target, hold=hold)
    print(f"{target.relative_to(STATIC.parent)}: {target.stat().st_size / 1e6:.2f} MB")


def _rounded_alpha(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, size - 1, size - 1), radius, fill=255)
    return mask


def logo(target: Path) -> None:
    points, values = sampled(
        adaptive.Learner2D(ring, bounds=((-1, 1), (-1, 1))), loss_goal=0.005
    )
    vmin, vmax = values.min(), values.max()

    def draw(fig, i):
        ax = fig.add_axes((0, 0, 1, 1))
        _triangles(ax, points[: frames[i]], values[: frames[i]], vmin, vmax, "black")

    frames = counts(9, len(points), 24 * 10)
    size, dpi = 500, 100
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        write_frames(draw, len(frames), (size / dpi, size / dpi), folder, dpi=dpi)
        alpha = _rounded_alpha(size, radius=int(size * 0.3))
        for frame in sorted(folder.glob("*.png")):
            image = Image.open(frame).convert("RGBA")
            image.putalpha(alpha)
            image.save(frame)
        _ffmpeg(
            "-framerate", 24, "-i", folder / "%04d.png", "-vf", "tpad=stop_mode=clone:stop_duration=3",
            "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-crf", 30, "-b:v", 0, target,
        )  # fmt: skip
    print(f"{target.relative_to(STATIC.parent)}: {target.stat().st_size / 1e6:.2f} MB")


ANIMATIONS = {
    "learner1d": (learner1d, STATIC / "readme" / "learner1d.gif"),
    "learner2d": (learner2d, STATIC / "readme" / "learner2d.gif"),
    "learnerND": (learnerND, STATIC / "readme" / "learnerND.gif"),
    "logo": (logo, STATIC / "logo_docs.webm"),
}


def main(names: list[str]) -> None:
    (STATIC / "readme").mkdir(exist_ok=True)
    for name in names or ANIMATIONS:
        function, target = ANIMATIONS[name]
        function(target)


if __name__ == "__main__":
    main(sys.argv[1:])
