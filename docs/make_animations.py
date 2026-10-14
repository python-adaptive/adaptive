"""Render the README animations and the animated docs logo from real learner runs.

Writes docs/_static/readme/{learner1d,learner2d,learnerND}.gif and
docs/_static/logo_docs.webm. Needs ffmpeg, and Chrome or Chromium for kaleido
(set BROWSER_PATH if it is not found).

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
from adaptive.learner.learner1D import uniform_loss  # noqa: E402
from adaptive.learner.learner2D import uniform_loss as uniform_loss_2d  # noqa: E402

STATIC = Path(__file__).parent / "_static"
DARK, ACCENT, NEW = "#2c3e6b", "#21918c", "#e4572e"
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


def peak(x, a=0.01, x0=0.137):
    # Off-center, so a homogeneous grid does not happen to hit the peak exactly.
    return x + a**2 / (a**2 + (x - x0) ** 2)


def ring(xy, a=0.2):
    x, y = xy
    return x + np.exp(-((x**2 + y**2 - 0.75**2) ** 2) / a**4)


def shell(xyz, a=0.4):
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
    xs, _ = sampled(adaptive.Learner1D(peak, bounds=(-1, 1)), npoints_goal=100)
    # Homogeneous sampling that keeps its earlier points, so nothing jumps.
    uniform = adaptive.Learner1D(peak, bounds=(-1, 1), loss_per_interval=uniform_loss)
    xs_uniform, _ = sampled(uniform, npoints_goal=100)

    def draw(fig, i):
        n = frames[i]
        axes = fig.subplots(1, 2, sharey=True)
        for ax, x, title in [
            (axes[0], xs_uniform[:n], "Homogeneous"),
            (axes[1], xs[:n], "Adaptive"),
        ]:
            order = np.sort(x)
            ax.plot(order, peak(order), color=DARK, lw=1.6, zorder=2)
            ax.scatter(x, peak(x), s=16, color=ACCENT, zorder=3, linewidths=0)
            ax.set_title(title)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.15, 1.35)
        if n:  # the points chosen last
            for ax, x in [(axes[0], xs_uniform[n - 1]), (axes[1], xs[n - 1])]:
                ax.scatter(x, peak(x), s=50, color=NEW, zorder=4)
        fig.suptitle(f"{n} points", fontsize=12, color="#555555", y=0.06)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.2, wspace=0.08)

    # One point per frame at first, then speed up.
    frames = np.concatenate(
        [np.arange(0, 40), np.linspace(40, len(xs), 30).astype(int)]
    )
    _render_gif(draw, len(frames), (8, 3.6), target, fps=12)


def _triangles(ax, points, values, vmin, vmax, edge_color):
    tri = mtri.Triangulation(points[:, 0], points[:, 1])
    ax.tripcolor(tri, values, shading="gouraud", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.triplot(tri, color=edge_color, lw=0.35, alpha=0.7)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.axis("off")


def learner2d(target: Path) -> None:
    bounds = ((-1, 1), (-1, 1))
    points, values = sampled(adaptive.Learner2D(ring, bounds), npoints_goal=1600)
    uniform = adaptive.Learner2D(ring, bounds, loss_per_triangle=uniform_loss_2d)
    points_uniform, values_uniform = sampled(uniform, npoints_goal=1600)
    vmin, vmax = values.min(), values.max()

    def draw(fig, i):
        n = frames[i]
        axes = fig.subplots(1, 2)
        _triangles(axes[0], points_uniform[:n], values_uniform[:n], vmin, vmax, "white")
        _triangles(axes[1], points[:n], values[:n], vmin, vmax, "white")
        axes[0].set_title("Homogeneous")
        axes[1].set_title("Adaptive")
        fig.suptitle(f"{n} points", fontsize=12, color="#555555", y=0.05)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1, wspace=0.05)

    frames = counts(4, len(points), 60, power=2.5)
    _render_gif(draw, len(frames), (7, 3.75), target, dpi=90)


def learnerND(target: Path) -> None:
    """Rotating version of the LearnerND plot on the "Algorithms and examples" page."""
    import plotly.graph_objects as go
    import plotly.io as pio

    points, values = sampled(
        adaptive.LearnerND(shell, bounds=[(-1, 1)] * 3), npoints_goal=5000
    )
    # Cut away the x > 0, y > 0 quarter to show the shell in cross-section.
    keep = ~((points[:, 0] > 0) & (points[:, 1] > 0))
    marker = {
        "size": 5,
        "color": values[keep],
        "colorscale": "Viridis",
        "line": {"color": "rgb(50,50,50)", "width": 0.5},
    }  # same style as LearnerND.plot_3D, with bigger markers
    x, y, z = points[keep].T
    fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker))
    hidden = {"visible": False}
    fig.update_layout(
        scene={
            "xaxis": hidden, "yaxis": hidden, "zaxis": hidden, "aspectmode": "cube",
            "camera": {"projection": {"type": "orthographic"}},
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        annotations=[
            {
                "text": f"LearnerND, {len(points)} points",
                "x": 0.5, "y": 0.02, "xref": "paper", "yref": "paper",
                "showarrow": False, "font": {"size": 15, "color": "#555555"},
            }
        ],
    )  # fmt: skip
    nframes = 90
    figures = []
    for i in range(nframes):
        # Sway back and forth around the view into the cut.
        angle = np.pi / 4 + np.radians(45) * np.sin(2 * np.pi * i / nframes)
        eye = {"x": 2 * np.cos(angle), "y": 2 * np.sin(angle), "z": 0.25}
        figures.append(go.Figure(fig).update_layout(scene_camera_eye=eye))
    with tempfile.TemporaryDirectory() as folder:
        files = [Path(folder) / f"{i:04d}.png" for i in range(nframes)]
        pio.write_images(figures, files, width=400, height=400)
        to_gif(Path(folder), target, hold=0)
    print(f"{target.relative_to(STATIC.parent)}: {target.stat().st_size / 1e6:.2f} MB")


def _render_gif(draw, nframes, figsize, target, hold=2.5, dpi=100, fps=15):
    with tempfile.TemporaryDirectory() as folder:
        write_frames(draw, nframes, figsize, Path(folder), dpi=dpi)
        to_gif(Path(folder), target, fps=fps, hold=hold)
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
