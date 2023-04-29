---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
execution:
  timeout: 300
---

```{code-cell} ipython3
:tags: [remove-input]

import functools
import subprocess
import tempfile
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

import adaptive


@functools.lru_cache
def make_cut(size, rad):
    # Make new images
    circle = Image.new("L", (rad * 2, rad * 2), color=1)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=0)
    alpha = Image.new("L", size, 0)

    # Crop circles
    w, h = size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))

    # To array
    return np.array(alpha)


@functools.lru_cache
def add_rounded_corners(size=(1000, 1000), rad=300):
    cut = make_cut(size, rad)
    cut = cut.reshape((*cut.shape, 1)).repeat(4, axis=2)

    # Set the corners to (252, 252, 252, 255) to match the RTD background #FCFCFC
    cut[:, :, -1] *= 255
    cut[:, :, :-1] *= 252
    return cut


def remove_rounded_corners(fname):
    im = Image.open(fname)
    ar = np.array(im)
    cut = make_cut(size=ar.shape[:-1], rad=round(ar.shape[0] * 0.3)).astype(bool)
    ar[:, :, -1] = np.where(~cut, ar[:, :, -1], 0)
    im_new = Image.fromarray(ar)
    im_new.save(fname)
    return im_new


def learner_till(till, learner, data):
    new_learner = adaptive.Learner2D(None, bounds=learner.bounds)
    new_learner.data = dict(data[:till])
    for x, y in learner._bounds_points:
        # always include the bounds
        new_learner.tell((x, y), learner.data[x, y])
    return new_learner


def plot_tri(learner, ax):
    tri = learner.ip().tri
    triang = mtri.Triangulation(*tri.points.T, triangles=tri.simplices)
    return ax.triplot(triang, c="k", lw=0.8, alpha=0.8)


def get_new_artists(npoints, learner, data, rounded_corners, ax):
    new_learner = learner_till(npoints, learner, data)
    line1, line2 = plot_tri(new_learner, ax)
    data = np.rot90(new_learner.interpolated_on_grid()[-1])
    im = ax.imshow(data, extent=(-0.5, 0.5, -0.5, 0.5), cmap="viridis")
    if rounded_corners is None:
        return im, line1, line2
    else:
        im2 = ax.imshow(rounded_corners, extent=(-0.5, 0.5, -0.5, 0.5), zorder=10)
        return im, line1, line2, im2


@functools.lru_cache
def create_and_run_learner():
    def ring(xy):
        import numpy as np

        x, y = xy
        a = 0.2
        return x + np.exp(-((x**2 + y**2 - 0.75**2) ** 2) / a**4)

    learner = adaptive.Learner2D(ring, bounds=[(-1, 1), (-1, 1)])
    adaptive.runner.simple(learner, loss_goal=0.005)
    return learner


def get_figure():
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    return fig, ax


def setup(nseconds=15):
    learner = create_and_run_learner()

    data = list(learner.data.items())

    fig, ax = get_figure()

    npoints = (len(data) * np.linspace(0, 1, 24 * nseconds) ** 2).astype(int)
    rounded_corners = add_rounded_corners(size=(1000, 1000), rad=300)
    return npoints, learner, data, rounded_corners, fig, ax


def animate_mp4(fname="source/_static/logo_docs.mp4", nseconds=15):
    npoints, learner, data, rounded_corners, fig, ax = setup()
    artists = [
        get_new_artists(n, learner, data, rounded_corners, ax) for n in tqdm(npoints)
    ]
    ani = animation.ArtistAnimation(fig, artists, blit=True)
    ani.save(fname, writer=FFMpegWriter(fps=24))


def animate_png(folder=None, nseconds=15):
    npoints, learner, data, rounded_corners, fig, ax = setup(nseconds)
    if folder is None:
        folder = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fnames = []
    ims = []
    for i, n in tqdm(enumerate(npoints), total=len(npoints)):
        fname = folder / f"logo_docs_{i:07d}.png"
        fnames.append(fname)
        npoints, learner, data, _, fig, ax = setup(nseconds)
        get_new_artists(n, learner, data, None, ax)
        fig.savefig(fname, transparent=True)
        ax.cla()
        plt.close(fig)
        im = remove_rounded_corners(fname)
        ims.append(im)
    return fnames, ims


def save_webp(fname_webp, ims):
    (im, *_ims) = ims
    im.save(
        fname_webp,
        save_all=True,
        append_images=_ims,
        opimize=False,
        durarion=2,
        quality=70,
    )


def save_webm(fname, fnames):
    args = [
        "ffmpeg",
        "-framerate",
        "24",
        "-f",
        "image2",
        "-i",
        str(fnames[0]).replace("0000000", "%07d"),
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        "yuva420p",
        "-crf",
        "23",  # 0 is lossless 51 is worst
        "-y",
        fname,
    ]
    return subprocess.run(args, capture_output=True)


if __name__ == "__main__":
    fname_mp4 = Path("_static/logo_docs.mp4")
    # if not fname_mp4.exists():
    #     animate_mp4(fname_mp4)
    fname_webm = fname_mp4.with_suffix(".webm")
    if not fname_webm.exists():
        fnames, ims = animate_png()
        save_webm(fname_webm, fnames)
```

```{eval-rst}
.. raw:: html

    <video autoplay loop muted playsinline webkit-playsinline
     style="width: 400px; max-width: 100%; margin: 0 auto; display:block;">
      <source src="_static/logo_docs.webm" type="video/mp4">
    </video><br>
```
