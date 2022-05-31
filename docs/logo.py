import os
import sys

import holoviews
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(".."))  # to get adaptive on the path

import adaptive  # noqa: E402, isort:skip

holoviews.notebook_extension("matplotlib")


def create_and_run_learner():
    def ring(xy):
        import numpy as np

        x, y = xy
        a = 0.2
        return x + np.exp(-((x**2 + y**2 - 0.75**2) ** 2) / a**4)

    learner = adaptive.Learner2D(ring, bounds=[(-1, 1), (-1, 1)])
    adaptive.runner.simple(learner, goal=lambda l: l.loss() < 0.01)
    return learner


def plot_learner_and_save(learner, fname):
    fig, ax = plt.subplots()
    tri = learner.interpolator(scaled=True).tri
    triang = mtri.Triangulation(*tri.points.T, triangles=tri.vertices)
    ax.triplot(triang, c="k", lw=0.8)
    ax.imshow(learner.plot().Image.I.data, extent=(-0.5, 0.5, -0.5, 0.5))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(fname, bbox_inches="tight", transparent=True, dpi=300, pad_inches=-0.1)


def add_rounded_corners(fname, rad):
    im = Image.open(fname)
    circle = Image.new("L", (rad * 2, rad * 2), 0)
    draw = ImageDraw.Draw(circle)
    draw.ellipse((0, 0, rad * 2, rad * 2), fill=255)
    alpha = Image.new("L", im.size, 255)
    w, h = im.size
    alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
    alpha.paste(circle.crop((0, rad, rad, rad * 2)), (0, h - rad))
    alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))
    alpha.paste(circle.crop((rad, rad, rad * 2, rad * 2)), (w - rad, h - rad))
    im.putalpha(alpha)
    return im


def main(fname="source/_static/logo_docs.png"):
    learner = create_and_run_learner()
    plot_learner_and_save(learner, fname)
    im = add_rounded_corners(fname, rad=200)
    im.thumbnail((200, 200), Image.ANTIALIAS)  # resize
    im.save(fname)


if __name__ == "__main__":
    main()
