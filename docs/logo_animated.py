import matplotlib.tri as mtri
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm.auto import tqdm

import adaptive


def learner_till(till, learner, data):
    new_learner = adaptive.Learner2D(None, bounds=learner.bounds)
    new_learner.data = {k: v for k, v in data[:till]}
    for x, y in learner._bounds_points:
        # always include the bounds
        new_learner.tell((x, y), learner.data[x, y])
    return new_learner


def plot_tri(learner, ax):
    tri = learner.ip().tri
    triang = mtri.Triangulation(*tri.points.T, triangles=tri.vertices)
    return ax.triplot(triang, c="k", lw=0.8, alpha=0.8)


def get_new_artists(npoints, learner, data):
    new_learner = learner_till(npoints, learner, data)
    line1, line2 = plot_tri(new_learner, ax)
    data = np.rot90(new_learner.interpolated_on_grid()[-1])
    im = ax.imshow(data, extent=(-0.5, 0.5, -0.5, 0.5), cmap="viridis")
    return im, line1, line2


def create_and_run_learner():
    def ring(xy):
        import numpy as np

        x, y = xy
        a = 0.2
        return x + np.exp(-((x ** 2 + y ** 2 - 0.75 ** 2) ** 2) / a ** 4)

    learner = adaptive.Learner2D(ring, bounds=[(-1, 1), (-1, 1)])
    adaptive.runner.simple(learner, goal=lambda l: l.loss() < 0.005)
    return learner


if __name__ == "__main__":
    learner = create_and_run_learner()

    data = list(learner.data.items())

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_xticks([])
    ax.set_yticks([])

    nseconds = 15
    npoints = (len(data) * np.linspace(0, 1, 24 * nseconds) ** 2).astype(int)

    artists = [get_new_artists(n, learner, data) for n in tqdm(npoints)]

    ani = animation.ArtistAnimation(fig, artists, blit=True)
    ani.save("logo.mp4", writer=FFMpegWriter(fps=24))
