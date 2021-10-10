import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

DEFAULT_CMAP = 'viridis'


def z_color(points, zmin, zmax, cmap=DEFAULT_CMAP):
    cmap = Colormap(zmin, zmax, cmap)
    colors = cmap(points[:, 2])
    return colors


def d1_color(points1, points2, cmap=DEFAULT_CMAP, log=False):
    t1 = cKDTree(points1, balanced_tree=False)
    t2 = cKDTree(points2, balanced_tree=False)
    dist1, _ = t2.query(points1, workers=-1)
    dist2, _ = t1.query(points2, workers=-1)
    if log:
        dist1 = np.log(dist1)
        dist2 = np.log(dist2)
    d = np.concatenate([dist1, dist2])
    dmin = np.quantile(d, 0.01)
    dmax = np.quantile(d, 0.99)
    cmap = Colormap(dmin, dmax, cmap)
    colors1 = cmap(dist1)
    colors2 = cmap(dist2)
    return colors1, colors2


class Colormap:
    def __init__(self, vmin, vmax, cmap=DEFAULT_CMAP):
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = plt.get_cmap(cmap)

    def __call__(self, v):
        rnge = self.vmax - self.vmin
        if rnge == 0.0:
            rnge = 1.0
        return (self.cmap((v - self.vmin) / rnge)[:, :3] * 255).astype(np.uint8)