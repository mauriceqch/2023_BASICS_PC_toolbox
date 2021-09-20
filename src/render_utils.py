import ctypes
import math

import numpy as np
from matplotlib import pyplot as plt


# Euler–Rodrigues formula
def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])


def euler_rotmat(angles):
    roll, pitch, yaw = angles
    sR = np.sin(roll)
    cR = np.cos(roll)
    sP = np.sin(pitch)
    cP = np.cos(pitch)
    sY = np.sin(yaw)
    cY = np.cos(yaw)

    return np.array([[cY * cP, -cY * sP * cR + sY * sR, cY * sP * sR + sY * cR, 0],
                     [sP, cP * cR, -cP * sR, 0],
                     [-sY * cP, sY * sP * cR + cY * sR, -sY * sP * sR + cY * cR, 0],
                     [0, 0, 0, 1]], dtype=np.float32)


def normalize(x):
    return x / np.linalg.norm(x)


class Camera:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.eye = np.array([0.0, 0.0, 5 * scale_factor], dtype=np.float32)
        self.up = normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        self.default_center = self.center.copy()
        self.default_eye = self.eye.copy()
        self.default_up = self.up.copy()

    def zoom(self, y):
        scale = 1 - y / 10
        self.eye = self.center + (self.eye - self.center) * scale

    def rotate(self, x, y):
        factor = 100
        R = rotation_matrix(self.up, -x / factor)
        self.eye = R @ self.eye
        self.up = R @ self.up
        R2 = rotation_matrix(np.cross(normalize(self.eye - self.center), self.up), y / factor)
        self.eye = R2 @ self.eye

    def translate(self, x, y):
        x_factor = x / 100
        y_factor = y / 100
        delta = self.scale_factor * (y_factor * self.up + x_factor * normalize(np.cross(self.eye - self.center, self.up)))
        self.center = self.center + delta
        self.eye = self.eye + delta

    def reset(self):
        self.center = self.default_center.copy()
        self.eye = self.default_eye.copy()
        self.up = self.default_up.copy()

DEFAULT_CMAP = 'viridis'

# def z_color(points, zmin, zmax, cmap=DEFAULT_CMAP):
#     cmap = Colormap(zmin, zmax, cmap)
#     colors = cmap(points[:, 2])
#     col_vbo = vbo.VBO(data=colors, usage=GL.GL_DYNAMIC_DRAW, target=GL.GL_ARRAY_BUFFER)
#     return col_vbo


class Colormap:
    def __init__(self, vmin, vmax, cmap=DEFAULT_CMAP):
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = plt.get_cmap(cmap)

    def __call__(self, v):
        rnge = self.vmax - self.vmin
        if rnge == 0.0:
            rnge = 1.0
        return self.cmap((v - self.vmin) / rnge)[:, :3].astype(np.float32)


def print_help():
    print("""
    LMB: Rotate
    RMB: Translate
    MW: Zoom in/out
    Ctrl + Scroll: Change point size
    r: reset camera
    c: switch color scheme (elevation, D1 distance, D1 log distance)
    """)

