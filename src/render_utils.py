import ctypes
import math

import numpy as np
from matplotlib import pyplot as plt


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


def pc_prog(ctx):
    return ctx.program(
        vertex_shader='''
            #version 330

            uniform mat4 model;
            in vec3 in_cube;
            in vec3 in_vert;
            in vec3 in_color;

            out vec3 v_color;

            void main() {
                v_color = in_color;
                gl_Position = model * vec4(in_vert + in_cube, 1.0);
            }
        ''',
        fragment_shader='''
            #version 330

            in vec3 v_color;

            out vec3 f_color;

            void main() {
                f_color = v_color;
            }
        ''',
    )


def get_cube(cube_size):
    cube = np.array([
        -1.0, -1.0, -1.0,
        -1.0, -1.0, 1.0,
        -1.0, 1.0, 1.0,
        1.0, 1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0, 1.0, -1.0,
        1.0, -1.0, 1.0,
        -1.0, -1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0, 1.0, -1.0,
        1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0,
        -1.0, 1.0, 1.0,
        -1.0, 1.0, -1.0,
        1.0, -1.0, 1.0,
        -1.0, -1.0, 1.0,
        -1.0, -1.0, -1.0,
        -1.0, 1.0, 1.0,
        -1.0, -1.0, 1.0,
        1.0, -1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, -1.0, -1.0,
        1.0, 1.0, -1.0,
        1.0, -1.0, -1.0,
        1.0, 1.0, 1.0,
        1.0, -1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, -1.0,
        -1.0, 1.0, -1.0,
        1.0, 1.0, 1.0,
        -1.0, 1.0, -1.0,
        -1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        -1.0, 1.0, 1.0,
        1.0, -1.0, 1.0
    ], dtype=np.float32).reshape((-1, 3))
    cube = (cube + 1) / 2
    cube = cube * cube_size
    return cube


def get_n_bits(xyz):
    xyz2 = xyz - np.min(xyz, axis=0)
    max_val = max(np.max(xyz2), np.max(xyz))
    return int(np.ceil(np.log2(max_val)))


class CircularTrajectory:
    def __init__(self, dist):
        self.dist = dist

    def __call__(self, percent):
        xVal = self.dist * np.sin(2 * np.pi * percent)
        yVal = self.dist * np.cos(2 * np.pi * percent)
        return [xVal, yVal, 0]
