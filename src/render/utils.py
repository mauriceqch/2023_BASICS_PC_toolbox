from subprocess import Popen, PIPE

import numpy as np
from plyfile import PlyData
from matplotlib import pyplot as plt
from numpy.lib.recfunctions import structured_to_unstructured

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


def img_to_vid(images, fps, vid_output_path):
    frames = len(images)
    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(fps), '-vcodec', 'bmp', '-i', '-',
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '17', '-r', str(fps), vid_output_path], stdin=PIPE,
              stdout=PIPE)
    for i in range(frames):
        im = images[i]
        im.save(p.stdin, 'BMP')
    p.stdin.close()
    p.wait()


def read_pc(pc_path):
    pc = PlyData.read(pc_path)
    xyz = structured_to_unstructured(pc['vertex'].data[['x', 'y', 'z']], dtype=np.float32)
    avail_attrs = [x.name for x in pc['vertex'].properties]

    attrs = {}
    rgb_attr = ['red', 'green', 'blue']
    if all(x in avail_attrs for x in rgb_attr):
        attrs['rgb'] = structured_to_unstructured(pc['vertex'].data[rgb_attr], dtype=np.uint8)

    return xyz, attrs


def add_noise(xyz, intensity):
    rng = np.random.default_rng(42)
    xyz = xyz + (rng.random(xyz.shape, dtype=np.float32) - 0.5) * intensity
    return xyz
