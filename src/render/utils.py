from subprocess import Popen, PIPE

import moderngl
import numpy as np
from PIL import Image
from plyfile import PlyData
from matplotlib import pyplot as plt
from numpy.lib.recfunctions import structured_to_unstructured
from pyrr.matrix44 import create_look_at, create_perspective_projection, create_from_translation, create_from_scale, create_from_eulers
from pyrr.vector import normalize

from render.trajectories import CircularTrajectory

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


class Camera:
    def __init__(self, xyz, aspect_ratio):
        # n_bits = get_n_bits(xyz)
        # res = (2 ** n_bits) - 1
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        xyz_mid = (xyz_max - xyz_min) / 2.0
        self.translate = -xyz_mid
        self.scale = [1 / np.abs(xyz + self.translate).max()] * 3
        self.rotate = [-np.pi / 2, 0, 0]
        self.eye = np.array([3, 0, 0], dtype=np.float32)
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.up = np.array([0, 0, 1], dtype=np.float32)
        self.fovy = 45.0
        self.aspect_ratio = aspect_ratio
        self.z_near = 0.1
        self.z_far = 10.0

    def build_mvp(self):
        # Normalize voxelized point cloud to [-1, 1]^3 range
        translate = create_from_translation(self.translate, dtype=np.float32)
        scale = create_from_scale(self.scale, dtype=np.float32)
        rotate = create_from_eulers(self.rotate, dtype=np.float32)
        # View and projection matrices
        view = create_look_at(self.eye, self.center, self.up, dtype=np.float32)
        proj = create_perspective_projection(self.fovy, self.aspect_ratio, self.z_near, self.z_far, dtype=np.float32)
        return translate @ scale @ rotate @ view @ proj

    def set_eye(self, new_eye):
        self.eye = np.asarray(new_eye, dtype=np.float32)
        front = normalize(self.center - self.eye)
        right = normalize(np.cross(front, self.up))
        self.up = normalize(np.cross(right, front))


def read_pc(pc_path):
    pc = PlyData.read(pc_path)
    xyz = structured_to_unstructured(pc['vertex'].data[['x', 'y', 'z']], dtype=np.float32)
    rgb = structured_to_unstructured(pc['vertex'].data[['red', 'green', 'blue']], dtype=np.uint8)
    return xyz, rgb


def add_noise(xyz, intensity):
    rng = np.random.default_rng(42)
    xyz = xyz + rng.random(xyz.shape, dtype=np.float32) * intensity
    return xyz


class Renderer:
    def __init__(self, ctx, xyz, rgb, resolution=(1920, 1080), fps=24, duration=5, trajectory=CircularTrajectory(3),
                 cube_size=1):
        self.fps = fps
        self.duration = duration
        self.frames = fps * duration
        self.resolution = resolution
        self.trajectory = trajectory
        self.camera = Camera(xyz, resolution[0] / resolution[1])
        cube = get_cube(cube_size)
        cube_vbo = ctx.buffer(cube.tobytes())
        self.prog = pc_prog(ctx)
        self.instances = xyz.shape[0]
        xyz_vbo = ctx.buffer(xyz.tobytes())
        rgb_vbo = ctx.buffer(rgb.tobytes())
        self.vao = ctx.vertex_array(self.prog, [
            (cube_vbo, '3f4', 'in_cube'),
            (xyz_vbo, '3f4 /i', 'in_vert'),
            (rgb_vbo, '3f1 /i', 'in_color'),
        ])

        self.fbo = ctx.simple_framebuffer(resolution)

    def __call__(self, i):
        percent = i / self.frames
        self.trajectory(percent, self.camera)
        mvp = self.camera.build_mvp()

        self.fbo.use()
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.prog['model'].write(mvp)
        self.vao.render(moderngl.TRIANGLES, instances=self.instances)
        return Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)

    def __len__(self):
        return self.frames
