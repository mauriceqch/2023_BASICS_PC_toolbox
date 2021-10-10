import moderngl
from PIL import Image

from render.camera import CenteredCamera
from render.trajectories import CircularTrajectory
from render.utils import get_cube, pc_prog, add_noise


class Renderer:
    def __init__(self, ctx, xyz, rgb, resolution=(1920, 1080), fps=24, duration=5, trajectory=CircularTrajectory(3),
                 cube_size=1, noise=0.01):
        self.fps = fps
        self.duration = duration
        self.frames = fps * duration
        self.resolution = resolution
        self.trajectory = trajectory
        self.camera = CenteredCamera(xyz, resolution[0] / resolution[1])
        cube = get_cube(cube_size) * (1 + noise) - cube_size * noise / 2
        cube_vbo = ctx.buffer(cube.tobytes())
        self.prog = pc_prog(ctx)
        self.instances = xyz.shape[0]

        xyz = add_noise(xyz, noise)  # Fix z-fighting
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


def get_cube_with_noise(cube_size, noise):
    return get_cube(cube_size) * (1 + noise) - cube_size * noise / 2


class InteractiveRenderer:
    def __init__(self, ctx, xyz, rgb, cube_size=1, noise=0.01):
        self.noise = noise
        cube = get_cube_with_noise(cube_size, noise)
        self.cube_vbo = ctx.buffer(cube.tobytes())
        self.prog = pc_prog(ctx)
        self.instances = xyz.shape[0]

        xyz = add_noise(xyz, noise)  # Fix z-fighting
        xyz_vbo = ctx.buffer(xyz.tobytes())
        self.rgb_vbo = ctx.buffer(rgb.tobytes())
        self.vao = ctx.vertex_array(self.prog, [
            (self.cube_vbo, '3f4', 'in_cube'),
            (xyz_vbo, '3f4 /i', 'in_vert'),
            (self.rgb_vbo, '3f1 /i', 'in_color'),
        ])

    def set_cube_size(self, size):
        cube = get_cube_with_noise(size, self.noise)
        self.cube_vbo.write(cube)

    def set_rgb(self, rgb):
        self.rgb_vbo.write(rgb.tobytes())

    def __call__(self, camera):
        mvp = camera.build_mvp()
        self.prog['model'].write(mvp)
        self.vao.render(moderngl.TRIANGLES, instances=self.instances)