import moderngl
from plyfile import PlyData
from numpy.lib.recfunctions import structured_to_unstructured
from pyrr.matrix44 import create_look_at, create_perspective_projection, create_from_translation, create_from_scale
from pyrr.vector import normalize
import numpy as np
from subprocess import Popen, PIPE

from PIL import Image

from src.render_utils import pc_prog, get_cube, get_n_bits, CircularTrajectory


class Camera:
    def __init__(self, xyz, eye):
        n_bits = get_n_bits(xyz)
        res = (2 ** n_bits) - 1
        self.scale = [2 / res] * 3
        self.translate = [-1.0] * 3
        self.eye = eye
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.up = np.array([0, 0, 1], dtype=np.float32)
        self.fovy = 45.0
        self.aspect_ratio = 1.0
        self.z_near = 0.1
        self.z_far = 10.0

    def build_mvp(self):
        # Normalize voxelized point cloud to [-1, 1]^3 range
        scale = create_from_scale(self.scale, dtype=np.float32)
        translate = create_from_translation(self.translate, dtype=np.float32)
        # View and projection matrices
        view = create_look_at(self.eye, self.center, self.up, dtype=np.float32)
        proj = create_perspective_projection(self.fovy, self.aspect_ratio, self.z_near, self.z_far, dtype=np.float32)
        return scale @ translate @ view @ proj

    def set_eye(self, new_eye):
        self.eye = np.asarray(new_eye, dtype=np.float32)
        front = normalize(self.center - self.eye)
        right = normalize(np.cross(front, self.up))
        self.up = normalize(np.cross(right, front))


def main():
    ctx = moderngl.create_standalone_context()
    ctx.enable_only(ctx.DEPTH_TEST)

    prog = pc_prog(ctx)

    print('Read file')
    pc = PlyData.read('047.ply')
    # pc = PlyData.read('091.ply')
    xyz = structured_to_unstructured(pc['vertex'].data[['x', 'y', 'z']], dtype=np.float32)
    rgb = structured_to_unstructured(pc['vertex'].data[['red', 'green', 'blue']], dtype=np.uint8)
    print(xyz.shape, rgb.shape)

    # Fix z-fighting
    rng = np.random.default_rng(42)
    xyz = xyz + rng.random(xyz.shape, dtype=np.float32) * 0.1

    print('Render')
    xyz_vbo = ctx.buffer(xyz.tobytes())
    rgb_vbo = ctx.buffer(rgb.tobytes())
    cube = get_cube(8)
    cube_vbo = ctx.buffer(cube.tobytes())
    vao = ctx.vertex_array(prog, [
        (cube_vbo, '3f4', 'in_cube'),
        (xyz_vbo, '3f4 /i', 'in_vert'),
        (rgb_vbo, '3f1 /i', 'in_color'),
    ])

    dist = 3
    eye = np.array([dist, 0, 0], dtype=np.float32)
    camera = Camera(xyz, eye)

    resolution = 1280
    fbo = ctx.simple_framebuffer((resolution, resolution))
    fbo.use()

    images = []

    frames = 256
    trajectory = CircularTrajectory(dist)
    for i in range(frames):
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        percent = i / frames
        eye = trajectory(percent)
        camera.set_eye(eye)
        mvp = camera.build_mvp()

        prog['model'].write(mvp)
        vao.render(moderngl.TRIANGLES, instances=xyz.shape[0])

        images.append(Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1))

    fps = 24
    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(fps), '-vcodec', 'bmp', '-i', '-',
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '17', '-r', str(fps), 'video.avi'], stdin=PIPE, stdout=PIPE)
    for i in range(frames):
        im = images[i]
        im.save(p.stdin, 'BMP')
    p.stdin.close()
    p.wait()

    print('Done')


if __name__ == '__main__':
    main()
