import moderngl
from plyfile import PlyData
from numpy.lib.recfunctions import structured_to_unstructured
from pyrr.matrix44 import create_look_at, create_perspective_projection, create_from_translation, create_from_scale
import numpy as np

from PIL import Image

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

cube_size = 8
cube = cube * cube_size

if __name__ == '__main__':
    ctx = moderngl.create_standalone_context()
    ctx.enable_only(ctx.DEPTH_TEST)

    prog = ctx.program(
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

    print('Read file')
    pc = PlyData.read('047.ply')
    # pc = PlyData.read('091.ply')

    res = 1024
    scale = create_from_scale([1 / (res / 2)] * 3, dtype=np.float32)
    translate = create_from_translation([-1.0] * 3, dtype=np.float32)
    view = create_look_at([3, 0, 0], [0, 0, 0], [0, 0, 1], dtype=np.float32)
    proj = create_perspective_projection(60, 1.0, 0.01, 100.0, dtype=np.float32)
    mvp = scale @ translate @ view @ proj

    xyz = structured_to_unstructured(pc['vertex'].data[['x', 'y', 'z']], dtype=np.float32)
    rgb = structured_to_unstructured(pc['vertex'].data[['red', 'green', 'blue']], dtype=np.uint8)

    # xyz = xyz[:1000]
    # rgb = rgb[:1000]
    print(xyz.shape, rgb.shape, cube.shape)

    print('Render')
    xyz_vbo = ctx.buffer(xyz.tobytes())
    rgb_vbo = ctx.buffer(rgb.tobytes())
    cube_vbo = ctx.buffer(cube.tobytes())
    vao = ctx.vertex_array(prog, [
        (cube_vbo, '3f4', 'in_cube'),
        (xyz_vbo, '3f4 /i', 'in_vert'),
        (rgb_vbo, '3f1 /i', 'in_color'),
    ])

    resolution = 2048
    fbo = ctx.simple_framebuffer((resolution, resolution))
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    prog['model'].write(mvp)
    vao.render(moderngl.TRIANGLES, instances=xyz.shape[0])

    print('Done')
    Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()
