import moderngl
from plyfile import PlyData
from numpy.lib.recfunctions import structured_to_unstructured
from pyrr.matrix44 import create_look_at, create_perspective_projection
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
            in vec3 in_vert;
            in vec3 in_color;

            out vec3 v_color;

            void main() {
                v_color = in_color;
                gl_Position = model * vec4(in_vert, 1.0);
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
    view = create_look_at([3, 0, 0], [0, 0, 0], [0, 0, 1], dtype=np.float32)
    proj = create_perspective_projection(60, 1.0, 0.1, 10.0, dtype=np.float32)
    mvp = view @ proj

    res = 1024

    xyz = structured_to_unstructured(pc['vertex'].data[['x', 'y', 'z']], dtype=np.float32)
    rgb = structured_to_unstructured(pc['vertex'].data[['red', 'green', 'blue']], dtype=np.uint8)

    # xyz = xyz[:1000]
    # rgb = rgb[:1000]

    print('Build cubes')
    # xyz = np.hstack([xyz,
    #                  xyz + np.array([1, 0, 0], dtype=np.float32),
    #                  xyz + np.array([0, 1, 0], dtype=np.float32),
    #                  ])
    xyz = np.hstack([xyz + c_el for c_el in cube])
    xyz = xyz.reshape((-1, 3))
    xyz = xyz / (res / 2) - 1.0

    rgb = np.repeat(rgb, int(xyz.shape[0] / rgb.shape[0]), axis=0)

    print(xyz, rgb)

    print('Render')
    xyz_vbo = ctx.buffer(xyz.tobytes())
    rgb_vbo = ctx.buffer(rgb.tobytes())
    vao = ctx.vertex_array(prog, [(xyz_vbo, '3f4', 'in_vert'), (rgb_vbo, '3f1', 'in_color')])

    resolution = 4096
    fbo = ctx.simple_framebuffer((resolution, resolution))
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    prog['model'].write(mvp)
    vao.render(moderngl.TRIANGLES)

    print('Done')
    Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1).show()
