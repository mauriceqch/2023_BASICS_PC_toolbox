import sys

import moderngl

from render.trajectories import FixedPointTrajectory
from render.utils import read_pc, Renderer


def render_pc(pc_path, output_path, show):
    ctx = moderngl.create_standalone_context()
    ctx.enable_only(ctx.DEPTH_TEST)
    xyz, rgb = read_pc(pc_path)
    eye = [2, 2, 2]
    # eye = [3, 0, 0]
    renderer = Renderer(ctx, xyz, rgb, fps=1, duration=1, trajectory=FixedPointTrajectory(eye))
    img = renderer(0)
    img.save(output_path)
    if show:
        img.show()


if __name__ == '__main__':
    render_pc(sys.argv[1], sys.argv[2], '--show' in sys.argv)
