import argparse
import sys

import moderngl

from render.trajectories import FixedPointTrajectory
from render.utils import read_pc
from render.renderers import Renderer


def render_pc(pc_path, output_path, show, eye, cube_size):
    ctx = moderngl.create_context(standalone=True)
    ctx.enable_only(ctx.DEPTH_TEST)
    xyz, rgb = read_pc(pc_path)
    renderer = Renderer(ctx, xyz, rgb, fps=1, duration=1, trajectory=FixedPointTrajectory(eye), cube_size=cube_size)
    img = renderer(0)
    img.save(output_path)
    if show:
        img.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--show', default=False, required=False, action='store_true')
    parser.add_argument('--eye', type=float, nargs=3, default=[2, 2, 2], required=False)
    parser.add_argument('--cube_size', type=float, default=1.0, required=False)

    args = parser.parse_args()
    render_pc(args.input_path, args.output_path, args.show, args.eye, args.cube_size)
