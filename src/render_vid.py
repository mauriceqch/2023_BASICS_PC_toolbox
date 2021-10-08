import argparse
from subprocess import PIPE, Popen

import moderngl
import sys

from tqdm import trange

from render.utils import img_to_vid, read_pc
from render.renderers import Renderer
from render.trajectories import HelixTrajectory


def render_pc(pc_path, output_path, cube_size):
    ctx = moderngl.create_context(standalone=True)
    ctx.enable_only(ctx.DEPTH_TEST)
    xyz, rgb = read_pc(pc_path)
    renderer = Renderer(ctx, xyz, rgb, trajectory=HelixTrajectory(3), cube_size=cube_size)

    frames = len(renderer)
    fps = renderer.fps
    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(fps), '-vcodec', 'bmp', '-i', '-',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '17', '-r', str(fps), output_path], stdin=PIPE,
              stdout=PIPE)
    for i in range(frames):
        im = renderer(i)
        im.save(p.stdin, 'BMP')
    p.stdin.close()
    p.wait()
    # img_to_vid(images, renderer.fps, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--cube_size', type=float, default=1.0, required=False)

    args = parser.parse_args()
    render_pc(args.input_path, args.output_path, args.cube_size)
