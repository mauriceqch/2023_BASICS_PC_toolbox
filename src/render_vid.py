import moderngl
import sys

from tqdm import trange

from render.utils import img_to_vid, read_pc, Renderer
from render.trajectories import HelixTrajectory


def render_pc(pc_path, output_path):
    ctx = moderngl.create_standalone_context()
    ctx.enable_only(ctx.DEPTH_TEST)
    xyz, rgb = read_pc(pc_path)
    renderer = Renderer(ctx, xyz, rgb, trajectory=HelixTrajectory(3))
    images = [renderer(i) for i in trange(len(renderer))]
    img_to_vid(images, renderer.fps, output_path)


if __name__ == '__main__':
    render_pc(sys.argv[1], sys.argv[2])
