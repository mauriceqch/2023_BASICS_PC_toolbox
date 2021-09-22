import moderngl

from tqdm import trange

from src.render.trajectories import FixedPointTrajectory
from src.render.utils import read_pc, add_noise, Renderer


def render_pc(pc_path, ctx):
    xyz, rgb = read_pc(pc_path)
    xyz = add_noise(xyz, 0.1)  # Fix z-fighting
    print(xyz.shape, rgb.shape)

    renderer = Renderer(ctx, xyz, rgb, fps=1, duration=1, trajectory=FixedPointTrajectory([3, 0, 0]))
    renderer(0).show()


def main():
    ctx = moderngl.create_standalone_context()
    ctx.enable_only(ctx.DEPTH_TEST)

    # render_pc(f'047.ply', ctx)
    # render_pc(f'001.ply', ctx)
    # render_pc(pc, ctx)
    for i in range(1, 6):
        render_pc(f'{i:03d}.ply', ctx)


if __name__ == '__main__':
    main()
