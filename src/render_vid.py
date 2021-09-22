import moderngl

from tqdm import trange

from src.render.utils import img_to_vid, read_pc, add_noise, Renderer


def render_pc(pc_path, ctx):
    xyz, rgb = read_pc(pc_path)
    xyz = add_noise(xyz, 0.1)  # Fix z-fighting
    print(xyz.shape, rgb.shape)

    renderer = Renderer(ctx, xyz, rgb)
    images = [renderer(i) for i in trange(len(renderer))]

    vid_output_path = f'{pc_path[:-4]}.avi'
    img_to_vid(images, renderer.fps, vid_output_path)


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
