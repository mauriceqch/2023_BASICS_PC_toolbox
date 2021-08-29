import os
import argparse
import numpy as np
from datetime import datetime
from pyntcloud import PyntCloud

from utils.parallel_process import Popen


# Example
# /snap/bin/cloudcompare.CloudCompare -SILENT -AUTO_SAVE OFF -O ./Lubna_Breakfast/LubnaSit.obj -SAMPLE_MESH POINTS 2000000 -C_EXPORT_FMT PLY -PLY_EXPORT_FMT BINARY_LE -SAVE_CLOUDS FILE ./LubnaSit.ply
def cc_convert(cc_path, input_path, output_path, n_samples, format='BINARY_LE'):
    cmd = [cc_path, '-SILENT', '-AUTO_SAVE', 'OFF', '-O', input_path,
           '-SAMPLE_MESH', 'POINTS', str(n_samples),
           '-C_EXPORT_FMT', 'PLY', '-PLY_EXPORT_FMT', format, '-SAVE_CLOUDS', 'FILE', output_path]
    devnull = open(os.devnull, 'wb')
    return Popen(cmd, stdout=devnull, stderr=devnull)


def process(input_path, output_path, n_samples, vg_size):
    output_folder, _ = os.path.split(output_path)
    if output_folder != '':
        os.makedirs(output_folder, exist_ok=True)

    print(f'{datetime.now().isoformat()} CC mesh sampling')
    ps = cc_convert('/snap/bin/cloudcompare.CloudCompare', input_path, output_path, n_samples)
    ps.wait()

    print(f'{datetime.now().isoformat()} Voxelization')
    pc = PyntCloud.from_file(output_path)
    dtypes = pc.points.dtypes
    coords = ['x', 'y', 'z']
    points = pc.points[coords].values
    n_pts_inter = len(points)
    points = points / np.max(points)
    points = points * (vg_size - 1)
    points = np.round(points)
    pc.points[coords] = points

    print(f'{datetime.now().isoformat()} Point merging')
    if len(set(pc.points.columns) - set(coords)) > 0:
        pc.points = pc.points.groupby(by=coords, sort=False).mean().reset_index()
    else:
        pc.points = pc.points.drop_duplicates()

    pc.points = pc.points.astype(dtypes)

    print(f'{datetime.now().isoformat()} Writing point cloud')
    pc.to_file(output_path)
    print(f'{datetime.now().isoformat()} Done')
    print(f'Input path: {input_path}')
    print(f'Output path: {output_path}')
    print(f'n_samples: {n_samples}')
    print(f'n_points after sampling: {n_pts_inter}')
    print(f'final n_points: {len(pc.points)}')
    print(f'columns: {pc.points.columns}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ds_mesh_to_pc.py',
        description='Mesh to voxelized PC.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_path', help='Source')
    parser.add_argument('output_path', help='Destination')
    parser.add_argument('--vg_size', type=int,
                        help='Voxel Grid resolution for x, y, z dimensions in number of geometry bits. Example: 10 bits = 1024 values.',
                        required=True)
    parser.add_argument('--n_samples', type=int, help='Number of samples', required=True)

    args = parser.parse_args()

    assert os.path.exists(args.input_path), f'{args.input_path} does not exist'
    assert args.vg_size > 0, f'vg_size must be positive'
    assert args.n_samples > 0, f'n_samples must be positive'

    process(args.input_path, args.output_path, args.n_samples, 2 ** args.vg_size)
 