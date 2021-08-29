import os
import argparse
import numpy as np
from datetime import datetime
from pyntcloud import PyntCloud


def pc_to_vpc(input_path, output_path, vg_size):
    print(f'{datetime.now().isoformat()} pc_to_vpc')
    print(f'Input path: {input_path}')
    print(f'Output path: {output_path}')
    output_folder, _ = os.path.split(output_path)
    if output_folder != '':
        os.makedirs(output_folder, exist_ok=True)

    print(f'{datetime.now().isoformat()} Reading file')
    pc = PyntCloud.from_file(input_path)
    print(f'{datetime.now().isoformat()} Voxelization')
    dtypes = pc.points.dtypes
    coords = ['x', 'y', 'z']
    points = pc.points[coords].values
    n_pts_inter = len(points)
    points = points - np.min(points, axis=0)
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

    args = parser.parse_args()

    assert os.path.exists(args.input_path), f'{args.input_path} does not exist'
    assert args.vg_size > 0, f'vg_size must be positive'

    pc_to_vpc(args.input_path, args.output_path, 2 ** args.vg_size)
 