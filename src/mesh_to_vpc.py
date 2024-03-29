import os
import argparse
import numpy as np
from datetime import datetime
from pyntcloud import PyntCloud

from mesh_to_pc import mesh_to_pc
from pc_to_vpc import pc_to_vpc
from utils.parallel_process import Popen


# Example
# /snap/bin/cloudcompare.CloudCompare -SILENT -AUTO_SAVE OFF -O ./Lubna_Breakfast/LubnaSit.obj -SAMPLE_MESH POINTS 2000000 -C_EXPORT_FMT PLY -PLY_EXPORT_FMT BINARY_LE -SAVE_CLOUDS FILE ./LubnaSit.ply
def cc_convert(cc_path, input_path, output_path, n_samples, format='BINARY_LE'):
    cmd = [cc_path, '-SILENT', '-AUTO_SAVE', 'OFF', '-O', input_path,
           '-SAMPLE_MESH', 'POINTS', str(n_samples),
           '-C_EXPORT_FMT', 'PLY', '-PLY_EXPORT_FMT', format, '-SAVE_CLOUDS', 'FILE', output_path]
    devnull = open(os.devnull, 'wb')
    return Popen(cmd, stdout=devnull, stderr=devnull)


def mesh_to_vpc(input_path, output_path, n_samples, vg_size):
    mesh_to_pc(input_path, output_path, n_samples)
    pc_to_vpc(output_path, output_path, vg_size)


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

    mesh_to_vpc(args.input_path, args.output_path, args.n_samples, 2 ** args.vg_size)
 