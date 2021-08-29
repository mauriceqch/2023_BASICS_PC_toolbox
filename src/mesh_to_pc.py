import os
import argparse
from datetime import datetime

from utils.parallel_process import Popen


# Example
# /snap/bin/cloudcompare.CloudCompare -SILENT -AUTO_SAVE OFF -O ./Lubna_Breakfast/LubnaSit.obj -SAMPLE_MESH POINTS 2000000 -C_EXPORT_FMT PLY -PLY_EXPORT_FMT BINARY_LE -SAVE_CLOUDS FILE ./LubnaSit.ply
def cc_convert(cc_path, input_path, output_path, n_samples, format='BINARY_LE'):
    cmd = [cc_path, '-SILENT', '-AUTO_SAVE', 'OFF', '-O', input_path,
           '-SAMPLE_MESH', 'POINTS', str(n_samples),
           '-C_EXPORT_FMT', 'PLY', '-PLY_EXPORT_FMT', format, '-SAVE_CLOUDS', 'FILE', output_path]
    devnull = open(os.devnull, 'wb')
    return Popen(cmd, stdout=devnull, stderr=devnull)


def mesh_to_pc(input_path, output_path, n_samples):
    print(f'{datetime.now().isoformat()} mesh_to_pc')
    print(f'Input path: {input_path}')
    print(f'Output path: {output_path}')
    print(f'n_samples: {n_samples}')
    output_folder, _ = os.path.split(output_path)
    if output_folder != '':
        os.makedirs(output_folder, exist_ok=True)

    print(f'{datetime.now().isoformat()} CC mesh sampling')
    ps = cc_convert('/snap/bin/cloudcompare.CloudCompare', input_path, output_path, n_samples)
    ps.wait()
    print(f'{datetime.now().isoformat()} Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ds_mesh_to_pc.py',
        description='Mesh to PC.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input_path', help='Source')
    parser.add_argument('output_path', help='Destination')
    parser.add_argument('--n_samples', type=int, help='Number of samples', required=True)

    args = parser.parse_args()

    assert os.path.exists(args.input_path), f'{args.input_path} does not exist'
    assert args.n_samples > 0, f'n_samples must be positive'

    mesh_to_pc(args.input_path, args.output_path, args.n_samples)
 