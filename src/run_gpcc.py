import argparse
import glob
import logging
import multiprocessing
import os
from utils.parallel_process import parallel_process, Popen

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def assert_exists(filepath):
    assert os.path.exists(filepath), f'{filepath} not found'


def run_gpcc(output_dir, tmc13_dir, pc_error, mpeg_cfg_path, input_pc, input_norm):
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    assert_exists(input_pc)
    assert_exists(input_norm)
    assert_exists(mpeg_cfg_path)
    tmc13 = os.path.join(tmc13_dir, 'build', 'tmc3', 'tmc3')
    tmc13_mf = os.path.join(tmc13_dir, 'scripts', 'Makefile.tmc13-step')
    assert_exists(tmc13)
    assert_exists(tmc13_mf)

    return Popen(['make',
                  '-f', tmc13_mf,
                  '-C', output_dir,
                  f'VPATH={mpeg_cfg_path}',
                  f'ENCODER={tmc13}',
                  f'DECODER={tmc13}',
                  f'PCERROR={pc_error}',
                  f'SRCSEQ={input_pc}',
                  f'NORMSEQ={input_norm}'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='run_gpcc.py', description='Run G-PCC experiments.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_folder', help='Input folder.')
    parser.add_argument('input_pattern', help='Input pattern.')
    parser.add_argument('output_folder', help='Output folder.')
    parser.add_argument('tmc13_dir', help='TMC13/G-PCC path.')
    parser.add_argument('tmc13_cfg', help='TMC13/G-PCC config path.')
    parser.add_argument('pc_error', help='pc_error path.')
    parser.add_argument('--num_parallel', help='Number of parallel jobs.', default=multiprocessing.cpu_count(), type=int)
    args = parser.parse_args()

    args.input_folder = os.path.abspath(args.input_folder)
    args.output_folder = os.path.abspath(args.output_folder)
    args.input_folder = os.path.normpath(args.input_folder)
    args.output_folder = os.path.normpath(args.output_folder)

    assert_exists(args.input_folder)
    assert_exists(args.tmc13_dir)
    assert_exists(args.tmc13_cfg)
    assert_exists(args.pc_error)

    cfgs = glob.glob(os.path.join(args.tmc13_cfg, "*"))
    print(cfgs)
    cfgs_names = [os.path.split(x)[1] for x in cfgs]
    input(f'{cfgs_names} G-PCC configurations have been found. Press ENTER to confirm.')

    files = glob.glob(os.path.join(args.input_folder, args.input_pattern), recursive=True)
    assert len(files) > 0
    logger.info(f'Found {len(files)} matching files')

    output_folders = [os.path.join(args.output_folder, os.path.splitext(x[len(args.input_folder)+1:])[0]) for x in files]

    logger.info('Starting GPCC experiments')
    params = []
    logger.info('Started GPCC experiments')

    for inpf, outf in zip(files, output_folders):
        for cfg, cfg_path in zip(cfgs_names, cfgs):
            cur_outf = os.path.join(outf, cfg)
            params.append((cur_outf, args.tmc13_dir, args.pc_error, cfg_path, inpf, inpf))

    # An SSD is highly recommended, extremely slow when running in parallel on an HDD due to parallel writes
    # If HDD, set parallelism to 1
    parallel_process(run_gpcc, params, args.num_parallel)

    logger.info('Finished GPCC experiments')
