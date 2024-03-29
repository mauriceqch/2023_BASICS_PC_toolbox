import shlex
import sys
import subprocess
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Popen(subprocess.Popen):
    def __init__(self, *args, stdout=None, stderr=None, **kwargs):
        super(Popen, self).__init__(*args, stdout=stdout, stderr=stderr, **kwargs)
        logger.info(args_to_str(self.args))
        self.stdout = stdout
        self.stderr = stderr


def subprocess_run(*args, check=True, **kwargs):
    with Popen(*args, **kwargs) as p:
        p.wait()
        if check:
            if p.returncode != 0:
                logs = ''
                if is_file(p.stdout):
                    with open(p.stdout.name, 'r') as f:
                        logs = f.read()
                raise RuntimeError(f'{" ".join([shlex.quote(x) for x in p.args])} returned with code {p.returncode}\n{logs}')


def is_file(f):
    return f is not None and f is not sys.stdout and f is not sys.stderr


def args_to_str(args):
    return " ".join([shlex.quote(x) for x in args])


def safe_close(f):
    if is_file(f):
        f.close()


def parallel_process(f, params, parallelism):
    procs = []
    try:
        with tqdm(total=len(params)) as pbar:
            i = 0
            while len(params) > 0 or len(procs) > 0:
                if len(procs) < parallelism and len(params) > 0:
                    param = params.pop()
                    procs.append(f(*param))
                elif len(procs) > 0:
                    try:
                        p = procs[i]
                        p.wait(0.1)
                        if p.returncode != 0:
                            logs = ''
                            if is_file(p.stdout):
                                with open(p.stdout.name, 'r') as f:
                                    logs = f.read()
                            raise RuntimeError(f'{args_to_str(p.args)} returned with code {p.returncode}\n{logs}')
                        safe_close(p.stdout)
                        safe_close(p.stderr)
                        pbar.update(1)
                        pbar.refresh()
                        procs.pop(i)
                    except subprocess.TimeoutExpired:
                        pass
                    i = (i + 1) % max(len(procs), 1)
    finally:
        for p in procs:
            p.terminate()
            safe_close(p.stdout)
            safe_close(p.stderr)
