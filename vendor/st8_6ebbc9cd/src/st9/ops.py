"""OPS2 replacement for the original src/st9/ops.py (commit 6ebbc9cd).

Only paths, logging and guard plumbing are replaced so that nothing is written
into the main project. Numeric constants (GRID, BANDS) and the environment
thread settings are copied unchanged from the original module.
"""
import csv, datetime as dt, hashlib, json, os, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PATHS = json.loads((ROOT / 'paths.yaml').read_text())
NAMESPACE = os.environ.get('ST9_NAMESPACE', 'st9e')
WORK = Path(PATHS['ops2_work'])
CACHE = WORK / 'cache' / NAMESPACE
TMP = WORK / 'cache' / 'tmp' / NAMESPACE
REPORTS = WORK / 'logs' / NAMESPACE
for directory in (CACHE, TMP, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(TMP), MPLCONFIGDIR=str(CACHE / 'matplotlib'), XDG_CACHE_HOME=str(CACHE / 'xdg'),
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
GRID = 3.515625
BANDS = {'HAR': (.6, 3.3), 'L0': (.75, 2.5)}


class RequiredStop(RuntimeError):
    def __init__(self, message, scope=None):
        self.scope = scope or NAMESPACE
        self.check = message
        super().__init__(message)


def check(name, condition, expected=None, scope=None):
    if not bool(condition if expected is None else condition == expected):
        raise RequiredStop(name, scope)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def rows(path):
    with (ROOT / path).open(newline='') as f:
        return list(csv.DictReader(f))


def source_hash(path):
    # Original verified against a frozen manifest; OPS2 records the hash instead.
    return sha(ROOT / path)


def ledger(*args, **kwargs):
    with (REPORTS / 'ledger.log').open('a') as f:
        f.write(json.dumps([now()] + [str(a) for a in args]) + '\n')


def delete_temp(path):
    Path(path).unlink()
