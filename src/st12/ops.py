"""Instruction05B paths, immutable-input checks and scoped phase records."""
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATHS = json.loads((ROOT / 'paths.yaml').read_text())
NAMESPACE = os.environ.get('ST12_NAMESPACE', 'st12')
NAMES = ('st12', 'st12e', 'st12c', 'st12d', 'st12v', 'st12x')
assert NAMESPACE in NAMES
CACHE = Path(PATHS['cache']) / NAMESPACE
TMP = Path(PATHS['cache']) / 'tmp' / NAMESPACE
REPORTS = ROOT / 'reports' / NAMESPACE
for directory in (CACHE, TMP, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(TMP), MPLCONFIGDIR=str(CACHE / 'matplotlib'),
                  XDG_CACHE_HOME=str(CACHE / 'xdg'), CUDA_CACHE_PATH=str(CACHE / 'cuda'),
                  TORCH_EXTENSIONS_DIR=str(CACHE / 'torch_extensions'), OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
GRID = 3.515625
BANDS = {'HAR': (.6, 3.3), 'L0': (.75, 2.5)}
READ_HASHES = {}
_MANIFEST = None


class RequiredStop(RuntimeError):
    def __init__(self, message, scope=None):
        self.scope = scope or NAMESPACE
        super().__init__(message)


class ResourceWait(RuntimeError):
    pass


def check(name, condition, expected=None, scope=None):
    if not bool(condition if expected is None else condition == expected):
        raise RequiredStop(name, scope)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with (ROOT / path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, records, fields=None):
    records = list(records)
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(records[0]))
        w.writeheader()
        w.writerows(records)


def append_row(path, record):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=list(record))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(record)
        f.flush()


def save_json(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temporary = p.with_name(p.name + '.writing.' + str(os.getpid()))
    temporary.write_text(raw)
    temporary.replace(p)


def path_key(path):
    p = ROOT / path
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def source_hash(path):
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = {r['path']: r for r in rows('outputs/st12_inputs_expected.csv')}
    key = path_key(path)
    check('Input has frozen checksum: ' + key, key in _MANIFEST)
    p = ROOT / path
    digest = sha(p)
    r = _MANIFEST[key]
    check('Frozen input checksum and size: ' + key,
          digest == r['sha256'] and p.stat().st_size == int(r['bytes']))
    READ_HASHES[key] = digest
    return digest


def prefix_guard(require_s0=True):
    if require_s0:
        s0=json.loads((ROOT / 'reports/st12/st12_s0_completion.json').read_text())
        check('S12-0 complete',s0['status']=='complete')
        scope=os.environ.get('ST12_SCOPE') or {'st12e':'E5R','st12d':'D5b','st12v':'V5R','st12x':'X5b','st12c':'C5'}.get(NAMESPACE)
        check('Current scope has no frozen input mismatch','shared' not in s0.get('blocked_scopes',[]) and scope not in s0.get('blocked_scopes',[]),scope=scope)
    paths = ['reports/ST4_INSTR05B_PREFIX.md', 'reports/ST4_PI_INSTR_05B.md',
             'reports/ST4_RL_DISP_07.md', 'reports/ST3_RL_DISP_EXIT_v1_2_5.md']
    for path in paths:
        tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', path], cwd=ROOT,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        clean = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', path], cwd=ROOT)
        check('Preregistration committed: ' + path, tracked.returncode == clean.returncode == 0)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def cpu_phase_seconds():
    total = 0.0
    for namespace in NAMES:
        for path in (ROOT / 'reports' / namespace).glob('*_phase.json'):
            r = json.loads(path.read_text())
            if r.get('resource') == 'CPU':
                total += float(r.get('wall_seconds', max(0, time.time() - r['started_epoch'])))
    return total


def budget_check(projected_additional_seconds=0):
    used = cpu_phase_seconds()
    if used + projected_additional_seconds > 216000:
        raise ResourceWait('CPU phase sum would exceed 60 hours')
    if (ROOT / 'reports/st12/st12_disk_limit.json').exists():
        raise ResourceWait('Disk sampler recorded a capacity condition')
    return used


def process_identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), process_start_ticks=fields[19], executable=str((p / 'exe').resolve()),
                command=' '.join(x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x))


def ledger(job, event, pid=None):
    append_row(f'outputs/{NAMESPACE}_process_ledger.csv',
               dict(job=job, **process_identity(pid or os.getpid()), recorded_utc=now(), event=event))


def progress(stage, done, total, started, status='running'):
    elapsed = time.monotonic() - started
    save_json(REPORTS / f'{NAMESPACE}_{stage}_progress.json',
              dict(stage=stage, status=status, done=int(done), total=int(total),
                   elapsed_seconds=elapsed, remaining_seconds=elapsed / done * (total - done) if done else None,
                   updated_utc=now()))


def delete_temp(path):
    p = Path(path)
    check('Delete only new consumed scratch', any(d.resolve() in p.resolve().parents for d in (CACHE, TMP)))
    append_row(f'outputs/{NAMESPACE}_deleted_temp.csv', dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
    p.unlink()


def finish(stage, function, resource='CPU', require_s0=True):
    started = time.monotonic()
    epoch = time.time()
    ledger(stage, 'started')
    phase = REPORTS / f'{NAMESPACE}_{stage}_phase.json'
    save_json(phase, dict(status='running', resource=resource, started_epoch=epoch, started_utc=now()))
    code = 0
    try:
        budget_check()
        commit = prefix_guard(require_s0)
        result = dict(function() or {})
        result.setdefault('status', 'complete')
        result['execution_commit'] = commit
    except RequiredStop as exc:
        code = 2
        result = dict(status='stopped_required_check', check=str(exc), scope=exc.scope)
        save_json(REPORTS / f'{NAMESPACE}_{stage}_required_stop.json', dict(at_utc=now(), **result))
    except ResourceWait as exc:
        code = 3
        result = dict(status='resource_wait', reason=str(exc))
        save_json(REPORTS / f'{NAMESPACE}_{stage}_resource_wait.json', dict(at_utc=now(), **result))
    except Exception as exc:
        import traceback
        code = 1
        result = dict(status='implementation_error', error_type=type(exc).__name__,
                      stack=[dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                             for f in traceback.extract_tb(exc.__traceback__)])
    wall = time.monotonic() - started
    result.update(process_wall_seconds=wall, finished_utc=now())
    save_json(REPORTS / f'{NAMESPACE}_{stage}_completion.json', result)
    save_json(REPORTS / f'{NAMESPACE}_{stage}_input_hashes.json', READ_HASHES)
    save_json(REPORTS / f'{NAMESPACE}_{stage}_exit.json', dict(exit_code=code, finished_utc=now()))
    save_json(phase, dict(status=result['status'], resource=resource, started_epoch=epoch,
                         wall_seconds=wall, finished_utc=now()))
    ledger(stage, 'completed' if code == 0 else 'stopped')
    print(json.dumps(dict(stage=stage, status=result['status'])), flush=True)
    if code:
        raise SystemExit(code)
