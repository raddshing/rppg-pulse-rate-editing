"""Stage4 paths, frozen input checks, resource records and blind-safe progress."""
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PATHS = json.loads((ROOT / 'paths.yaml').read_text())
CACHE = Path(PATHS['cache']) / 'st4'
TMP = Path(PATHS['cache']) / 'tmp/st4'
REPORTS = ROOT / 'reports/st4'
for directory in (CACHE, TMP, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(TMP), MPLCONFIGDIR=str(CACHE / 'matplotlib'),
                  XDG_CACHE_HOME=str(CACHE / 'xdg'), CUDA_CACHE_PATH=str(CACHE / 'cuda'),
                  TORCH_EXTENSIONS_DIR=str(CACHE / 'torch_extensions'),
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
GRID = 3.515625
BANDS = {'HAR': (0.6, 3.3), 'L0': (0.75, 2.5)}
MODELS = ('deepphys', 'efficientphys', 'factorizephys', 'physformer', 'physnet', 'rhythmformer', 'tscan')
RW15 = sorted([f'st1_mmpd_m03_rw_{m}_{a}_all_s100' for m in MODELS for a in ('pm', 'um')]
              + ['st1_mmpd_m03_rw_ibvpnet_pm_all_s100'])
RG6 = sorted(f'st1_mmpd_m03_rg1_{a}_s{s}' for a in ('ubfc', 'maubfc') for s in range(3))
READ_HASHES = {}
_MANIFEST = None


class RequiredStop(RuntimeError):
    def __init__(self, message, scope='bundle'):
        self.scope = scope
        super().__init__(message)


def check(name, condition):
    if not bool(condition):
        raise RequiredStop(name)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
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
    with (ROOT / path).open('a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=list(record))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(record)
        f.flush()


def save_json(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def source_hash(path):
    global _MANIFEST
    p = ROOT / path
    key = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    if _MANIFEST is None:
        _MANIFEST = {}
        for name in ('reports/st1_exit_manifest.csv', 'reports/st2_exit_manifest.csv',
                     'reports/st3b/st3b_supplemental_input_manifest.csv',
                     'outputs/st4_st3exit_manifest.csv', 'outputs/st4_input_hash_checks.csv'):
            for r in rows(name):
                _MANIFEST[r['path']] = r.get('expected_sha256', r['sha256'])
    check('Input has an existing frozen checksum', key in _MANIFEST)
    digest = sha(p)
    check('Frozen input checksum unchanged', digest == _MANIFEST[key])
    READ_HASHES[key] = digest
    return digest


def prefix_guard():
    check('Stage3 exit freeze complete', (REPORTS / 'st4_s0_completion.json').exists())
    p = 'reports/ST4_INSTR01v2_PREFIX.md'
    result = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', p], cwd=ROOT)
    tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', p], cwd=ROOT, capture_output=True)
    check('Prerequisite prefix committed before dependent calculation', result.returncode == 0 and tracked.returncode == 0)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def budget_check():
    approval = json.loads((REPORTS / 'instruction_approval.json').read_text())
    elapsed = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(approval['approved_at_utc'])).total_seconds()
    # A conservative setup reserve covers document reads before the account receipt.
    elapsed += 900
    idle = REPORTS / 'st4_review_idle_intervals.json'
    if idle.exists():
        for r in json.loads(idle.read_text())['intervals']:
            if r.get('end_utc'):
                elapsed -= (dt.datetime.fromisoformat(r['end_utc']) - dt.datetime.fromisoformat(r['start_utc'])).total_seconds()
    check('Whole CPU bundle below 36 hours', elapsed < 129600)
    stop = REPORTS / 'st4_required_stop.json'
    check('No unresolved required bundle stop', not stop.exists() or
          json.loads(stop.read_text()).get('status') == 'resolved_implementation_error')
    check('Working cache below 40 GiB', not (REPORTS / 'st4_disk_limit.json').exists())
    return elapsed


def progress(stage, done, total, started, status='running'):
    elapsed = time.monotonic() - started
    remaining = elapsed / done * (total - done) if done else None
    value = dict(stage=stage, status=status, done=int(done), total=int(total),
                 elapsed_seconds=round(elapsed, 3), remaining_seconds=round(remaining, 3) if remaining is not None else None,
                 updated_utc=now())
    save_json(REPORTS / ('st4_' + stage + '_progress.json'), value)
    print(json.dumps(value), flush=True)


def process_identity(pid):
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=int(pid), process_start_ticks=fields[19], executable=str((p / 'exe').resolve()),
                command=' '.join(x.decode() for x in (p / 'cmdline').read_bytes().split(b'\0') if x))


def ledger(job, event, pid=None):
    record = process_identity(pid or os.getpid())
    append_row('outputs/st4_process_ledger.csv', dict(job=job, pid=record['pid'],
               process_start_ticks=record['process_start_ticks'], started_utc=now(),
               executable=record['executable'], command=record['command'], event=event))


def delete_temp(path):
    p = Path(path)
    check('Only Stage4 temporary files may be deleted', any(d.resolve() in p.resolve().parents for d in (CACHE, TMP)))
    append_row('outputs/st4_deleted_temp.csv', dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
    p.unlink()


def finish(stage, function):
    started = time.monotonic()
    ledger(stage, 'started')
    code = 0
    try:
        budget_check()
        commit = prefix_guard()
        result = function() or {}
        result = dict(status='complete', prefix_commit=commit, **result)
    except RequiredStop as e:
        code = 2
        result = dict(status='stopped_required_check', check=str(e), scope=e.scope)
        destination = 'st4_required_stop.json' if e.scope == 'bundle' else 'st4_' + stage + '_required_stop.json'
        save_json(REPORTS / destination, dict(stage=stage, at_utc=now(), **result))
    except Exception as e:
        import traceback
        code = 1
        result = dict(status='implementation_error', error_type=type(e).__name__,
                      stack=[dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                             for f in traceback.extract_tb(e.__traceback__)])
    result.update(process_wall_seconds=time.monotonic() - started, finished_utc=now())
    save_json(REPORTS / ('st4_' + stage + '_completion.json'), result)
    save_json(REPORTS / ('st4_' + stage + '_input_hashes.json'), READ_HASHES)
    save_json(REPORTS / ('st4_' + stage + '_exit.json'), dict(exit_code=code, finished_utc=now()))
    progress_file = REPORTS / ('st4_' + stage + '_progress.json')
    if progress_file.exists():
        state = json.loads(progress_file.read_text())
        state.update(status=result['status'], remaining_seconds=0, updated_utc=now())
        save_json(progress_file, state)
    ledger(stage, 'completed' if code == 0 else 'stopped')
    print(json.dumps(dict(stage=stage, status=result['status'], **({k: result[k] for k in ('check', 'error_type') if k in result}))), flush=True)
    if code:
        raise SystemExit(code)
