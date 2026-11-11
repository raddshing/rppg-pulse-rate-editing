"""Separate ST6 ownership, accounting and fresh disk measurements."""
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
PATHS = json.loads((ROOT / 'paths.yaml').read_text())
NAMES = ['st30' + x for x in 'prugteaq']
BASELINE_COMMIT = '19fdd8be3'


class ResourceWait(RuntimeError):
    pass


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with (ROOT / path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'item'):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.writing.' + str(os.getpid()))
    with tmp.open('w') as f:
        f.write(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def rows(path):
    with (ROOT / path).open(newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, values, fields=None):
    values = list(values)
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in values for k in r))
    tmp = p.with_name(p.name + '.writing.' + str(os.getpid()))
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(values)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def append(path, value):
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=list(value))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(value)
        f.flush()
        os.fsync(f.fileno())


def identity(pid=None):
    pid = pid or os.getpid()
    p = Path('/proc') / str(pid)
    fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, process_start_ticks=fields[19], executable=str((p / 'exe').resolve()))


def prefix_guard():
    receipt = json.loads((ROOT / 'reports/st30p/s6_0_complete.json').read_text())
    assert receipt['status'] == 'complete'
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', receipt['commit'], head], cwd=ROOT, check=True)
    for name in ['reports/ST6_PI_INSTR_01.md', 'reports/ST6_RL_DISP_01.md',
                 'reports/st30p/run_plan.csv', 'reports/st30p/environment_lock.json',
                 'reports/st30p/disk_authority.json', 'reports/st30p/seed_collision_check.json']:
        expected=receipt['fixed_hashes'][name]
        if name=='reports/ST6_PI_INSTR_01.md' and sha(name)!=expected:
            latest=ROOT/'reports/st30p/amendment_v1_5.json'
            amendment=json.loads((latest if latest.exists() else ROOT/'reports/st30p/amendment_v1_4.json').read_text())
            assert amendment['status']=='applied'
            assert sha(amendment['original_instruction_path'])==expected
            assert amendment['original_instruction_sha256']==expected
            assert sha(name)==amendment['instruction_sha256']
            assert amendment['instruction_sha256'] in (ROOT/'reports/ST6_INSTR01_PREFIX.md').read_text()
            for p,h in amendment['authority_hashes'].items():assert sha(p)==h,p
            if latest.exists():
                previous=json.loads((ROOT/'reports/st30p/amendment_v1_4.json').read_text())
                assert sha(amendment['previous_instruction_path'])==previous['instruction_sha256']
        else:
            assert sha(name)==expected,name
    return head


def configure(namespace):
    assert namespace in NAMES
    cache = Path(PATHS['cache']) / namespace
    tmp = Path(PATHS['cache']) / 'tmp' / namespace
    for p in [cache, tmp, ROOT / 'reports' / namespace]:
        p.mkdir(parents=True, exist_ok=True)
    os.environ.update(TMPDIR=str(tmp), MPLCONFIGDIR=str(cache / 'matplotlib'),
        XDG_CACHE_HOME=str(cache / 'xdg'), CUDA_CACHE_PATH=str(cache / 'cuda'),
        TORCH_EXTENSIONS_DIR=str(cache / 'torch_extensions'), OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')


def size(root):
    total = allocated = files = 0
    for here, dirs, names in os.walk(root, followlinks=False):
        for name in names:
            p = Path(here) / name
            try:
                s = p.lstat()
                if p.is_symlink() or not p.is_file():
                    continue
            except FileNotFoundError:
                continue
            files += 1
            total += s.st_size
            allocated += s.st_blocks * 512
    return dict(files=files, bytes=total, allocated_bytes=allocated)


def disk_sample():
    cache = Path(PATHS['cache'])
    bulk = Path(PATHS['bulk'])
    parts = []
    # Same working namespaces as the old rule, plus later and ST6 namespaces.
    # Retained M inputs remain counted; installed runtimes and legacy caches do not.
    for base in [cache, cache / 'tmp']:
        for p in sorted(base.iterdir()):
            if re.fullmatch(r'st(?:9|1[0-9]|2[0-8]|30)[a-z]*', p.name):
                parts.append(dict(path=str(p), **size(p)))
    new_bulk = [dict(path=str(bulk / ns), **size(bulk / ns)) for ns in NAMES]
    result = dict(sample_utc=now(), sample_epoch=time.time(),
        working_cache_bytes=sum(p['bytes'] for p in parts),
        working_cache_allocated_bytes=sum(p['allocated_bytes'] for p in parts),
        ST6_bulk_bytes=sum(p['bytes'] for p in new_bulk),
        ST6_bulk_allocated_bytes=sum(p['allocated_bytes'] for p in new_bulk),
        cache_free_bytes=shutil.disk_usage(cache).free, bulk_free_bytes=shutil.disk_usage(bulk).free,
        root_free_bytes=shutil.disk_usage(ROOT).free)
    return result, parts, new_bulk


def capacity(add_bulk=0, add_cache=0):
    r = json.loads((ROOT / 'reports/st30p/disk_status.json').read_text())
    if r['status'] != 'sampling' or time.time() - r['last']['sample_epoch'] > 90:
        raise ResourceWait('Fresh ST6 disk measurement unavailable')
    s = r['last']
    current=ROOT/'reports/st30p/disk_authority_v1_3.json'
    limits = json.loads((current if current.exists() else ROOT/'reports/st30p/disk_authority.json').read_text())
    if s['working_cache_bytes'] + add_cache > limits['cache_limit_bytes']:
        raise ResourceWait('Working cache including retained M inputs exceeds 150 GiB')
    if s['ST6_bulk_bytes'] + add_bulk > limits['new_bulk_cap_bytes']:
        raise ResourceWait('New ST6 retained bulk projection exceeds fixed PI cap')
    if shutil.disk_usage(PATHS['bulk']).free - add_bulk < limits['bulk_free_write_floor_bytes']:
        raise ResourceWait('Bulk free would fall below 50 GB')
    if current.exists() and shutil.disk_usage(PATHS['cache']).free-add_cache < limits['cache_write_floor_bytes']:
        raise ResourceWait('Working-cache filesystem free would fall below 20 GB')
    return s


@contextlib.contextmanager
def heavy_cpu_slot():
    locks = ROOT / 'reports/st30p/locks'
    locks.mkdir(exist_ok=True)
    handle = None
    while handle is None:
        for slot in range(3):
            f = (locks / ('CPU' + str(slot))).open('a')
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle = f
                break
            except BlockingIOError:
                f.close()
        if handle is None:
            time.sleep(1)
    try:
        yield
    finally:
        handle.close()


def phase(namespace, name, function, heavy=False, resource='ST6_CPU'):
    configure(namespace)
    root = ROOT / 'reports' / namespace
    p = root / (name + '_phase.json')
    assert not p.exists(), 'Never overwrite any attempted phase: ' + str(p)
    epoch = time.time()
    timer = time.monotonic()
    record = dict(status='running', resource=resource, started_epoch=epoch, started_utc=now(), **identity())
    save(p, record)
    code = 0
    try:
        head = prefix_guard()
        if heavy:
            with heavy_cpu_slot():
                capacity()
                result = dict(function() or {})
        else:
            result = dict(function() or {})
        result.setdefault('status', 'complete')
        result['execution_commit'] = head
    except ResourceWait as exc:
        code = 3
        result = dict(status='resource_wait', reason=str(exc))
    except Exception as exc:
        code = 1
        result = dict(status='execution_inability', error_type=type(exc).__name__, reason=str(exc))
        (root / (name + '_error.txt')).write_text(traceback.format_exc())
    result.update(finished_utc=now(), wall_seconds=time.monotonic() - timer)
    save(root / (name + '_completion.json'), result)
    save(p, dict(record, status=result['status'], wall_seconds=result['wall_seconds'], finished_utc=result['finished_utc']))
    print(json.dumps(clean(dict(stage=name, **result)), ensure_ascii=False), flush=True)
    if code:
        raise SystemExit(code)
    return result


def disk_sampler():
    samples = 0
    while True:
        tick = time.monotonic()
        sample, cache, bulk = disk_sample()
        append('outputs/st30p_disk_samples.csv', sample)
        samples += 1
        save('reports/st30p/disk_status.json', dict(status='sampling', samples=samples, last=sample,
            cache_scope=cache, new_bulk_scope=bulk, **identity(),
            scope='60 second samples; no continuous peak claim; retained old working cache included'))
        while time.monotonic() < tick + 60:
            if (ROOT / 'reports/st30p/disk_close_requested.json').exists():
                save('reports/st30p/disk_status.json', dict(status='closed', samples=samples, last=sample, closed_utc=now()))
                return dict(status='complete', samples=samples)
            time.sleep(min(1, max(0, tick + 60 - time.monotonic())))


if __name__ == '__main__':
    assert sys.argv[1:] == ['disk-sampler']
    phase('st30p', 'disk_sampler', disk_sampler, resource='ST6_CONTROL')
