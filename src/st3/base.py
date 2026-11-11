"""Small input/hash and recording helpers for the fixed Stage 3 protocol."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PATHS = json.loads((ROOT / 'paths.yaml').read_text())
CACHE = Path(PATHS['cache']) / 'st3'
TMP = CACHE / 'tmp'
TMP.mkdir(parents=True, exist_ok=True)
os.environ.update(TMPDIR=str(TMP), MPLCONFIGDIR=str(CACHE / 'matplotlib'),
                  XDG_CACHE_HOME=str(CACHE / 'xdg'), CUDA_CACHE_PATH=str(CACHE/'cuda'),
                  TORCH_EXTENSIONS_DIR=str(CACHE/'torch_extensions'), OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.path.insert(0, str(ROOT))
from src.st3.progress import update

SUBJECTS = ['subject' + str(i) for i in (2, 4, 7, 8, 12, 13, 14, 18, 20, 21, 29)]
MODELS = ('deepphys', 'efficientphys', 'factorizephys', 'physformer', 'physnet', 'rhythmformer', 'tscan')
RW15 = sorted(['st1_mmpd_m03_rw_%s_%s_all_s100' % (m, p)
               for m in MODELS for p in ('pm', 'um')] + ['st1_mmpd_m03_rw_ibvpnet_pm_all_s100'])
RG_A = ['st1_mmpd_m03_rg1_%s_s%d' % (a, s) for a in ('ubfc', 'maubfc') for s in range(3)]
RG_PURE = ['st1_rg_tscan_%s_s%d' % (a, s) for a in ('ubfc', 'maubfc') for s in range(3)]
UNSUP_A = ['st1_mmpd_m03_rw_%s_unsup_mmpd_all_s100' % m for m in ('chrom', 'pos')]
BANDS = {'HAR': (.6, 3.3), 'L0': (.75, 2.5)}
GRID = 3.515625
READ_HASHES = {}
MANIFESTS = None


class RequiredStop(RuntimeError):
    def __init__(self, check, scope='bundle'):
        self.check, self.scope = check, scope
        super().__init__(check)


def check(name, value, expected=True, tolerance=0, scope='bundle'):
    ok = abs(value - expected) <= tolerance if isinstance(value, (int, float)) and isinstance(expected, (int, float)) else value == expected
    if not ok:
        raise RequiredStop(name, scope)


def sha(path, progress=None):
    digest = hashlib.sha256()
    read = 0
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
            read += len(block)
            if progress is not None:
                progress(read)
    return digest.hexdigest()


def rows(path):
    with (ROOT / path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, records, fields=None):
    records = list(records)
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save_json(path, data):
    (ROOT / path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def source_hash(path):
    global MANIFESTS
    path = ROOT / path
    rel = str(path.relative_to(ROOT))
    if MANIFESTS is None:
        MANIFESTS = {}
        for name in ('st1_exit_manifest.csv', 'st2_exit_manifest.csv'):
            MANIFESTS.update({r['path']: r for r in rows('reports/' + name)})
    check('Input in frozen manifest', rel in MANIFESTS)
    value = sha(path)
    check('Frozen input SHA256', value, MANIFESTS[rel]['sha256'])
    READ_HASHES[rel] = value
    return value


def emit(stage, status='running', **progress):
    # Never emit data-bearing exception arguments, window IDs or findings.
    payload = dict(stage=stage, status=status, **progress)
    with (ROOT / 'reports/st3/st3_ops_status.jsonl').open('a') as handle:
        handle.write(json.dumps(payload) + '\n')
    print(json.dumps(payload), flush=True)


def wall_check():
    approval = json.loads((ROOT / 'reports/st3/instruction_approval.json').read_text())
    from datetime import datetime, timezone
    start = datetime.fromisoformat(approval['started_utc'])
    check('Bundle CPU wall ceiling', (datetime.now(timezone.utc) - start).total_seconds() < 24 * 3600)


def finish(stage, fn, success_status='complete'):
    began = time.monotonic()
    try:
        fn()
        status = success_status
        receipt = dict(status=status)
    except RequiredStop as exc:
        status = 'stopped_required_check'
        receipt = dict(status=status, check=exc.check, scope=exc.scope)
    except Exception as exc:
        import traceback
        status = 'implementation_error'
        receipt = dict(status=status, error_type=type(exc).__name__,
                       stack_locations=[dict(file=Path(f.filename).name, line=f.lineno, function=f.name)
                                        for f in traceback.extract_tb(exc.__traceback__)])
    receipt['process_wall_seconds'] = time.monotonic() - began
    phase = Path(fn.__code__.co_filename).stem
    save_json('reports/st3/st3_' + phase + '_completion.json', receipt)
    with (ROOT / 'reports/st3/st3_job_receipts.jsonl').open('a') as handle:
        handle.write(json.dumps(dict(phase=phase,stage=stage,**receipt))+'\n')
    save_json('reports/st3/st3_' + stage.lower().replace('-', '_') + '_completion.json', receipt)
    emit(stage, status=status, **{k: receipt[k] for k in ('check', 'scope', 'error_type') if k in receipt})
    if status != success_status:
        update(stage, status=status, label=receipt.get('check', status))
        raise SystemExit(1)
