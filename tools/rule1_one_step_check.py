"""Diagnostic rule 1, 6.91 가 condition 2: before the six Pearson-only runs, one step of the standard RhythmFormer run and one step
of the wrapper (train/train_pc_pearson.py) with the same seed and pool must differ only in the loss.

Runs train/train_pc.py and train/train_pc_pearson.py with max_steps = 1 into scratch folders (deleted afterwards), then compares:
- the first batch: exposures.csv of step 1 (kind, index, ... every column), byte for byte;
- initial weights: result.json initial_weights_sha256 and initial_matches_record;
- settings dump: the settings train_pc.py's setup() returns for RhythmFormer (both runs use it; dumped once) plus every
  result.json field other than the timing, the loss value and the checkpoint hash;
- the loss: criterion class (RhythmFormer_Loss vs PearsonOnly, from the wrapper's loss.json) and the step-1 loss values (reported).
Usage: rule1_one_step_check.py SEED POOL_MANIFEST|ORIGINAL ARM_LABEL GPU     Output: outputs/rule1/one_step_<ARM_LABEL>_s<SEED>.json
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.08: machine paths, see config/paths.example.json
REPRO = CFG['python_repro']
SKIP = {'seconds', 'final_loss', 'checkpoint_sha256', 'worker_shutdown_error', 'loss', 'blas_threads', 'status'}


def run(script, seed, pool, out, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPS2_DATA_WORKERS='3')
    p = subprocess.run([REPRO, '-B', '-s', str(WORK / 'train' / script), 'RhythmFormer', str(seed), pool, str(out), '1'],
                       env=env, capture_output=True, text=True)
    (out.parent / f'{out.name}.log').write_text(p.stdout[-4000:] + p.stderr[-4000:])
    return p.returncode


def dump(mode, seed, gpu):
    """Settings as each run builds them: train_pc.py setup() for RhythmFormer, then the backend initialize() (patched by the
    wrapper for 'pearson'): config, criterion class, optimizer and scheduler settings."""
    code = f"""
import json, sys, runpy
sys.argv = ['x', 'RhythmFormer', '{seed}', 'ORIGINAL', '/dev/null']
if '{mode}' == 'pearson':
    src = open('{WORK}/train/train_pc_pearson.py').read().split('backend.initialize = initialize')[0] + 'backend.initialize = initialize'
    exec(compile(src, 'train_pc_pearson.py (patch part)', 'exec'), dict(__file__='{WORK}/train/train_pc_pearson.py', __name__='wrapper_patch'))
sys.path.insert(0, '{WORK}/train')
import train_pc
config = train_pc.setup('RhythmFormer')[0]
import src.st12c.rhythm_backend_v2 as be
model, criterion, optimizer, scheduler = be.initialize(config, {seed})
g = optimizer.param_groups[0]
print(json.dumps(dict(config=config, criterion=type(criterion).__name__, optimizer=type(optimizer).__name__,
                      optimizer_settings={{k: (v if isinstance(v, (int, float, str, bool, type(None))) else list(v) if isinstance(v, tuple) else str(v)) for k, v in g.items() if k != 'params'}},
                      scheduler=type(scheduler).__name__), sort_keys=True))
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    p = subprocess.run([REPRO, '-B', '-s', '-c', code], env=env, capture_output=True, text=True, cwd=str(WORK / 'train'))
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    seed, pool, label, gpu = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    scratch = WORK / 'cache' / 'rule1' / 'one_step'
    scratch.mkdir(parents=True, exist_ok=True)
    a, b = scratch / f'standard_{label}_s{seed}', scratch / f'pearson_{label}_s{seed}'
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)
    rc = dict(standard=run('train_pc.py', seed, pool, a, gpu), pearson=run('train_pc_pearson.py', seed, pool, b, gpu))
    ra, rb = json.loads((a / 'result.json').read_text()), json.loads((b / 'result.json').read_text())
    ea, eb = (a / 'exposures.csv').read_text().splitlines(), (b / 'exposures.csv').read_text().splitlines()
    sa, sb = (a / 'steps.csv').read_text().splitlines(), (b / 'steps.csv').read_text().splitlines()
    sys.path.insert(0, str(WORK / 'train'))
    other = sorted(k for k in set(ra) | set(rb) if k not in SKIP and ra.get(k) != rb.get(k))
    res = dict(seed=seed, pool=pool, arm=label, returncodes=rc,
               first_batch_equal=ea == eb, exposure_rows=len(ea) - 1,
               initial_weights_equal=ra['initial_weights_sha256'] == rb['initial_weights_sha256'],
               initial_weights_sha256=ra['initial_weights_sha256'], initial_matches_record=[ra['initial_matches_record'], rb['initial_matches_record']],
               other_result_fields_differing=other, criterion=dict(standard='RhythmFormer_Loss (train_pc.py, archived)',
                                                                    pearson=json.loads((b / 'loss.json').read_text())['loss']),
               step1_rows=dict(standard=sa[1:], pearson=sb[1:]), checkpoint_sha_differs=ra['checkpoint_sha256'] != rb['checkpoint_sha256'],
               result_fields_compared=sorted(k for k in set(ra) | set(rb) if k not in SKIP))
    da, db = dump('standard', seed, gpu), dump('pearson', seed, gpu)
    orig = WORK / 'cache' / 'train' / f'RhythmFormer_{label}_s{seed}' / 'exposures.csv'  # OPS2 standard run, if it exists
    if orig.exists():  # added for (ii-pc): its standard runs are OPS2 runs (the (ii) standard runs are archived October runs)
        eo = orig.read_text().splitlines()
        res['first_batch_equals_original_standard_run'] = [eo[0]] + [x for x in eo[1:] if x.split(',')[0] == '1'] == ea
        res['original_standard_run_initial_weights_equal'] = (
            json.loads(orig.with_name('result.json').read_text())['initial_weights_sha256'] == ra['initial_weights_sha256'])
    res['settings_dump'] = dict(standard=da, pearson=db)
    res['settings_keys_differing'] = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
    res['only_loss_differs'] = bool(res['settings_keys_differing'] == ['criterion'] and res['first_batch_equal'] and res['initial_weights_equal'] and not other and rc['standard'] == 0 and rc['pearson'] == 0)
    out = WORK / 'outputs' / 'rule1' / f'one_step_{label}_s{seed}.json'
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('returncodes', 'first_batch_equal', 'initial_weights_equal', 'other_result_fields_differing', 'only_loss_differs')}))
    for d in (a, b):
        shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    main()
