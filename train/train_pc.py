"""Training harness for (ii-pc) and the paired (ii) equivalence check (repro environment).

The step loop is the archived one (st9t/worker.py, st12c/rhythm_train_worker_v2.py and
st30t/pooled_train.py share it): same initialize/step backends, same FixedBatches stream
(2 originals + 2 pool items, default_rng([seed,0]) / ([seed,1])), same epochs and steps,
last-epoch checkpoint only. Only output locations differ (OPS2 workspace).
Usage: train_pc.py BACKBONE SEED POOL_MANIFEST OUT_DIR [MAX_STEPS]
  POOL_MANIFEST rows carry the original pool row plus ops2_view_path/ops2_rawlabel_path
  (uint8 storage, brief 02 section 6.1; model preprocessing at load time in train/u8_store.py),
  or 'ORIGINAL' to train on the original stored (ii) pool for the harness check.
  OPS2_ARM=i (brief 02 6.12): the paired (i) run's pool, stream arm and order instead of (ii)
  (PhysNet st9t arm i, RhythmFormer st13x arm i, matched FactorizePhys st30t FPm_A PURE arm i);
  (i-pc) PhysNet/RhythmFormer rows add ops2_label_path (new label file, stored (i) input read as is),
  matched FactorizePhys rows add ops2_view_path/ops2_rawlabel_path (uint8 storage). Default OPS2_ARM=ii.
  OPS2_LOSS=rf_author (brief 02 6.19 C / 6.29, PhysNet only): loss swap. The RhythmFormer author loss
  (src/st12c/rhythm_loss_author.RhythmFormer_Loss: 0.2 x negative Pearson + 1.0 x frequency cross-entropy + 1.0 x HR
  distribution KL, Fs 30, computed per sample and averaged over the batch as in the archived RhythmFormer step) replaces
  PhysNet's negative Pearson. diff_flag = 1 because PhysNet labels are DiffNormalized (the toolbox RhythmFormer trainer
  sets diff_flag from the label type). Everything else is the PhysNet (ii) run: model, initial weights, optimizer,
  scheduler, batch stream, the PhysNet step's batch standardization of label and prediction, epochs and steps.
  OPS2_LR_FACTOR=f (brief 02 6.67 마, diagnostic rule 6, replicate runs): the configured learning rate (the maximum of the
  OneCycle schedule; also the optimizer's initial argument) is multiplied by f, e.g. 1.000001. Unset: unchanged.
  Applied after setup(), so every other setting, the initial-weight digest check, pool, order and stream stay as they are.
"""
import os  # noqa: E402  (first: thread settings must precede numpy/torch/cv2 imports; children inherit)
# brief 02 6.14/6.16: OPS2_THREADS=one (default) fixes BLAS/OpenMP to 1 thread; OPS2_THREADS=default states the
# library defaults of the earlier unrestricted runs on this 16-CPU server explicitly (OpenMP 8, OpenBLAS 16,
# MKL 8). OpenCV (pthreads) keeps its unrestricted default of 16 threads in both modes, also under CPU affinity.
os.environ.update({'one': dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                   'default': dict(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='16', MKL_NUM_THREADS='8')}[
    os.environ.get('OPS2_THREADS', 'one')], OPENCV_FOR_THREADS_NUM='16')
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
V = WORK / 'vendor' / 'st8_6ebbc9cd'
sys.path.insert(0, str(V))
os.chdir(V)


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


MAIN_OUT = Path('<MACHINE_PATH>')  # read-only (i) pool manifests


def setup(backbone, arm_kind=None):
    arm_kind = arm_kind or os.environ.get('OPS2_ARM', 'ii')
    assert arm_kind in ('ii', 'i')
    config, original, pool, key, init_file, arm = setup_ii(backbone)
    if arm_kind == 'i':
        if backbone == 'PhysNet':
            pool = sorted([r for r in rows(MAIN_OUT / 'st9g_pool_PhysNet_i_manifest.csv') if r['model'] == 'PhysNet' and r['arm'] == 'i'],
                          key=lambda r: int(r['pool_index']))
            assert [int(r['pool_index']) for r in pool] == list(range(len(pool)))
        elif backbone == 'RhythmFormer':
            s = json.loads(Path('<MACHINE_PATH>').read_text())
            assert s['model'] == config and s['original_manifest'] == 'outputs/st12c_a_original_manifest.csv'
            pool = sorted(rows(MAIN_OUT / 'st13x_rf_i_pool_manifest.csv'), key=lambda r: int(r['pool_index']))
        else:
            pool = sorted(rows(MAIN_OUT / 'st30t_FPm_A_PURE_i_pool.csv'), key=lambda r: int(r['training_index']))
            assert [int(r['training_index']) for r in pool] == list(range(len(pool)))
        assert not any(r['generation_excluded'] == 'True' for r in pool)
        arm = 'i'
    return config, original, pool, key, init_file, arm


def setup_ii(backbone):
    if backbone == 'PhysNet':
        s = json.loads((V / 'reports/st9t/st9t_settings.json').read_text())
        config = s['models']['PhysNet']
        original = sorted([r for r in rows(V / 'outputs/st9g_original_manifest.csv') if r['model'] == 'PhysNet'],
                          key=lambda r: int(r['chunk_order']))
        pool = sorted(rows(V / 'outputs/st9g_pool_PhysNet_ii-h_manifest.csv'), key=lambda r: int(r['pool_index']))
        key, init_file, arm = 'PhysNet', 'reports/st9t/st9t_initial_weights.json', 'ii-h'
    elif backbone == 'RhythmFormer':
        s = json.loads((V / 'reports/st12c/st12c_a_training_settings.json').read_text())
        config = s['model']
        original = sorted(rows(V / 'outputs/st12c_a_original_manifest.csv'), key=lambda r: int(r['chunk_order']))
        pool = sorted(rows(V / 'outputs/st12c_a_pool_manifest.csv'), key=lambda r: int(r['pool_index']))
        key, init_file, arm = 'RhythmFormer', 'reports/st12c/st12c_initial_weights.json', 'ii'
    else:
        s = json.loads((V / 'reports/st30t/FPm_A_settings.json').read_text())
        config = s['configs']['PURE']
        original = sorted(rows(V / 'outputs/st30t_FPm_A_PURE_originals.csv'), key=lambda r: int(r['training_index']))
        pool = sorted(rows(V / 'outputs/st30t_FPm_A_PURE_ii_pool.csv'), key=lambda r: int(r['training_index']))
        key, init_file, arm = 'FactorizePhys', 'reports/st9t/st9t_initial_weights.json', 'ii'
    return config, original, pool, key, init_file, arm


def main():
    backbone, seed, pool_manifest, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    config, original, pool, key, init_file, arm = setup(backbone)
    lr_factor = os.environ.get('OPS2_LR_FACTOR', '')
    lr_configured = float(config['learning_rate'])
    if lr_factor:  # brief 02 6.67 마: replicate runs (diagnostic rule 6)
        config = dict(config, learning_rate=lr_configured * float(lr_factor))
    total = int(config['epochs']) * int(config['steps_per_epoch'])
    max_steps = int(sys.argv[5]) if len(sys.argv) > 5 else total
    if pool_manifest != 'ORIGINAL':
        new = {(r['chunk_id'], r['augmentation_index']): r for r in rows(pool_manifest)}
        partial = os.environ.get('OPS2_PARTIAL_POOL') == '1'  # equivalence check: only drawn items exist
        assert partial or len(new) == len(pool), 'complete pool required for training'
        # stats: 6.6; label: 6.12; view_offset: (i-pc-g80) chunk-only views (train/u8_store.py item_input), '' elsewhere
        keys = ('ops2_view_path', 'ops2_rawlabel_path', 'ops2_stats_path', 'ops2_label_path', 'ops2_view_offset')
        pool = [dict(r, **{k: new[(r['chunk_id'], r['augmentation_index'])].get(k, '') for k in keys})
                if (r['chunk_id'], r['augmentation_index']) in new else r for r in pool]
        # brief 02 6.43 라: label arms with a changed target carry it in target_HR (target_HR_original = the pool's);
        # it is used only for the exposure record (labels come from the item files)
        pool = [dict(r, target_HR=new[(r['chunk_id'], r['augmentation_index'])]['target_HR'])
                if new.get((r['chunk_id'], r['augmentation_index']), {}).get('target_HR_original') not in (None, '') else r
                for r in pool]
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(WORK / 'train'))
    from u8_store import loader
    loss_swap = os.environ.get('OPS2_LOSS', '') == 'rf_author'
    if backbone == 'RhythmFormer':
        assert not loss_swap, 'loss swap is defined for PhysNet only'
        from src.st12c.rhythm_backend_v2 import initialize, step
        model, criterion, optimizer, scheduler = initialize(config, seed)
    else:
        from src.st9s.backend import initialize, step as step9
        name = 'FactorizePhys' if backbone == 'FPm' else 'PhysNet'
        model, criterion, optimizer, scheduler = initialize(name, config, seed, 'cuda:0')
    if loss_swap:
        assert backbone == 'PhysNet', 'loss swap is defined for PhysNet only'
        from src.st12c.rhythm_loss_author import RhythmFormer_Loss
        criterion = RhythmFormer_Loss()

        def step_rf_loss(model, criterion, optimizer, scheduler, x, y, epoch):
            # the archived PhysNet step (src/st9s/backend.step) with only the criterion line replaced
            optimizer.zero_grad()
            y = (y - torch.mean(y)) / torch.std(y)
            pred = model(x)[0]
            pred = (pred - torch.mean(pred)) / torch.std(pred)
            loss = sum(criterion(pred[i], y[i], epoch, 30, 1) for i in range(len(x))) / len(x)
            if not torch.isfinite(loss):
                return loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss
    from src.st9s.backend import state_digest
    initial = state_digest(model)
    expected = json.loads((V / init_file).read_text())[f'{key}|{seed}']['sha256']
    assert initial == expected, 'initial weights differ from the recorded same-seed digest'
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    workers = int(os.environ.get('OPS2_DATA_WORKERS', '4'))
    iterator = iter(loader(backbone, original, pool, seed, arm, total, workers))
    out.mkdir(parents=True, exist_ok=True)
    losses, exposures = [], []
    started = time.time()
    status = 'running'
    try:
        for k in range(max_steps):
            tick = time.time()
            x, y, kinds, numbers, high = next(iterator)
            x = x.to('cuda:0', non_blocking=True)
            y = y.to('cuda:0', non_blocking=True)
            epoch = k // int(config['steps_per_epoch']) + 1
            if backbone == 'RhythmFormer':
                loss = step(model, criterion, optimizer, scheduler, x, y, epoch - 1)
            elif loss_swap:
                loss = step_rf_loss(model, criterion, optimizer, scheduler, x, y, epoch - 1)
            else:
                loss = step9(name, model, criterion, optimizer, scheduler, x, y)
            torch.cuda.synchronize()
            value = float(loss.detach())
            losses.append(dict(step=k + 1, epoch=epoch, loss=value, learning_rate=optimizer.param_groups[0]['lr'],
                               step_seconds=time.time() - tick))
            for kind, n in zip(kinds.tolist(), numbers.tolist()):
                r = (original if kind == 0 else pool)[n]
                exposures.append(dict(step=k + 1, kind=kind, index=n,
                                      label_HR=float(r['reference_HR'] if kind == 0 else r['target_HR'])))
            if not np.isfinite(value):
                status = 'stopped_nonfinite_loss'
                break
            if (k + 1) % 50 == 0:
                print(f'{backbone} s{seed} step {k + 1}/{max_steps} loss {value:.4f} {time.time() - started:.0f}s', flush=True)
        if status == 'running':
            status = 'complete' if max_steps == total else 'partial_check'
        if status == 'complete':
            final = out / 'final_epoch.pth'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                            epoch=int(config['epochs']), step=total, run_id=out.name, model_name=backbone, seed=seed,
                            initial_weights_sha256=initial, pool_manifest=pool_manifest), final)
    finally:
        shutdown_error = ''
        if hasattr(iterator, '_shutdown_workers'):
            try:  # a worker abort at shutdown must not lose the step/exposure records
                iterator._shutdown_workers()
            except RuntimeError as e:
                shutdown_error = str(e)
        with open(out / 'steps.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(losses[0]))
            w.writeheader()
            w.writerows(losses)
        with open(out / 'exposures.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(exposures[0]))
            w.writeheader()
            w.writerows(exposures)
    result = dict(status=status, backbone=backbone, seed=seed, steps=len(losses), total=total, final_loss=losses[-1]['loss'],
                  initial_weights_sha256=initial, initial_matches_record=True, pool_manifest=pool_manifest,
                  seconds=time.time() - started, checkpoint_sha256=sha(out / 'final_epoch.pth') if status == 'complete' else '',
                  worker_shutdown_error=shutdown_error, loss=('rf_author (diff_flag 1)' if loss_swap else 'archived'),
                  learning_rate_configured=lr_configured, learning_rate_factor=lr_factor or None,
                  learning_rate_used=float(config['learning_rate']),
                  blas_threads={k: os.environ.get(k, '') for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    (out / 'result.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
