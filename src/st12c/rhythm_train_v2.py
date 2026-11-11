"""Resume the fixed RF sequence after restoring the pinned author loss helpers."""
import os
os.environ['ST12_NAMESPACE'] = 'st12c'
from src.st12.ops import *
from src.st12 import gpu
from src.st12c.rhythm_repair import verify_repair
import sys


def main():
    arm = sys.argv[1]
    physical = int(sys.argv[2])
    check('Fixed RF arm/GPU', arm in ['0', 'ii'] and physical in [0, 1])
    repair = verify_repair()
    settings = json.loads((REPORTS / 'st12c_a_training_settings.json').read_text())
    summary_path = REPORTS / 'st12c_a_timing_summary_v2.json'
    summary = json.loads(summary_path.read_text())
    check('Corrected timing completed and committed before training',
          sha(summary_path) in (ROOT / 'reports/ST4_INSTR05B_PREFIX.md').read_text() and
          summary['training_can_start'])
    timing = rows('outputs/st12c_a_timing_v2.csv')
    estimates = {a: float(next(r['projected_run_seconds'] for r in timing
                              if r['arm'] == ('ii-h' if a == 'ii' else a))) for a in ['0', 'ii']}
    pending = [r for r in settings['runs'] if not (REPORTS / (r['run_id'] + '_result.json')).exists()]
    if len(pending) == len(settings['runs']):
        projected = gpu.total('C5_a') + sum(estimates[r['arm']] for r in pending)
        cap = gpu.caps()['C5_a']
        if projected > cap:
            save_json(ROOT / 'reports/st12/C5_a_resource_wait.json', dict(status='resource_wait',
                reason='Restored author-loss200step projection before first complete RF training run',
                projected_seconds=projected, cap_seconds=cap,
                additional_seconds_with15percent_margin=max(0, 1.15 * projected - cap), at_utc=now()))
            raise ResourceWait('Corrected RF timing projection exceeds account before training')
    results, exposure, durations = [], [], []
    for request in settings['runs']:
        run_id = request['run_id']
        canonical = REPORTS / (run_id + '_result.json')
        if canonical.exists():
            result = json.loads(canonical.read_text())
            check('Preserve terminal RF run', result['status'] in ['complete', 'stopped_nonfinite_loss'])
        else:
            if request['arm'] != arm:
                continue
            path = REPORTS / (run_id + '_request.json')
            attempt = 2 if run_id == repair['failed_run_id'] else 1
            if attempt == 2:
                check('Same immutable failed run request and seed', path.exists() and
                      json.loads(path.read_text()) == request and sha(path) == repair['failed_request_sha256'])
            else:
                check('No implicit RF training retry', not path.exists())
                save_json(path, request)
            account = gpu.run(run_id, 'C5_a', physical, 'repro', 'src.st12c.rhythm_train_worker_v2',
                [str(path)], attempt=attempt, next_unit_seconds=max(durations + [estimates[arm]]))
            result = json.loads((REPORTS / (run_id + '_attempt' + str(attempt) + '_result.json')).read_text())
            if result['status'] == 'resource_wait':
                raise ResourceWait('RF training waits with completed runs retained')
            check('RF training complete or nonfinite current run', account['exit_code'] == 0 and
                  result['status'] in ['complete', 'stopped_nonfinite_loss'], scope='C5_a')
            result.update(GPU_seconds=account['GPU_seconds'], training_attempt=attempt,
                          loss_repair_sha256=repair['receipt_sha256'])
            save_json(canonical, result)
        if result['status'] == 'complete':
            durations.append(result['GPU_seconds'])
        results.append({k: result[k] for k in ['run_id', 'model', 'arm', 'seed', 'status',
            'completed_steps', 'completed_epochs', 'checkpoint_path', 'checkpoint_sha256',
            'checkpoint_bytes', 'GPU_seconds', 'validation_calls', 'evaluation_calls']})
        exposure.append(dict(run_id=run_id, arm=result['arm'], seed=result['seed'], **result['exposure'],
            high_original_fraction=result['exposure']['high_original'] / max(1, result['exposure']['original']),
            high_augmented_fraction=result['exposure']['high_augmented'] / max(1, result['exposure']['augmented'])))
        write_rows('outputs/st12c_a_runs.csv', results)
        write_rows('outputs/st12c_a_exposure.csv', exposure)
    return dict(arm=arm, runs=len(results), completed=sum(r['status'] == 'complete' for r in results),
        nonfinite_stops=sum(r['status'] == 'stopped_nonfinite_loss' for r in results),
        validation_calls=0, evaluation_pending=True, loss_repair_sha256=repair['receipt_sha256'])


if __name__ == '__main__':
    finish('rhythm_train_v2_' + sys.argv[1], main, resource='ANCILLARY_GPU_CONTROLLER')
