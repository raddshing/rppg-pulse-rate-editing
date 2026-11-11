"""Blind-safe progress that can be watched from an ordinary SSH terminal."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PATHS = json.loads((ROOT / 'paths.yaml').read_text())
PROGRESS = Path(PATHS['cache']) / 'st3/progress'
STAGES = [('S3-0', 'Freeze and readiness'), ('S3-1', 'GT defects and restoration'),
          ('S3-2', 'Mechanism measurements'), ('S3-3', 'Gates and baselines'),
          ('S3-4', 'Counterfactuals and editor validation'), ('S3-5', 'PI review and delivery')]
# Remaining phases are forecasts until their own throughput is measured.
PHASES={
 'S3-0':[('S/G/driver geometry readiness',60)],
 'S3-1':[('GT defect map',600),('r0 full GT-path reproduction',600),
         ('GT-only diffusion training',2700),('Restoration validation',180),
         ('Shared restoration null controls',9300),('C4 descriptive RG1 unit gains',1020)],
 'S3-2':[('Driver-only HB exclusions and nulls',1800),('C1 W and skin-signal spectra',27000),
         ('HB fixed two-way cluster bootstrap',600),('HC saved-wave sensitivity',3000)],
 'S3-3':[('C2 PURE features',600),('C2 MMPD-A features',1200),
         ('Official bT and b0 baselines',3600)],
 'S3-4':[('3R streaming clips and RW15 inference',8640),('3E additive-editor validation',12960)],
 'S3-5':[('Blind crosstab and frozen-input checks',300),('Face-free result figures',120),
         ('Final-size figure inspection',600),('Final archive checks',30)]}


def update(key, *, done=0, total=None, status='running', label=None,
           projected_seconds=None, reset=False):
    """Only work quantities/times and fixed stage labels may enter this file."""
    if key not in dict(STAGES):
        raise ValueError('Unknown authorized progress stage')
    PROGRESS.mkdir(parents=True, exist_ok=True)
    path = PROGRESS / (key + '.json')
    old = json.loads(path.read_text()) if path.exists() and not reset else {}
    now = time.time()
    started = old.get('started_epoch', now)
    if old.get('status') == 'pending' and status == 'running':
        started = now
    elapsed = now - started
    previous_done = old.get('phase_start_done', done)
    phase_start = old.get('phase_started_epoch', now)
    if old.get('status') == 'pending' and status == 'running':
        previous_done, phase_start = done, now
    if label is not None and label != old.get('label'):
        previous_done, phase_start = done, now
    if label == old.get('label') and done < old.get('done',0):
        # Recomputing a repaired implementation starts a new rate estimate;
        # preserve the total stage clock and charged resource receipts.
        previous_done,phase_start=done,now
    phase_elapsed = now - phase_start
    remaining = None
    if total is not None and done > previous_done:
        remaining = phase_elapsed / (done - previous_done) * max(0, total - done)
    elif projected_seconds is not None:
        remaining = max(0, projected_seconds - phase_elapsed)
    elif old.get('remaining_seconds') is not None:
        remaining = max(0, old['remaining_seconds'] - (now - old['updated_epoch']))
    if status == 'complete':
        remaining = 0
    if status.startswith(('stopped','implementation_error')):
        remaining = 0
    if status.startswith('waiting'):
        remaining = None
    record = dict(stage=key, label=label or old.get('label', dict(STAGES)[key]),
                  status=status, done=done, total=total, started_epoch=started,
                  updated_epoch=now, phase_started_epoch=phase_start,
                  phase_start_done=previous_done, elapsed_seconds=elapsed,
                  remaining_seconds=remaining,phase_elapsed_seconds=phase_elapsed)
    tmp = PROGRESS / (key + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(record) + '\n')
    os.replace(tmp, path)
    return record


def duration(seconds):
    if seconds is None:
        return '?'
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return '%02d:%02d:%02d' % (hours, minutes, seconds)


def snapshot():
    now = time.time()
    forecast_path=PROGRESS/'st3_phase_forecasts.json'
    try:forecasts=json.loads(forecast_path.read_text())
    except (OSError,ValueError):forecasts={}
    rows = []
    for key, title in STAGES:
        path = PROGRESS / (key + '.json')
        try:
            r = json.loads(path.read_text())
        except (OSError, ValueError):
            r = dict(stage=key, label=title, status='pending', done=0, total=None,
                     elapsed_seconds=0, remaining_seconds=None)
        if r['status'] == 'running':
            r['elapsed_seconds'] = now - r['started_epoch']
            r['phase_elapsed_seconds'] = now - r['phase_started_epoch']
            if r['remaining_seconds'] is not None:
                # ETA is an estimate, never turn stalled work into completion.
                r['remaining_seconds'] = max(0, r['remaining_seconds'] -
                                                (now - r['updated_epoch']))
                if r['remaining_seconds'] == 0:
                    r['remaining_seconds'] = None
        phases=PHASES.get(key,[])
        matches=[i for i,(name,_) in enumerate(phases) if name==r['label']]
        future=sum(forecasts.get(name,seconds) for name,seconds in phases[matches[0]+1:]) if matches else 0
        r['future_phase_forecast_seconds']=future
        r['overall_remaining_seconds']=r['remaining_seconds']+future if r['remaining_seconds'] is not None else None
        if r['status']=='complete' or r['status'].startswith(('stopped','implementation_error')):
            r['overall_remaining_seconds']=0
        rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=2)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    rendered = 0
    try:
        while True:
            rows = snapshot()
            if args.json:
                print(json.dumps(rows), flush=True)
            else:
                if rendered and sys.stdout.isatty():
                    sys.stdout.write('\033[%dA' % rendered)
                lines = []
                for r in rows:
                    total, done = r['total'], r['done']
                    ratio = min(1, done / total) if total else (1 if r['status'] == 'complete' else 0)
                    bar = '#' * int(ratio * 20) + '-' * (20 - int(ratio * 20))
                    count = ('%d/%d' % (done, total)) if total is not None else '?/?'
                    display_elapsed=r.get('phase_elapsed_seconds',r['elapsed_seconds']) if r['status']=='running' else r['elapsed_seconds']
                    line = '%s [%s] %5.1f%% %s [%s<%s] %s: %s' % (
                        r['stage'], bar, 100 * ratio, count, duration(display_elapsed),
                        duration(r['remaining_seconds']), r['status'], r['label'])
                    lines.append(line)
                complete = sum(r['status'] == 'complete' for r in rows[:5])
                stopped=sum(r['status'].startswith(('stopped','implementation_error')) for r in rows[:5])
                pending = [r for r in rows[:5] if r['status'] != 'complete' and not r['status'].startswith(('stopped','implementation_error'))]
                if (ROOT/'reports/st3/st3_pi_tag_save_notification.json').exists() and rows[5]['status']!='complete' and not rows[5]['status'].startswith(('stopped','implementation_error')):
                    pending.append(rows[5])
                known = all(r['overall_remaining_seconds'] is not None for r in pending)
                # A sum is a conservative sequential estimate when jobs overlap.
                eta = sum(r['overall_remaining_seconds'] for r in pending) if known else None
                lines.append('Compute stages complete %d/5, stopped %d/5 | compute + reporting remaining (sequential estimate): %s | '
                             'Delivery: %s' % (complete,stopped,duration(eta),rows[5]['status']))
                for line in lines:
                    print(('\033[2K' if sys.stdout.isatty() else '') + line, flush=True)
                rendered = len(lines)
            if not args.watch:
                break
            time.sleep(max(.2, args.interval))
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    main()
