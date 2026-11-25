"""Brief 02 7.02 마: B9 (frame-number template subtraction) results copied from the outputs finished on 11-12 (no recalculation).

Source: outputs/b9/b9_results.csv and b9_counts.csv (b9/analyze_b9.py), RhythmFormer MCD-dev, draft rule, wide band, seeds
20261131-33, common windows of the 21 runs x 2 conditions; one bootstrap function for all arms and conditions (b1/boot.py,
B = 10,000). 'standard' = the stored development inference read at the time (outputs/1b, before the guarded re-inference);
'B9' = the re-inference with the frame-number mod-12 template subtracted from the native frames (b9/line_removal.py).
Line: predicted HR within 147.66-154.69 bpm (key-frame line).
Copied: line share of the high-HR windows per arm (before / after removal and the change); high-HR MAE contrasts C2
(ii-pc)-(ii), C3 (i-pc)-(i), C4 (ii-pc)-(i-pc) (and C1 (ii-pc)-(0)) under both conditions and their change; file dates.
Writes outputs/posthoc702/ma/{ma_line,ma_contrasts,ma_counts,ma_provenance}.csv.
"""
import os
from datetime import datetime

from p702_common import SRC_B9, WORK, np, pd, stamp, time, write

NAMES = {'(ii-pc)-(ii)': 'C2', '(i-pc)-(i)': 'C3', '(ii-pc)-(i-pc)': 'C4', '(ii-pc)-(0)': 'C1'}


def mtime(p):
    return datetime.fromtimestamp(os.stat(p).st_mtime).strftime('%Y-%m-%d %H:%M')


def main():
    t0 = time.time()
    src = WORK / 'outputs' / 'b9'
    r = pd.read_csv(src / 'b9_results.csv')
    arms = ['(0)', '(i)', '(ii)', '(i-pc)', '(ii-pc)', '(ii-lag)', '(ii-inv)']
    line = r[r.item.isin(arms) & r.metric.isin(['line_high', 'line_low', 'MAE_high', 'MAE_low'])].copy()
    line['condition'] = line.condition.map({'standard': 'before removal (standard)', 'B9': 'after removal (B9)',
                                            'B9 - standard': 'change (after - before)'})
    line['source'] = SRC_B9
    con = r[r.item.isin(NAMES) & r.metric.isin(['MAE_high', 'MAE_low'])].copy()
    con.insert(1, 'name', con.item.map(NAMES))
    con['condition'] = con.condition.map({'standard': 'before removal (standard)', 'B9': 'after removal (B9)',
                                          'B9 - standard': 'change (after - before)'})
    con['source'] = SRC_B9
    ch = con[con.condition.eq('change (after - before)') & con.name.isin(['C2', 'C3', 'C4']) & con.metric.eq('MAE_high')]
    counts = pd.read_csv(src / 'b9_counts.csv').assign(source=SRC_B9)
    guarded_log = WORK / 'outputs' / 'guarded' / 'mcd_dev_infer_log.csv'
    g = pd.read_csv(guarded_log)
    gtimes = f"{g.finished.min()} to {g.finished.max()} ({len(g)} recordings)"
    prov = pd.DataFrame([
        dict(item='B9 window predictions and results', value=f"b9_results.csv {mtime(src / 'b9_results.csv')}; "
                                                              f"re-inference windows {mtime(src / 'mcd_dev_preds_b9_g3.csv.gz')}"),
        dict(item='per-recording inference check', value=mtime(src / 'check_infer.csv')),
        dict(item='guarded MCD-dev re-inference (log, finished)', value=gtimes),
        dict(item='predictions guarded', value='no: B9 and its standard comparison were read from the stored development '
                                               'inference as read on 11-12; the guarded re-inference ran on 11-14'),
        dict(item='largest |change| of a high-HR contrast (C2-C4)',
             value=f"{ch.point.abs().max():.4f} bpm ({ch.loc[ch.point.abs().idxmax(), 'name']})"),
    ]).assign(source=SRC_B9)
    write(line, 'ma_line.csv', 'ma')
    write(con, 'ma_contrasts.csv', 'ma')
    write(counts, 'ma_counts.csv', 'ma')
    write(prov, 'ma_provenance.csv', 'ma')
    print(prov.to_string())
    stamp('ma', seconds=time.time() - t0, copied_from=['outputs/b9/b9_results.csv', 'outputs/b9/b9_counts.csv'], recalculated=False,
          max_abs_change_C2_C4_MAE_high=float(ch.point.abs().max()))


if __name__ == '__main__':
    main()
