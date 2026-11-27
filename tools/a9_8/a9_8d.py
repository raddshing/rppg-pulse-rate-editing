"""OPS2 A9 item 8d (7.13 1): VIPL-HR window flow: recordings and windows removed at each exclusion step, and the reference HR
and position within the recording (window 1-5) of included and excluded windows.

Sources (stored, unchanged): stage (a) recordings.csv (recording exclusions, windows per recording) and references.csv (window
rows: 200-ms flag, signal defects, reference class, verification, gt_HR mean); stage (c2) support (windows valid for every
non-defective confirmatory key) and readouts. Steps, per set (primary: source1 and source3; secondary: source2):
  1 recordings of the set; 2 recordings excluded by stage (a) (reason); 3 windows created (the first five non-overlapping
  10-s windows inside the video and the BVP record, consecutive from the first; position = window index + 1);
  4 windows with an interval > 200 ms between frames (primary analysis excludes them; primary set only);
  5 windows not verified (signal defect; reference estimators not in agreement); 6 windows without a finite reference
  (gt_HR mean); 7 windows outside the confirmatory support (some confirmatory key without a valid readout, e.g. a packed
  prediction shorter than the window end). What remains is the window set of the primary analysis (and of the secondary set).
Reference HR = gt_HR mean of the window (the reference of the primary analysis); high = >= 100 bpm.
Writes outputs/a9_8/8d/{8d_flow,8d_windows}.csv.
"""
import a9_common as C
from a9_common import np, pd


def describe(z):
    ref = z.gt_HR_mean[np.isfinite(z.gt_HR_mean)]
    pos = z.window_index.astype(int) + 1
    return dict(windows=len(z), recordings=z.recording_id.nunique(), subjects=z.subject_id.nunique(),
                reference_finite=len(ref), reference_median=float(ref.median()) if len(ref) else np.nan,
                reference_q1=float(ref.quantile(.25)) if len(ref) else np.nan, reference_q3=float(ref.quantile(.75)) if len(ref) else np.nan,
                reference_ge100=int((ref >= 100).sum()), share_ge100=float((ref >= 100).mean()) if len(ref) else np.nan,
                **{f'position_{k}': int((pos == k).sum()) for k in range(1, 6)})


def main():
    recs = pd.read_csv(C.VSA / 'recordings.csv', dtype={'subject_id': str})
    refs = pd.read_csv(C.VSA / 'references.csv', dtype={'subject_id': str})
    sup = pd.read_csv(C.VC2 / 'support.csv')
    support = set(sup[sup.layer.eq('HAR') & sup.in_support.astype(bool)].window_id)
    flow, wins = [], []
    for st in ('primary', 'secondary'):
        r = recs[recs.set.eq(st)]
        flow.append(dict(set=st, step=1, description='recordings of the set', recordings_removed=0, windows_removed=0,
                         recordings_remaining=len(r), windows_remaining=np.nan))
        for reason, g in r[r.exclusion.notna()].groupby('exclusion'):
            flow.append(dict(set=st, step=2, description=f'recording excluded: {reason}', recordings_removed=len(g), windows_removed=0,
                             recordings_remaining=np.nan, windows_remaining=np.nan))
        kept = r[r.exclusion.isna()]
        w = refs[refs.set.eq(st) & refs.recording_id.isin(kept.recording_id)].copy()
        assert len(w) == int(kept.windows.sum())
        flow.append(dict(set=st, step=3, description='windows created (first five 10-s windows inside video and BVP record)',
                         recordings_removed=int((kept.windows == 0).sum()), windows_removed=0,
                         recordings_remaining=w.recording_id.nunique(), windows_remaining=len(w),
                         windows_per_recording=';'.join(f'{k}:{v}' for k, v in kept.windows.value_counts().sort_index().items())))
        w['out'] = ''
        steps = []
        if st == 'primary':
            steps.append((4, 'interval > 200 ms between frames (200-ms rule)', w.gap200.astype(bool)))
        defect = w.defect_any.astype(int).eq(1)
        steps.append((5, 'not verified: signal defect (d1-d4)', ~w.verified.astype(bool) & defect))
        steps.append((5, 'not verified: reference estimators disagree', ~w.verified.astype(bool) & ~defect & w.reference_class.eq('disagreement')))
        steps.append((5, 'not verified: harmonic suspect', ~w.verified.astype(bool) & ~defect & w.reference_class.eq('harmonic_suspect')))
        steps.append((5, 'not verified: other', ~w.verified.astype(bool) & ~defect & w.reference_class.eq('agreement')))
        steps.append((6, 'no finite reference (gt_HR mean)', ~np.isfinite(w.gt_HR_mean)))
        steps.append((7, 'outside the confirmatory support (a confirmatory key without a valid readout)', ~w.window_id.isin(support)))
        for k, desc, m in steps:
            m = m & w.out.eq('')
            before = w[w.out.eq('')]
            w.loc[m, 'out'] = desc
            after = w[w.out.eq('')]
            flow.append(dict(set=st, step=k, description=desc, recordings_removed=before.recording_id.nunique() - after.recording_id.nunique(),
                             windows_removed=int(m.sum()), recordings_remaining=after.recording_id.nunique(), windows_remaining=len(after)))
            wins.append(dict(set=st, group=f'excluded at step {k}: {desc}', **describe(w[w.out.eq(desc)])))
        final = w[w.out.eq('')]
        wins.append(dict(set=st, group='included (analysis windows)', **describe(final)))
        wins.append(dict(set=st, group='all created windows', **describe(w)))
        flow.append(dict(set=st, step=8, description='analysis windows', recordings_removed=0, windows_removed=0,
                         recordings_remaining=final.recording_id.nunique(), windows_remaining=len(final)))
    C.write(pd.DataFrame(flow), '8d_flow.csv', '8d')
    C.write(pd.DataFrame(wins), '8d_windows.csv', '8d')
    C.stamp('8d')
    print(pd.DataFrame(flow)[['set', 'step', 'description', 'recordings_removed', 'windows_removed', 'recordings_remaining', 'windows_remaining']].to_string())


if __name__ == '__main__':
    main()
