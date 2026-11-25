"""OPS2 7.07t: figure 4 data for VIPL-HR, primary set (reference: mean of gt_HR, the primary analysis of A7/(c1)), mean
signed error per 10-bpm reference bin, arms (0), (ii-pc), (i-simple), three backbones, seeds 20261131-20261133.

Same path as the M4 bins of stage (c1) (eval5/vipl_c.py -> families.fam_m4): windows of the population 'primary' (verified,
primary set, no 200-ms gap), HAR band, inside the (c1)/(c2) common support and valid for the key (stage3_analyze.restrict);
per seed the mean of (predicted - reference) over the bin's windows, then the mean over the three seeds; windows and
subjects are those of the first seed (fam_m4). Readouts: stage (c2) (every key of (b1) and (b2)). Check: the rows of (0) and
(ii-pc) equal the stored (c1) m4_bins.csv (HAR, primary) in windows, subjects and signed error.
Backbone names as in the earlier figure data (PI 7.07u): FactorizePhys (the analysis tables' 'FactorizePhys (matched)'),
PhysNet, RhythmFormer; test_set VIPL-HR.
Writes outputs/707/fig4_vipl_binned_signed_error.csv with columns test_set, backbone, arm, reference_HR_bin_bpm,
bin_center_bpm, windows, subjects, mean_signed_error_bpm. Usage (main environment): vipl_fig4.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W / 'eval5'))
V7 = W / 'cache' / 'vipl707'
ARMS = ('(0)', '(ii-pc)', '(i-simple)')


def main():
    import stage3_check as SC
    SC.init(str(W / 'vendor' / 'st8_6ebbc9cd'))
    import stage3_analyze as S3
    import vipl_c
    sa = V7 / 'stage_a' / 'VIPL'
    refs = pd.read_csv(sa / 'references.csv', dtype={'subject_id': str})
    choice = json.loads((sa / 'bvp_rate.json').read_text())['choice']
    pops = vipl_c.populations(refs, choice)
    rd = pd.read_csv(V7 / 'stage_c2' / 'VIPL' / 'readouts.csv.gz')
    keys = pd.concat([pd.read_csv(V7 / f / 'VIPL' / 'checkpoints.csv') for f in ('stage_b_confirmatory', 'stage_b_exploratory')], ignore_index=True)
    kk = keys[['key', 'backbone', 'arm', 'seed', 'variant', 'set', 'view']].rename(columns={'set': 'key_set'})
    bad = set(pd.read_csv(V7 / 'stage_c2' / 'VIPL' / 'defects.csv').key.dropna())
    sup = pd.read_csv(V7 / 'stage_c2' / 'VIPL' / 'support.csv')
    support = set(zip(sup.window_id, sup.layer))
    d = rd.merge(kk, on='key').merge(pops, on=['window_id', 'layer'])
    d = d[~d.key.isin(bad)].copy()
    d['in_support'] = [(a, b) in support for a, b in zip(d.window_id, d.layer)]
    d['dataset'] = 'VIPL-HR'
    d['backbone'] = d.backbone.map(S3.NAME)
    d['strategy'] = d.arm
    d['run_id'] = d.key
    d['available'] = d.valid
    z = d[d.variant.isin(S3.PRIMARY) & d.seed.isin(vipl_c.CONF_SEEDS) & d.arm.isin(ARMS) & d.layer.eq('HAR') & d.population.eq('primary')]
    assert z.key.nunique() == 27, z.key.nunique()
    m4 = S3.AN.fam_m4(S3.restrict(z), [])
    assert (m4.seeds == 3).all()
    out = pd.DataFrame(dict(test_set='VIPL-HR', backbone=m4.backbone.replace({'FactorizePhys (matched)': 'FactorizePhys'}), arm=m4.strategy, reference_HR_bin_bpm=m4.bin,
                            bin_center_bpm=[int(b.split('-')[0]) + 5 for b in m4.bin], windows=m4.windows, subjects=m4.subjects,
                            mean_signed_error_bpm=m4.signed_error))
    out = out.sort_values(['backbone', 'arm', 'bin_center_bpm']).reset_index(drop=True)
    # check against the stored (c1) M4 bins for the confirmatory arms
    c1 = pd.read_csv(V7 / 'stage_c1' / 'VIPL' / 'm4_bins.csv')
    c1 = c1[c1.layer.eq('HAR') & c1.population.eq('primary') & c1.strategy.isin(['(0)', '(ii-pc)'])].copy()
    c1['backbone'] = c1.backbone.replace({'FactorizePhys (matched)': 'FactorizePhys'})
    m = out.merge(c1, left_on=['backbone', 'arm', 'reference_HR_bin_bpm'], right_on=['backbone', 'strategy', 'bin'])
    check = dict(c1_rows=len(c1), matched=len(m), windows_equal=bool((m.windows_x == m.windows_y).all()),
                 subjects_equal=bool((m.subjects_x == m.subjects_y).all()),
                 max_abs_difference=float(np.abs(m.mean_signed_error_bpm - m.signed_error).max()))
    p = W / 'outputs' / '707' / 'fig4_vipl_binned_signed_error.csv'
    out.to_csv(p, index=False)
    print(json.dumps(dict(rows=len(out), arms=sorted(out.arm.unique()), bins=sorted(out.reference_HR_bin_bpm.unique(), key=lambda b: int(b.split('-')[0])),
                          check_against_c1=check), indent=1))


if __name__ == '__main__':
    main()
