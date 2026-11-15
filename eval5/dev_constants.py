"""Constant predictors (ACCESS_OPS2_BRIEF_02 6.46 가 7, 6.48 라): the reference-HR mean and median of the development part of the
same dataset, wide band (HAR): MMPD-A automatic Q* windows (723; the B rule) and MCD-rPPG development-half draft-rule
windows (high and low), both from the stage-1 outputs of the development data (the rehearsal's stage 1).
Usage: dev_constants.py STAGE1_DEV_DIR OUT_DIR  -> OUT_DIR/constants_MMPD.json, constants_MCD.json
"""
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    s1, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(s1 / 'MMPD' / 'populations.csv')
    q = p[p.layer.eq('HAR') & p.in_Qstar.astype(bool)].reference_HR.astype(float)
    json.dump(dict(dataset='MMPD-A', population='automatic Q* (B rule)', windows=int(len(q)), mean=float(q.mean()), median=float(q.median())),
              open(out / 'constants_MMPD.json', 'w'), indent=1)
    m = pd.read_csv(s1 / 'MCD' / 'populations.csv')
    r = m[m.population.eq('draft_rule') & m.layer.eq('HAR')].reference_HR.astype(float)
    json.dump(dict(dataset='MCD-rPPG dev half', population='draft rule (high and low)', windows=int(len(r)), mean=float(r.mean()), median=float(r.median())),
              open(out / 'constants_MCD.json', 'w'), indent=1)
    print(open(out / 'constants_MMPD.json').read(), open(out / 'constants_MCD.json').read())


if __name__ == '__main__':
    main()
