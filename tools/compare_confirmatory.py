"""Brief 02 6.48 마 1 / 6.68 나 (support test): two confirmatory.csv tables of eval5/stage3_analyze.py (all checkpoints, and one
exploratory checkpoint left out) compared cell by cell on point, ci_lo, ci_hi and grade per (analysis, backbone, name,
contrast, metric). Writes OUT.csv (keys, largest absolute difference, grades); exit 0 if all values and grades are equal.
Usage: compare_confirmatory.py FIRST.csv SECOND.csv OUT.csv
"""
import sys

import numpy as np
import pandas as pd

KEY = ['analysis', 'backbone', 'name', 'contrast', 'metric']


def main():
    a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
    m = a.merge(b, on=KEY, how='outer', suffixes=('', '_t'), indicator=True)
    d = np.nanmax(np.abs(m[['point', 'ci_lo', 'ci_hi']].to_numpy(float) - m[['point_t', 'ci_lo_t', 'ci_hi_t']].to_numpy(float)), axis=1)
    m['max_abs_difference'] = d
    m['grade_equal'] = m.grade.fillna('') == m.grade_t.fillna('')
    m[KEY + ['_merge', 'max_abs_difference', 'grade', 'grade_t', 'grade_equal']].to_csv(sys.argv[3], index=False)
    ok = m._merge.eq('both').all() and (m.max_abs_difference == 0).all() and m.grade_equal.all()
    print('cells', len(m) * 3, 'rows only in one table', int(m._merge.ne('both').sum()), 'max abs difference', float(np.nanmax(d)),
          'grade changes', int((~m.grade_equal).sum()))
    sys.exit(0 if ok else 3)


if __name__ == '__main__':
    main()
