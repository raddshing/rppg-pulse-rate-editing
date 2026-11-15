"""Brief 02 6.68 나: two hash lists ('SHA-256  relative name', e.g. stage1_hashes.txt of two runs) compared name by name:
identical, differs, only in the first, only in the second. Writes OUT.csv; exit 0 if every name of the first list is
present and identical in the second (names only in the second are new outputs of the later code and are listed).
Usage: compare_hash_lists.py FIRST.txt SECOND.txt OUT.csv
"""
import sys

import pandas as pd


def read(p):
    out = {}
    for line in open(p):
        h, n = line.rstrip('\n').split('  ', 1)
        out[n] = h
    return out


def main():
    a, b = read(sys.argv[1]), read(sys.argv[2])
    rows = [dict(name=n, result=('identical' if a[n] == b.get(n) else 'differs') if n in b else 'only in the first')
            for n in sorted(a)]
    rows += [dict(name=n, result='only in the second') for n in sorted(set(b) - set(a))]
    r = pd.DataFrame(rows)
    r.to_csv(sys.argv[3], index=False)
    print(r.result.value_counts().to_dict())
    sys.exit(0 if not r.result.isin(['differs', 'only in the first']).any() else 3)


if __name__ == '__main__':
    main()
