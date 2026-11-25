"""Brief 02 7.04e 가 1: tensor SHA-256 check of a pool fetched back from Drive (or regenerated), item by item.

Every row of the training manifest that names an OPS2 pool file (ops2_view_path) is loaded exactly as the training loader
loads it (train/u8_store.item_input: stored uint8 view, raw label, stats, context offset and chunk length), and the
SHA-256 of the resulting input and label tensors (u8_store.npy_sha) is compared with ops2_input_sha256 and
ops2_label_sha256 of the manifest (recorded when the pool was generated). With REFERENCE_MANIFEST, the same SHA-256 are
also compared with that manifest's rows of the same (chunk, augmentation index), e.g. the confirmatory (ii-pc) pool for
pc_base. Passes only if every item is present and every SHA-256 is equal. On a pass writes
cache/train704/pool_tensor_check_<NAME>.ok (content: the manifest path); exits 1 otherwise.
Writes outputs/704/pool_checks/<NAME>.csv (aggregate). Usage (repro environment):
pool_tensor_check_704.py NAME BACKBONE(PhysNet|FPm) MANIFEST [REFERENCE_MANIFEST]
"""
import csv
import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'train'))
import u8_store  # noqa: E402


def rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def main():
    name, backbone, manifest = sys.argv[1], sys.argv[2], sys.argv[3]
    ref = rows(sys.argv[4]) if len(sys.argv) > 4 else None
    rr = [r for r in rows(manifest) if r.get('ops2_view_path')]
    key = lambda r: (r['chunk_id'], r['augmentation_index'])  # noqa: E731
    by = {key(r): r for r in ref} if ref else {}
    n = missing = x_eq = y_eq = rx_eq = ry_eq = 0
    for r in rr:
        n += 1
        if not Path(r['ops2_view_path']).is_file() or not Path(r['ops2_rawlabel_path']).is_file():
            missing += 1
            continue
        x, y = u8_store.item_input(backbone, r)
        xs, ys = u8_store.npy_sha(x), u8_store.npy_sha(y)
        x_eq += xs == r['ops2_input_sha256']
        y_eq += ys == r['ops2_label_sha256']
        if ref:
            q = by.get(key(r))
            rx_eq += bool(q) and xs == q['ops2_input_sha256']
            ry_eq += bool(q) and ys == q['ops2_label_sha256']
    passed = n > 0 and missing == 0 and x_eq == y_eq == n and (not ref or (len(by) == n and rx_eq == ry_eq == n))
    out = dict(pool=name, backbone=backbone, items=n, missing_files=missing, input_sha256_equal_manifest=x_eq,
               label_sha256_equal_manifest=y_eq, reference_items=len(by) if ref else '',
               input_sha256_equal_reference=rx_eq if ref else '', label_sha256_equal_reference=ry_eq if ref else '',
               passed=passed, source='7.04 pool check (tensors as loaded for training)')
    d = WORK / 'outputs' / '704' / 'pool_checks'
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f'{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out))
        w.writeheader()
        w.writerow(out)
    print(json.dumps(out))
    if passed:
        (WORK / 'cache' / 'train704' / f'pool_tensor_check_{name}.ok').write_text(str(manifest))
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
