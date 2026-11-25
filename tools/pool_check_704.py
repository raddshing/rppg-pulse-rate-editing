"""Brief 02 7.04b 나 (A6 item 4): check a regenerated October FactorizePhys pool against the October manifest.

Reads the manifest written by train/gen_pool.py (arm ii) or train/gen_i_pc.py (arm i, shift 'original'). Each row
carries the SHA-256 of the input and label tensors as the training loader reloads them from the stored uint8 files
(u8_store.model_input), and whether each equals the SHA-256 of the October tensor in the October manifest.
Passes only if every item of the October manifest (generation_excluded False) is present once and both SHA-256 are
equal for every item. On a pass it writes cache/train704/pool_check_FPm_<arm>.ok (content: the manifest path, used
by the training runner); on a failure it writes nothing there and exits 1.
Writes outputs/704/trainer_check/pool_check_FPm_<arm>.csv (aggregate). Usage: pool_check_704.py ARM (i, ii or i-pc)
"""
import csv
import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
V = WORK / 'vendor' / 'st8_6ebbc9cd'
CFG = json.loads((Path(__file__).resolve().parents[1] / 'config' / 'paths.json').read_text())  # 7.08: machine paths, see config/paths.example.json
MAIN_OUT = Path(CFG['october_project']) / 'outputs'  # read only
CONF = {'ii': (WORK / 'cache/pools/FPm_original_all_manifest.csv', V / 'outputs/st30g_FPm_ii_manifest.csv',
               'input_sha_equals_original', 'label_sha_equals_original'),
        'i': (WORK / 'cache/pools_i/FPm_i_original_all_manifest.csv', MAIN_OUT / 'st30t_FPm_A_PURE_i_pool.csv',
              'input_sha_equals_i', 'label_sha_equals_i'),
        # (i-pc) (shift d): the reference is the manifest of the confirmatory (i-pc) pool (its tensor SHA-256 as reloaded),
        # since (i-pc) items are not (i) items; regenerated into cache/pools_i_704 so that manifest stays in place
        'i-pc': (WORK / 'cache/pools_i_704/FPm_i_d_all_manifest.csv', WORK / 'cache/pools_i/FPm_i_d_all_manifest.csv', None, None)}


def rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def main():
    arm = sys.argv[1]
    new_p, oct_p, xi, yi = CONF[arm]
    new = rows(new_p)
    october = [r for r in rows(oct_p) if str(r.get('generation_excluded', 'False')) != 'True']
    key = lambda r: (r['chunk_id'], r['augmentation_index'])  # noqa: E731
    nk = [key(r) for r in new]
    ok_keys = set(nk) == {key(r) for r in october} and len(nk) == len(set(nk)) == len(october)
    by = {key(r): r for r in october}
    # independent recheck: the SHA-256 recorded by the generator against the October manifest
    ref_x, ref_y = ('input_sha256', 'label_sha256') if xi else ('ops2_input_sha256', 'ops2_label_sha256')
    x_eq = sum(r['ops2_input_sha256'] == by[key(r)][ref_x] for r in new if key(r) in by)
    y_eq = sum(r['ops2_label_sha256'] == by[key(r)][ref_y] for r in new if key(r) in by)
    flag_x = sum(r[xi] == 'True' for r in new) if xi else x_eq  # (i-pc): no generator flag against this reference
    flag_y = sum(r[yi] == 'True' for r in new) if yi else y_eq
    passed = ok_keys and x_eq == y_eq == flag_x == flag_y == len(october)
    out = dict(backbone='FactorizePhys-matched', arm=f'({arm})', items_october=len(october), items_regenerated=len(new),
               same_item_set=ok_keys, input_sha256_equal=x_eq, label_sha256_equal=y_eq,
               generator_flag_input_equal=flag_x, generator_flag_label_equal=flag_y,
               stored_bytes=sum(int(r.get('ops2_view_bytes') or 0) + int(r.get('ops2_rawlabel_bytes') or 0) + int(r.get('ops2_stats_bytes') or 0) for r in new),
               passed=passed, source='7.04 regenerated pool vs October manifest')
    d = WORK / 'outputs' / '704' / 'trainer_check'
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f'pool_check_FPm_{arm}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out))
        w.writeheader()
        w.writerow(out)
    print(json.dumps(out))
    if passed:
        (WORK / 'cache' / 'train704' / f'pool_check_FPm_{arm}.ok').write_text(str(new_p))
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
