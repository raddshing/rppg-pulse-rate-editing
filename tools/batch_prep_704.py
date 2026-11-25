"""Brief 02 7.04 다 (before training a batch): label SHA-256 list and configuration diff of one post hoc batch.

For every arm of the batch, against its reference run (the run of the same backbone and arm with the negative Pearson
loss and the same labels: confirmatory arms for S, M1 and M3; PhysNet-Std (batch S and rule 5) for M2):
  - label list (server only, per item): the SHA-256 of the float32 label that the loss receives in the new run and in the
    reference run, loaded by the same code path as train/u8_store.U8Tensors.__getitem__ (stored label, uint8 pool item via
    u8_store.item_input, then the std_cumsum transform for PhysNet-Std). Item order and pool merge as train/train_pc.main.
    A reference pool whose files are no longer on the server (the confirmatory (ii-pc) pools; the October FactorizePhys (i)
    and (ii) pools) is compared through the label SHA-256 recorded in its manifest (generation-time reload of the same
    tensors); the regenerated or fetched pool used instead has passed the item-by-item tensor check.
  - configuration (train/train_704.settings for the new run and for its reference run): every key, and the keys that
    differ. Allowed: the key that the batch changes, and, where the pool file set differs (pc_base for (ii-pc); the
    regenerated FactorizePhys (i), (ii), (i-pc) pools), pool_manifest.
Aggregates (outputs/704/batch_prep/<BATCH>_*.csv). Usage (repro environment): batch_prep_704.py BATCH (S, M1, M2, M3, P_PhysNet, P_FPm, P_RhythmFormer, R) [ARMS PART]
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'train'))
import train_704  # noqa: E402  (imports train_pc: working directory = vendor, as in training)
import u8_store  # noqa: E402
import numpy as np  # noqa: E402

OUT = WORK / 'outputs' / '704' / 'batch_prep'
SRV = WORK / 'cache' / 'train704' / 'batch_prep_server_only'
C = WORK / 'cache'
PHYS = {  # arm: (OPS2_704_ARM, OPS2_ARM, new pool manifest, reference pool manifest, reference labels from)
    '(0)': ('0', 'ii', 'ORIGINAL', 'ORIGINAL', 'files'),
    '(i)': ('', 'i', 'ORIGINAL', 'ORIGINAL', 'files'),
    '(i-pc)': ('', 'i', str(C / 'pools_i/PhysNet_i_d_all_manifest.csv'), str(C / 'pools_i/PhysNet_i_d_all_manifest.csv'), 'files'),
    '(i-simple)': ('', 'i', str(C / 'pools_i/PhysNet_i_zero_all_manifest.csv'), str(C / 'pools_i/PhysNet_i_zero_all_manifest.csv'), 'files'),
    '(ii)': ('', 'ii', 'ORIGINAL', 'ORIGINAL', 'files'),
    '(ii-lag)': ('', 'ii', str(C / 'pools/PhysNet_lag_shift_all_manifest.csv'), str(C / 'pools/PhysNet_lag_shift_all_manifest.csv'), 'files'),
    '(ii-pc)': ('', 'ii', str(C / 'pools_lab/PhysNet_pc_base_train_manifest.csv'), str(C / 'pools/PhysNet_polarity_explicit_all_manifest.csv'), 'manifest'),
}
PHYS_STD_REF = dict(PHYS, **{  # PhysNet-Std reference runs: batch S and rule 5 ((ii-pc)-Std used pc_base)
    '(ii-pc)': ('', 'ii', str(C / 'pools_lab/PhysNet_pc_base_train_manifest.csv'), str(C / 'pools_lab/PhysNet_pc_base_train_manifest.csv'), 'files')})
FPM = {
    '(0)': ('0', 'ii', 'ORIGINAL', 'ORIGINAL', 'files'),
    '(i)': ('', 'i', str(C / 'pools_i/FPm_i_original_all_manifest_abs.csv'), 'ORIGINAL', 'manifest'),
    '(i-pc)': ('', 'i', str(C / 'pools_i_704/FPm_i_d_all_manifest.csv'), str(C / 'pools_i/FPm_i_d_all_manifest.csv'), 'manifest'),
    '(i-simple)': ('', 'i', str(C / 'pools_i/FPm_i_zero_all_manifest.csv'), str(C / 'pools_i/FPm_i_zero_all_manifest.csv'), 'files'),
    '(ii)': ('', 'ii', str(C / 'pools/FPm_original_all_manifest_abs.csv'), 'ORIGINAL', 'manifest'),
    '(ii-lag)': ('', 'ii', str(C / 'pools/FPm_lag_shift_all_manifest.csv'), str(C / 'pools/FPm_lag_shift_all_manifest.csv'), 'files'),
    '(ii-pc)': ('', 'ii', str(C / 'pools_lab/FPm_pc_base_all_manifest.csv'), str(C / 'pools/FPm_polarity_explicit_all_manifest.csv'), 'manifest'),
}
ARMS4 = ['(0)', '(i)', '(i-pc)', '(i-simple)']
ARMS7 = ARMS4 + ['(ii)', '(ii-lag)', '(ii-pc)']
STD = {'OPS2_LABEL_REPR': 'std_cumsum'}
MCC = {'OPS2_704_LOSS': 'mcc'}
BATCH = {  # batch: (backbone, arm table, arms, new environment, reference environment, changed key)
    'S': ('PhysNet', PHYS, ARMS4, STD, {}, 'label_representation'),
    'M1': ('PhysNet', PHYS, ARMS7, MCC, {}, 'loss'),
    'M2': ('PhysNet', PHYS_STD_REF, ARMS7, dict(STD, **MCC), STD, 'loss'),
    'M3': ('FPm', FPM, ARMS7, MCC, {}, 'loss'),
}
# 7.07 다 / A6 item 3 (P): the second pool against the confirmatory run of the same arm (settings with the first seed for
# both; the seeds differ by design). Every key but pool_manifest must be equal; the originals' labels must be equal and
# every pool item's label differs (new target HR; the loader's label of each second-pool item equals its manifest SHA,
# checked by tools/pool_tensor_check_704.py). Reference pool labels from the confirmatory manifests.
ARMS6 = ARMS4 + ['(ii)', '(ii-pc)']
P2 = C / 'pools_p2'


def p_table(b, table):
    return {a: table[a] if a == '(0)' else ('', table[a][1], str(P2 / f'{b}_p2_{a.strip("()")}_manifest.csv'), table[a][3], 'manifest')
            for a in ARMS6}


BATCH['P_PhysNet'] = ('PhysNet', p_table('PhysNet', PHYS), ARMS6, {}, {}, 'pool_manifest')
BATCH['P_FPm'] = ('FPm', p_table('FPm', FPM), ARMS6, {}, {}, 'pool_manifest')
# A6 item 9 / PI 7.07i item 4 (conditional runs, 11-24): RhythmFormer with the MCC time term (train/train_704r.py) against
# the confirmatory RhythmFormer runs (archived loss), seven arms, the pools of the confirmatory runs (their result.json
# pool_manifest). Every key but loss must be equal and every label equal (same files).
RF = {
    '(0)': ('0', 'ii', 'ORIGINAL', 'ORIGINAL', 'files'),
    '(i)': ('', 'i', 'ORIGINAL', 'ORIGINAL', 'files'),
    '(i-pc)': ('', 'i', str(C / 'pools_i/RhythmFormer_i_d_all_manifest.csv'), str(C / 'pools_i/RhythmFormer_i_d_all_manifest.csv'), 'files'),
    '(i-simple)': ('', 'i', str(C / 'pools_i/RhythmFormer_i_zero_all_manifest.csv'), str(C / 'pools_i/RhythmFormer_i_zero_all_manifest.csv'), 'files'),
    '(ii)': ('', 'ii', 'ORIGINAL', 'ORIGINAL', 'files'),
    '(ii-lag)': ('', 'ii', str(C / 'pools/RhythmFormer_lag_shift_all_manifest.csv'), str(C / 'pools/RhythmFormer_lag_shift_all_manifest.csv'), 'files'),
    '(ii-pc)': ('', 'ii', str(C / 'pools/RhythmFormer_polarity_explicit_all_manifest.csv'), str(C / 'pools/RhythmFormer_polarity_explicit_all_manifest.csv'), 'files'),
}
BATCH['R'] = ('RhythmFormer', RF, ARMS7, {'OPS2_704R': '1'}, {}, 'loss')
# R part 2 (PI 7.07k item 3): the confirmatory (ii-lag) and (ii-pc) pools rebuilt by train/gen_pool.py into
# cache/pools_r704 (tensor SHA of every item equal to the confirmatory manifest, tools/pool_tensor_check_704.py); labels
# against the SHA recorded in the confirmatory manifests (their files are no longer on the server). Allowed: loss and
# pool_manifest (rebuilt pool file set).
RF2 = {
    '(ii-lag)': ('', 'ii', str(C / 'pools_r704/RhythmFormer_lag_shift_all_manifest.csv'), str(C / 'pools/RhythmFormer_lag_shift_all_manifest.csv'), 'manifest'),
    '(ii-pc)': ('', 'ii', str(C / 'pools_r704/RhythmFormer_polarity_explicit_all_manifest.csv'), str(C / 'pools/RhythmFormer_polarity_explicit_all_manifest.csv'), 'manifest'),
}
BATCH['R2'] = ('RhythmFormer', RF2, ['(ii-lag)', '(ii-pc)'], {'OPS2_704R': '1'}, {}, 'loss')
# 7.07 / A6 item 3 (P, RhythmFormer): second pools against the confirmatory RhythmFormer runs, as P_PhysNet and P_FPm.
BATCH['P_RhythmFormer'] = ('RhythmFormer', p_table('RhythmFormer', RF), ARMS6, {}, {}, 'pool_manifest')


def rows(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))


def std(y, repr_):
    if repr_ == 'std_cumsum':  # train/u8_store.py U8Tensors.__getitem__ (PhysNet only)
        c = np.cumsum(y.astype(np.float64))
        y = np.ascontiguousarray((c - c.mean()) / c.std(), dtype=np.float32)
    return y


def label(backbone, kind, r, repr_):
    """SHA-256 (.npy bytes, u8_store.npy_sha) of the float32 label as the loader gives it, after the std transform."""
    if kind == 1 and r.get('ops2_view_path'):
        _, y = u8_store.item_input(backbone, r)
    elif kind == 1 and r.get('ops2_label_path'):
        y = np.load(r['ops2_label_path'], allow_pickle=False)
    else:
        y = np.load(r['label_path'], allow_pickle=False)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if backbone == 'PhysNet':
        y = std(y, repr_)
    return u8_store.npy_sha(y)  # SHA-256 of the .npy bytes, as the manifests record tensor SHA-256


def items(backbone, manifest):
    """(kind, index, row) in the order of train_pc.main: originals, then the pool with the manifest merged in."""
    config, original, pool, key, init_file, a = train_704.setup(backbone)
    if manifest != 'ORIGINAL':
        new = {(r['chunk_id'], r['augmentation_index']): r for r in rows(manifest)}
        assert len(new) == len(pool)
        ks = ('ops2_view_path', 'ops2_rawlabel_path', 'ops2_stats_path', 'ops2_label_path', 'ops2_view_offset')
        pool = [dict(r, **{k: new[(r['chunk_id'], r['augmentation_index'])].get(k, '') for k in ks}) for r in pool]
    return [(0, i, r) for i, r in enumerate(original)] + [(1, i, r) for i, r in enumerate(pool)], a


def setenv(env, ops2_arm, arm0):
    for k in ('OPS2_LABEL_REPR', 'OPS2_ARM', 'OPS2_704_LOSS', 'OPS2_704R'):
        os.environ.pop(k, None)
    os.environ.update(dict(env, OPS2_ARM=ops2_arm))
    train_704.ARM = arm0
    train_704.LOSS = env.get('OPS2_704_LOSS', 'pearson')


def main():
    name = sys.argv[1]
    backbone, table, arms, env_new, env_ref, changed = BATCH[name]
    if len(sys.argv) > 2:  # a subset of arms (pools prepared at different times); outputs named <BATCH>_<part>_*
        arms = [a for a in arms if a in sys.argv[2].split(',')]
        name_out = f'{name}_{sys.argv[3]}'
    else:
        name_out = name
    OUT.mkdir(parents=True, exist_ok=True)
    SRV.mkdir(parents=True, exist_ok=True)
    summary, diff, full = [], [], []
    for arm in arms:
        arm0, ops2_arm, man_new, man_ref, ref_from = table[arm]
        conf = {}
        for tag, env, man in (('reference', env_ref, man_ref), ('new', env_new, man_new)):
            setenv(env, ops2_arm, arm0)
            fn = train_704.settings
            if env.get('OPS2_704R'):  # batch R: the settings of train/train_704r.py (MCC time term)
                import train_704r
                fn = train_704r.settings_mcc
            conf[tag] = fn(backbone, 20261131, man, Path(f'{name}_{arm}_{tag}'))
            conf[tag].pop('out_dir')
        for k in sorted(set(conf['new']) | set(conf['reference'])):
            a, b = conf['reference'].get(k), conf['new'].get(k)
            full.append(dict(batch=name, arm=arm, key=k, reference=a, new=b, differs=a != b))
            if a != b:
                diff.append(dict(batch=name, arm=arm, key=k, reference=a, new=b,
                                 allowed=k == changed or (k == 'pool_manifest' and man_new != man_ref)))
        setenv(env_new, ops2_arm, arm0)
        its, stream_arm = items(backbone, man_new)
        ref_rows, ref_man = {}, {}
        if ref_from == 'manifest':  # reference pool files not on the server: label SHA recorded in its manifest
            setenv(env_ref, ops2_arm, arm0)
            ref_its, _ = items(backbone, man_ref)
            ref_rows = {(k, i): r for k, i, r in ref_its}
            setenv(env_new, ops2_arm, arm0)
            # 7.04 fix (2026-11-23): the SHA is read from the row of the reference manifest itself. The merged row of
            # items() keeps only the file-path keys of the manifest, so its SHA columns were those of the stream's base
            # pool ((i) for (i-pc)); the first M3 run of this tool stopped on that (outputs moved aside, *_attempt1).
            ref_man = {} if man_ref == 'ORIGINAL' else {(q['chunk_id'], q['augmentation_index']): q for q in rows(man_ref)}
        lst, h = [], hashlib.sha256()
        for kind, i, r in its:
            new = label(backbone, kind, r, env_new.get('OPS2_LABEL_REPR', ''))
            if kind == 1 and ref_from == 'manifest':
                q = ref_rows[(kind, i)]
                assert (q['chunk_id'], q['augmentation_index']) == (r['chunk_id'], r['augmentation_index'])
                if ref_man:  # 'ORIGINAL' reference ((i), (ii)): the October pool rows of the setup, as before
                    q = ref_man[(r['chunk_id'], r['augmentation_index'])]
                ref = q.get('ops2_label_sha256') or q.get('label_sha256')
                assert not env_ref.get('OPS2_LABEL_REPR'), 'manifest SHA is of the loader label (no std transform)'
            else:
                ref = label(backbone, kind, r, env_ref.get('OPS2_LABEL_REPR', ''))
            lst.append(dict(arm=arm, kind='original' if kind == 0 else 'pool', index=i, chunk_id=r.get('chunk_id', ''),
                            augmentation_index=r.get('augmentation_index', ''), label_used_sha256_new=new,
                            label_used_sha256_reference=ref, reference_from='manifest SHA' if kind == 1 and ref_from == 'manifest' else 'files'))
            h.update(f'{kind},{i},{new}\n'.encode())
        with open(SRV / f'{name}_{arm.strip("()")}_labels.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(lst[0]))
            w.writeheader()
            w.writerows(lst)
        summary.append(dict(batch=name, backbone=backbone, arm=arm, stream_arm=stream_arm, items=len(lst),
                            originals=sum(x['kind'] == 'original' for x in lst), pool_items=sum(x['kind'] == 'pool' for x in lst),
                            labels_equal_reference=sum(x['label_used_sha256_new'] == x['label_used_sha256_reference'] for x in lst),
                            labels_differ_reference=sum(x['label_used_sha256_new'] != x['label_used_sha256_reference'] for x in lst),
                            reference_pool_labels_from='manifest SHA' if ref_from == 'manifest' else 'files',
                            distinct_label_sha256=len({x['label_used_sha256_new'] for x in lst}),
                            ordered_list_sha256=h.hexdigest(), source='7.04 new training (labels as loaded)'))
        print(json.dumps(summary[-1]), flush=True)
    for fn, rr in (('label_sha_summary', summary), ('config_diff', diff), ('config_full', full)):
        if rr:
            with open(OUT / f'{name_out}_{fn}.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rr[0]))
                w.writeheader()
                w.writerows(rr)
    print(json.dumps(diff, default=str))


if __name__ == '__main__':
    main()
