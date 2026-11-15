"""S4-1a contact-only audit and the single private PI review assignment."""
from src.st4.base import *
from src.st4.signals import common_hr, reference_estimators, classify
import numpy as np
import pandas as pd
from scipy.io import loadmat


KEY = ROOT / 'outputs/st4_private/st4_ref_blind_key.csv'


def inputs():
    for p in ('data/splits/mmpd_split_st1.json', 'data/index/mmpd_index_st1.csv',
              'outputs/st2_window_categories.csv', 'outputs/st3_gt_defect_map.csv',
              'outputs/st3_c3_targets.csv', 'outputs/st3_c3_clips.csv', 'data/index/pure_index_st1.csv'):
        source_hash(p)
    split = json.loads((ROOT / 'data/splits/mmpd_split_st1.json').read_text())
    subjects = set(split['A'])
    index = {r['recording_id']: r for r in rows('data/index/mmpd_index_st1.csv') if r['subject_id'] in subjects}
    cats = pd.read_csv(ROOT / 'outputs/st2_window_categories.csv')
    p = cats[cats.in_P & cats.subject_id.isin(subjects)].sort_values('window_id').reset_index(drop=True)
    check('P has220 A recordings and1100 unique fixed windows', len(index) == 220 and len(p) == 1100 and p.window_id.is_unique
          and set(p.window_id) == {f'MMPD|{rid}|{j}' for rid in index for j in range(5)})
    defects = pd.read_csv(ROOT / 'outputs/st3_gt_defect_map.csv').set_index('window_id').reindex(p.window_id)
    p['defect_binary'] = defects[['d1', 'd2', 'd3']].eq(1).any(axis=1).astype(int).to_numpy()
    check('Corrected binary defect count58', p.defect_binary.sum() == 58)
    for name in ('d1', 'd2', 'd3'):
        p[name] = defects[name].to_numpy()
    targets = pd.read_csv(ROOT / 'outputs/st3_c3_targets.csv')
    h = targets[targets.role.eq('c1R')]
    f = targets[targets.role.eq('c2R')]
    clips = pd.read_csv(ROOT / 'outputs/st3_c3_clips.csv')
    eligible = clips[clips.variant.eq('identity') & clips.window_id.isin(h.window_id) & clips.excluded.eq(0)]
    check('c1R163 with10 subjects37 recordings and162 fixed boxes', len(h) == 163 and h.subject_id.nunique() == 10
          and h.recording_id.nunique() == 37 and len(eligible) == 162
          and set(h.window_id) - set(eligible.window_id) == {'MMPD|subject7/p7_8|1'})
    check('UB5F53 and layer membership40/44', len(f) == 53 and f.c2_HAR.sum() == 40 and f.c2_L0.sum() == 44)
    pure = [r for r in rows('data/index/pure_index_st1.csv') if r['recording_id'].endswith('-01')]
    check('PURE task01 ten recordings60 windows', len(pure) == 10 and sum(int(r['frame_count']) // 300 for r in pure) == 60)
    # Read only frozen memberships here, never predictions or model-specific labels.
    source_hash('outputs/st2_window_run_table.parquet')
    membership = pd.read_parquet(ROOT / 'outputs/st2_window_run_table.parquet', columns=['window_id', 'layer', 'run_id', 'in_P', 'fail_common'])
    common = membership[membership.in_P & membership.run_id.isin(RW15)].drop_duplicates(['window_id', 'layer'])
    check('Fixed common failures76/79', {layer: int(((common.layer == layer) & (common.fail_common == 1)).sum())
                                       for layer in BANDS} == {'HAR': 76, 'L0': 79})
    save_json(REPORTS / 'st4_initial_population_checks.json', dict(status='passed', P=1100, defect=58, c1R=163,
               c1R_fixed_box=162, F=53, common_HAR=76, common_L0=79, PURE_task01=60, RW15=len(RW15), RG6=len(RG6), r2=13))
    return p, index


def raw_gt(row, allowed):
    check('MMPD-A selected before raw contact read', row['subject_id'] in allowed)
    path = Path(row['label_path'])
    check('Declared original MMPD path', Path(PATHS['raw']['MMPD']).resolve() in path.resolve().parents)
    raw = loadmat(path, variable_names=['GT_ppg'])['GT_ppg'].T.reshape(-1)
    check('Raw30Hz contact frame alignment without resampling', len(raw) == int(row['frame_count']) and float(row['processing_fs']) == 30)
    return raw


def choose(audit):
    rng = np.random.default_rng(20261111)
    pool = audit[audit.in_review_pool].sort_values('window_id').window_id.tolist()
    selected = pool if len(pool) <= 60 else [pool[int(i)] for i in rng.choice(len(pool), 60, replace=False)]
    chosen = [dict(window_id=w, sampling_group='pool') for w in selected]
    for high in (True, False):
        candidates = audit[audit.defect_binary.eq(0) & audit.ref_class.eq('agreement') &
                           (audit.h_FFT_HAR.ge(90) if high else audit.h_FFT_HAR.lt(90))].sort_values('window_id').window_id.tolist()
        check('At least five prescribed agreement controls in each stratum', len(candidates) >= 5)
        chosen += [dict(window_id=candidates[int(i)], sampling_group='control_high' if high else 'control_low')
                   for i in rng.choice(len(candidates), 5, replace=False)]
    ordered = [chosen[int(i)] for i in rng.permutation(len(chosen))]
    return [dict(item_id=f'R{i:02d}', **row) for i, row in enumerate(ordered, 1)]


def audit():
    started = time.monotonic()
    p, index = inputs()
    allowed = {r['subject_id'] for r in index.values()}
    records = []
    for i, (rid, group) in enumerate(p.groupby('recording_id', sort=True), 1):
        raw = raw_gt(index[rid], allowed)
        for row in group.itertuples():
            values = raw[int(row.window_index) * 300:(int(row.window_index) + 1) * 300]
            h_fft = float(row.hr_label_bpm)
            estimates = reference_estimators(values, h_fft)
            cl = classify(h_fft, estimates['h_ACF'], estimates['h_IBI'])
            rc = {layer: common_hr(values, layer) for layer in BANDS}
            records.append(dict(window_id=row.window_id, recording_id=rid, subject_id=row.subject_id,
                window_index=int(row.window_index), population='P', ground_truth_path='raw_GT_ppg_no_resampling;HAR=frozen_hr_label_bpm;L0=common_Rc',
                h_FFT_HAR=h_fft, h_FFT_L0=rc['L0'], R_c_HAR=rc['HAR'], R_c_L0=rc['L0'], **estimates,
                d1=row.d1, d2=row.d2, d3=row.d3, defect_binary=row.defect_binary, T1=row.T1, T1b=row.T1b,
                flag_a_state=row.flag_a_state, flag_e_state=row.flag_e_state, motion=row.motion, exercise=row.exercise,
                ref_class=cl, in_review_pool=bool(row.defect_binary == 0 and h_fft >= 90 and cl != 'agreement'),
                agreement_control=False, selected_from_pool=False, PI_peak_count='', PI_unreadable='', HR_PI='',
                Qstar_HAR='', Qstar_L0='', Qstar_plus_unread='', Qstar_without_T1='', PI_reference_error_suspect=''))
        if i % 10 == 0 or i == len(index):
            progress('ref_audit', i, len(index), started)
            budget_check()
    table = pd.DataFrame(records).sort_values('window_id').reset_index(drop=True)
    check('Known missing original reference retained', table.loc[table.window_id.eq('MMPD|subject29/p29_3|4'), 'h_FFT_HAR'].isna().all())
    assignments = choose(table)
    KEY.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    KEY.parent.chmod(0o700)
    check('No prior Stage4 reference key is overwritten', not KEY.exists())
    with os.fdopen(os.open(KEY, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['item_id', 'window_id', 'sampling_group'])
        writer.writeheader()
        writer.writerows(assignments)
    for assignment in assignments:
        column = 'selected_from_pool' if assignment['sampling_group'] == 'pool' else 'agreement_control'
        table.loc[table.window_id.eq(assignment['window_id']), column] = True
    table.to_csv(ROOT / 'outputs/st4_ref_audit.csv', index=False)
    save_json(REPORTS / 'st4_ref_sample_locked.json', dict(status='locked_pending_prefix_append', n_review=len(assignments),
        key_path=str(KEY.relative_to(ROOT)), key_sha256=sha(KEY), audit_sha256=sha(ROOT / 'outputs/st4_ref_audit.csv'),
        assignment_seed=20261111, key_permissions='0600', parent_permissions='0700', new_PI_notice_received=False))
    return dict(reference_audit_complete=True, Qstar_finalized=False, review_count=len(assignments),
                next='Commit sole-key checksum and sample count before drawing review waveforms')


if __name__ == '__main__':
    finish('ref_audit', audit)
