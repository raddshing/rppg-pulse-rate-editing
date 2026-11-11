"""Prescribed E3R-a component lag and synthetic sign controls; no inference."""
from src.st9e.generator import *
from src.st9e.lag import lag_value
from src.st9e.model_view import ViewWorker
import resource


def signal(clip, face):
    return editor_band(green(weighted_rgb(clip, face)))


def sign_controls(pure, allowed):
    row = pure['01-01']
    gt = contact(row, 'PURE', allowed)
    start = 863
    original = common_hr(gt[start:start + 300], 'HAR')
    clip = np.full((300, 64, 64, 3), 128, np.uint8)
    mask = np.ones((300, 64, 64), np.float32)
    out = []
    for tau in [-10, 10]:
        g0, goal = ppg_vectors(gt, start, original, original, tau)
        key = int(hashlib.sha256(f'ST9E|sign|{tau}'.encode()).hexdigest()[:8], 16)
        value, diag = edited(clip, clip.astype(np.float32), mask, mask.astype(bool),
            g0, goal, [0, 1, 0], original, 'ED2W-S', .01, 1.,
            'sign', 'ii-h', rounding_rng=np.random.default_rng([0, key]))
        lag, corr = lag_value(signal(value, mask), editor_band(gt[start:start + 300]), list(range(-15, 16)))
        path = CACHE / f'consumed_sign_{tau}.npy'
        np.save(path, value)
        digest = sha(path)
        delete_temp(path)
        out.append(dict(tau=tau, L=lag, correlation=corr, passed=lag == tau,
            native_sha256=digest, original_HAR=original, ground_truth_path=row['label_path'], **diag))
    result = dict(cases=out, all_passed=all(r['passed'] for r in out), no_notch=True)
    save_json(REPORTS / 'st9e_sign_test.json', result)
    return result


def main():
    setting_path = REPORTS / 'st9e_component_settings.json'
    check('E3R-a actual settings committed', sha(setting_path) in (ROOT / 'reports/ST4_INSTR03_PREFIX.md').read_text())
    settings = json.loads(setting_path.read_text())
    for path, digest in settings['source_hashes'].items():
        check('Frozen E3 dependency matches actual supplement', source_hash(path) == digest)
    _, pure, allowed = indices()
    signs = sign_controls(pure, allowed)
    direction = next(r for r in rows('outputs/st4_editor_directions.csv') if r['dataset'] == 'PURE')
    v = np.array([float(direction['v_' + c]) for c in 'RGB'])
    out = []
    assets = []
    started = time.monotonic()
    worker = ViewWorker()
    try:
        for done, c in enumerate(settings['cases'], 1):
            rid = c['recording_id']
            row = pure[rid]
            files = frame_files(row)
            gt = contact(row, 'PURE', allowed)
            start = c['source_start']
            target = c['target_HR']
            tau = c['tau_rec']
            check('Frozen target rounding string', str(float(target)) == c['target_key'])
            clip = read_pure(files, start)
            face, roi, failure = face_masks(clip)
            first = read_pure(files, 0, 1)[0]
            physview, physbox = worker.view(clip, 'PhysNet', first)
            original = common_hr(gt[start:start + 300], 'HAR')
            notched = notch_all(clip, bands_for(original))
            values, label, diag = ed2w_context(clip, face, roi, gt, start, original,
                target, tau, v, physview, physbox[0], rid, 'ii-h', notched=notched)
            edit = values.pop()
            g0, goal = ppg_vectors(gt, start, original, target, tau)
            key = int(hashlib.sha256(f'ST7B|{rid}|ii-h|{float(target)}'.encode()).hexdigest()[:8], 16)
            twin, twin_diag = edited(clip, notched, face, roi, g0, goal, v, original,
                'ED2W-S', 0., diag['w'], rid, 'ii-h', rounding_rng=np.random.default_rng([0, key]))
            check('Twin injection alone set to zero', twin_diag['injection_A'] == 0.)
            edited_y = signal(edit, face)
            twin_y = signal(twin, face)
            delta = edited_y - twin_y
            p = editor_band(label)
            search = list(range(max(-15, tau - 6), min(15, tau + 6) + 1))
            lag, corr = lag_value(delta, p, search)
            full_lag, full_corr = lag_value(delta, p, list(range(-15, 16)))
            twin_lag, twin_corr = lag_value(twin_y, p, list(range(-15, 16)))
            original_y = signal(clip, face)
            denominator = float(np.sqrt(np.mean(original_y ** 2)))
            ratio_value = float(np.sqrt(np.mean(delta ** 2)) / denominator) if denominator > 0 else None
            directory = CACHE / 'E3_tests' / rid
            directory.mkdir(parents=True, exist_ok=True)
            native_path = directory / 'consumed_ii-h_native.npy'
            twin_path = directory / 'consumed_twin_native.npy'
            np.save(native_path, edit)
            np.save(twin_path, twin)
            digest = sha(native_path)
            twin_digest = sha(twin_path)
            out.append(dict(recording_id=rid, subject_id=row['subject_id'], source_start=start,
                population='frozen02B_20PURE_contexts', ground_truth_path=row['label_path'],
                target_HR=target, tau_rec=tau, L_inj=lag, correlation=corr, passed=lag == tau,
                lag_signed_difference=None if lag is None else lag - tau,
                L_inj_full=full_lag, full_correlation=full_corr, twin_L=twin_lag,
                twin_correlation=twin_corr, old_edited_L=c['old_edited_L'],
                injection_over_natural_RMS=ratio_value, original_y_RMS=denominator,
                mask_failure_fraction=failure, regenerated_native_sha256=digest,
                old_native_sha256=c['old_native_sha256'], native_hash_match=digest == c['old_native_sha256'],
                twin_native_sha256=twin_digest, search_low=search[0], search_high=search[-1]))
            write_rows('outputs/st9e_component_lag.csv', out)
            # These selected test views serve only the explicitly prescribed harness/timing.
            np.save(directory / 'original_label.npy', gt[start:start + 300])
            np.save(directory / 'ii-h_label.npy', label)
            edited_view, _ = worker.view(edit, 'PhysNet', reuse_boxes=physbox)
            np.save(directory / 'PhysNet_original.npy', physview)
            np.save(directory / 'PhysNet_ii-h.npy', edited_view)
            aligned_start = start // 30 * 30
            aligned = read_pure(files, aligned_start, start + 300 - aligned_start)
            _, original_boxes = worker.view(aligned, 'FactorizePhys')
            boxes = original_boxes[(start + np.arange(300)) // 30 - aligned_start // 30]
            view, _ = worker.view(clip, 'FactorizePhys', reuse_boxes=boxes, reuse_boxes_per_frame=True)
            np.save(directory / 'FactorizePhys_original.npy', view)
            view, _ = worker.view(edit, 'FactorizePhys', reuse_boxes=boxes, reuse_boxes_per_frame=True)
            np.save(directory / 'FactorizePhys_ii-h.npy', view)
            assets.append(dict(recording_id=rid, path=str(directory), target_HR=target,
                source_start=start, ground_truth_path=row['label_path'], original_HAR=original))
            save_json(REPORTS / 'st9e_harness_test_assets.json', dict(cases=assets, test_only=True, B_training_use=False))
            delete_temp(native_path)
            delete_temp(twin_path)
            del clip, face, roi, notched, edit, twin, aligned, view, physview, edited_view
            progress('component_lag', done, 20, started)
            budget_check((time.monotonic() - started) / done * (20 - done) + 7200)
    finally:
        worker.close()
    passed = all(r['passed'] for r in out) and signs['all_passed']
    result = dict(status='passed' if passed else 'failed', cases=len(out), exact_lag_matches=sum(r['passed'] for r in out),
        native_hash_matches=sum(r['native_hash_match'] for r in out), sign_tests_passed=signs['all_passed'],
        E3R_a_passed=passed, affected_scope_if_failed='ii-h/iv-h pre-generation and training only',
        all_prescribed_comparisons_completed=True, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    save_json(REPORTS / 'st9e_component_summary.json', result)
    return result


if __name__ == '__main__':
    finish('component', main)
