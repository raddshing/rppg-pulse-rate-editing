"""Prepare authorized original PURE training tensors before B-G-L calibration."""
import os
os.environ['ST9_NAMESPACE'] = 'st9g'
from src.st9e.native import *
from src.st9e.model_view import ViewWorker
from src.st9g.preprocess import diff_normalize_data, diff_normalize_label, standardized_label


def main():
    path = REPORTS / 'st9g_original_settings.json'
    check('Original tensor settings committed', sha(path) in (ROOT / 'reports/ST4_INSTR03_PREFIX.md').read_text())
    settings = json.loads(path.read_text())
    for p, digest in settings['source_hashes'].items():
        check('Frozen original tensor dependency', source_hash(p) == digest)
    _, pure, allowed = indices()
    bulk = Path(PATHS['bulk']) / 'st9g/original'
    bulk.mkdir(parents=True, exist_ok=True)
    recordings = sorted({r['recording_id'] for r in settings['chunks']})
    manifest = []
    started = time.monotonic()
    worker = ViewWorker()
    try:
        for done, rid in enumerate(recordings, 1):
            budget_check()
            check('Bulk free at least50GB before original writes', __import__('shutil').disk_usage(bulk).free >= 50_000_000_000)
            files = frame_files(pure[rid])
            label = contact(pure[rid], 'PURE', allowed)
            clip = read_pure(files, 0, len(files))
            for model in ['PhysNet', 'FactorizePhys']:
                chunks = [r for r in settings['chunks'] if r['recording_id'] == rid and r['model'] == model]
                if not chunks:
                    continue
                view, _ = worker.view(clip, model)
                if model == 'PhysNet':
                    data = diff_normalize_data(view)
                    labels = diff_normalize_label(label)
                else:
                    data = view
                    labels = standardized_label(label)
                directory = bulk / model / rid
                directory.mkdir(parents=True, exist_ok=True)
                for c in chunks:
                    start = c['start_frame']
                    length = c['chunk_length']
                    x = directory / f'chunk_{c["chunk_index"]}_input.npy'
                    y = directory / f'chunk_{c["chunk_index"]}_label.npy'
                    check('Never overwrite an original instruction03 tensor', not x.exists() and not y.exists())
                    np.save(x, np.ascontiguousarray(data[start:start + length], dtype=np.float32))
                    np.save(y, np.ascontiguousarray(labels[start:start + length], dtype=np.float32))
                    manifest.append(dict(**c, input_path=str(x), label_path=str(y), input_bytes=x.stat().st_size,
                        label_bytes=y.stat().st_size, input_sha256=sha(x), label_sha256=sha(y),
                        population='frozen_Stage1_PURE_training_chunks', ground_truth_path=pure[rid]['label_path']))
                del data, view, labels
            del clip
            write_rows('outputs/st9g_original_manifest.csv', manifest)
            progress('originals', done, len(recordings), started)
            budget_check((time.monotonic() - started) / done * (len(recordings) - done) + 7200)
    finally:
        worker.close()
    check('All1493 frozen original training chunks prepared', len(manifest) == 1493)
    return dict(recordings=len(recordings), original_chunks=len(manifest),
        bytes=sum(r['input_bytes'] + r['label_bytes'] for r in manifest), original_preprocessing_whole_recording=True)


if __name__ == '__main__':
    finish('originals', main)
