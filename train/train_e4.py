"""7.09b E4: high-HR oversampling baseline (arm (0-os)), a wrapper around train/train_704.py (train_pc.py unchanged).

Batches as in the augmented arms: 2 originals from the unchanged original stream (default_rng([seed, 0]) permutations)
plus 2 more ORIGINAL chunks in place of the 2 pool items. Each of these 2 is drawn with the augmentation stream
default_rng([seed, 1]): with probability 40/70 from the chunks with reference HR >= 100 bpm, otherwise from the others,
uniformly within the class (with replacement). 40/70 is the share of targets >= 100 bpm under h* ~ U(70, 140), so the
expected share of labels >= 100 bpm per batch equals that of the augmented arms (34-35% on PURE). No new sample is made:
the 2 extra chunks are loaded from the original tensors with their own labels. Configuration, initial weights, optimizer,
scheduler, loss, steps (PhysNet 5,580, RhythmFormer 4,470, FactorizePhys 1,490) and inference are those of arm (0).
Writes OUT_DIR/settings_e4.json (draw rule, class sizes) besides the files of train_704/train_pc.
Usage: train_e4.py BACKBONE SEED OUT_DIR [MAX_STEPS]
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.pop('OPS2_704_ARM', None)  # arm handled here
import train_704  # noqa: E402  (imports train_pc: thread variables, vendor path, working directory)
import numpy as np  # noqa: E402

P_HIGH = 40 / 70
train_pc = train_704.train_pc


def classes(original):
    high = [i for i, r in enumerate(original) if float(r['reference_HR']) >= 100]
    low = [i for i, r in enumerate(original) if float(r['reference_HR']) < 100]
    return high, low


def setup(backbone, arm_kind=None):
    config, original, _, key, init_file, _ = train_pc.setup_ii(backbone)
    pool = [dict(r, target_HR=r['reference_HR'], e4_source='original chunk') for r in original]
    return config, original, pool, key, init_file, 'e4'


def patch_streams(original):
    import src.st9t.data as data
    high, low = classes(original)
    base = data.SourceStreams

    class Streams(base):
        def batch(self, arm, batch_size=4):
            if arm != 'e4':
                return base.batch(self, arm, batch_size)
            first = self.next_original(batch_size // 2)
            extra = []
            for _ in range(batch_size // 2):
                pick = high if self.augmentation.random() < P_HIGH else low
                extra.append(pick[int(self.augmentation.integers(0, len(pick)))])
            return first, np.asarray(extra, int), ['e4'] * len(extra)
    data.SourceStreams = Streams
    return high, low


def main():
    backbone, seed, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    max_steps = sys.argv[4] if len(sys.argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    config, original, pool, key, init_file, arm = setup(backbone)
    high, low = patch_streams(original)
    train_pc.setup = setup
    (out / 'settings_e4.json').write_text(json.dumps(dict(
        backbone=backbone, seed=seed, arm='(0-os)', originals=len(original), high_chunks=len(high), low_chunks=len(low),
        high_subjects=sorted({str(original[i]['subject_id']) for i in high}), p_high_extra=P_HIGH,
        batch='2 originals (original stream) + 2 originals drawn by class (augmentation stream)',
        expected_high_share_per_batch=(2 * len(high) / len(original) + 2 * P_HIGH) / 4), indent=1))
    # 'ORIGINAL': train_pc reads no pool manifest
    sys.argv = [sys.argv[0], backbone, str(seed), 'ORIGINAL', str(out)] + ([max_steps] if max_steps else [])
    train_704.setup = setup  # train_704.main installs its own setup into train_pc; keep ours
    train_704.main()


if __name__ == '__main__':
    main()
