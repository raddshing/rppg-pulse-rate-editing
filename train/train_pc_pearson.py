"""Diagnostic rule 1 (docs/STAGE5_DEV_DIAGNOSTICS.md; brief 02 6.89 다 3): RhythmFormer trained with the negative Pearson loss only.

Runs train/train_pc.py (unchanged; its bytes stay those of docs/A4_CODE_SHA256.txt) with one substitution made before its main()
starts: src.st12c.rhythm_backend_v2.initialize returns, instead of the author loss RhythmFormer_Loss (0.2 x negative Pearson +
1.0 x frequency cross-entropy + 1.0 x HR KL), the negative-Pearson term of that loss alone (weight 1, the author's nan -> 0 rule).
Model, initial weights, optimizer, scheduler, the RhythmFormer step (prediction standardization, per-sample criterion averaged over
the batch), pool, order, epochs and steps are those of the standard RhythmFormer run. RhythmFormer only.
Writes <out>/loss.json beside train_pc.py's result.json (result.json keeps train_pc.py's own 'loss' field).
Usage: train_pc_pearson.py RhythmFormer SEED POOL_MANIFEST OUT_DIR
"""
import os  # thread settings as train/train_pc.py, before numpy/torch are imported here
os.environ.update({'one': dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
                   'default': dict(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='16', MKL_NUM_THREADS='8')}[
    os.environ.get('OPS2_THREADS', 'one')], OPENCV_FOR_THREADS_NUM='16')
import json  # noqa: E402
import runpy  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
assert sys.argv[1] == 'RhythmFormer', 'rule 1 is defined for RhythmFormer only'
sys.path.insert(0, str(HERE.parent / 'vendor' / 'st8_6ebbc9cd'))
sys.path.insert(0, str(HERE.parent / 'vendor' / 'st8_6ebbc9cd' / 'third_party' / 'rPPG-Toolbox'))
import torch  # noqa: E402
import src.st12c.rhythm_backend_v2 as backend  # noqa: E402
from src.st12c.rhythm_loss_author import Neg_Pearson  # noqa: E402


class PearsonOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion_Pearson = Neg_Pearson()

    def forward(self, pred_ppg, labels, epoch, FS, diff_flag):
        loss_time = self.criterion_Pearson(pred_ppg.view(1, -1), labels.view(1, -1))
        return torch.zeros_like(loss_time) if torch.isnan(loss_time) else loss_time


_initialize = backend.initialize


def initialize(settings, seed, device='cuda:0'):
    model, criterion, optimizer, scheduler = _initialize(settings, seed, device)
    assert type(criterion).__name__ == 'RhythmFormer_Loss'
    return model, PearsonOnly(), optimizer, scheduler


backend.initialize = initialize
out = Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
(out / 'loss.json').write_text(json.dumps(dict(loss='negative Pearson only (author Neg_Pearson, weight 1, nan -> 0)',
                                               replaced='src.st12c.rhythm_loss_author.RhythmFormer_Loss',
                                               train_script='train/train_pc.py (unchanged)', instruction='brief 02 6.89 다 3')))
sys.argv = [str(HERE / 'train_pc.py')] + sys.argv[1:]
runpy.run_path(str(HERE / 'train_pc.py'), run_name='__main__')
