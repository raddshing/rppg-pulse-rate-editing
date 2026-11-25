"""Brief 02 A6 item 9 (conditional runs; PI 7.07i item 4, 11-24): RhythmFormer with the MCC time term, seven arms of A6
item 2, seeds 20261131-20261133, the pools and settings of the confirmatory RhythmFormer runs.

Wrapper around train/train_704.py (whose bytes stay those used by the P runs): apply() makes two process-local
substitutions before train_704.main starts, and nothing else:
  - src.st12c.rhythm_backend_v2.initialize -> a wrapper that calls the archived initialize unchanged (seeding, model,
    archived criterion, optimizer, scheduler) and returns train/rhythm_mcc.RhythmFormer_Loss_MCC() in place of the archived
    criterion, so that the unchanged step uses 0.2 x MCC in place of 0.2 x negative Pearson. Building either loss uses no
    random numbers. (11-24 02:10: the first version replaced the module name RhythmFormer_Loss, which the archived
    __init__ uses in super(RhythmFormer_Loss, self); the first three runs stopped at construction with RecursionError,
    before any step; original kept as cache/ops/train_704r.py.before_recursion_fix_20261124);
  - train_704.settings -> settings_mcc: the same settings with the loss field naming the MCC term (settings_704.json,
    loss_704.json). The result.json written by train_pc keeps its own field 'loss' (OPS2_LOSS switch, unset: 'archived').
Arms as for train_704: (0) OPS2_704_ARM=0 ORIGINAL; (i) OPS2_ARM=i ORIGINAL; (i-pc), (i-simple) OPS2_ARM=i with their
manifest; (ii) OPS2_ARM=ii ORIGINAL; (ii-lag), (ii-pc) OPS2_ARM=ii with their manifest. OPS2_704_LOSS stays unset.
Usage (repro environment): train_704r.py RhythmFormer SEED POOL_MANIFEST|ORIGINAL OUT_DIR [MAX_STEPS]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_704  # noqa: E402  (imports train_pc: vendor path, working directory)

LOSS_TEXT = ('RhythmFormer loss, 0.2 x MCC (train/rhythm_mcc.py: train/mcc_loss.MCCLoss, 0.6-3.3 Hz) in place of '
             '0.2 x negative Pearson; frequency and HR terms unchanged')
_settings = train_704.settings


def settings_mcc(backbone, seed, manifest, out):
    s = _settings(backbone, seed, manifest, out)
    assert backbone == 'RhythmFormer' and train_704.LOSS == 'pearson'
    s['loss'] = LOSS_TEXT
    return s


def apply():
    import src.st12c.rhythm_backend_v2 as RB
    from rhythm_mcc import RhythmFormer_Loss_MCC
    _initialize = RB.initialize

    def initialize(settings, seed, device='cuda:0'):
        model, criterion, optimizer, scheduler = _initialize(settings, seed, device)
        assert type(criterion).__name__ == 'RhythmFormer_Loss'
        criterion = RhythmFormer_Loss_MCC()
        print(f'criterion: {type(criterion).__name__}, time term {type(criterion.criterion_Pearson).__name__}', flush=True)
        return model, criterion, optimizer, scheduler
    RB.initialize = initialize
    train_704.settings = settings_mcc


if __name__ == '__main__':
    assert sys.argv[1] == 'RhythmFormer', 'RhythmFormer only'
    apply()
    train_704.main()
