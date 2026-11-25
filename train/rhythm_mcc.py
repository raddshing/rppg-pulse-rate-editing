"""Brief 02 A6 item 9 (conditional runs; PI 7.07i item 4, 11-24): the RhythmFormer loss with the MCC term.

RhythmFormer_Loss_MCC is the archived RhythmFormer loss (src/st12c/rhythm_loss_author.RhythmFormer_Loss:
0.2 x time term + 1.0 x frequency cross-entropy + 1.0 x HR distribution term, per sample, Fs 30) with its time term
criterion_Pearson (negative Pearson, 1 - r) replaced by train/mcc_loss.MCCLoss(Fs=30): -MCC, band 0.6-3.3 Hz, the
implementation of the M1-M3 runs (A6 item 2). forward is inherited unchanged: the per-sample call
criterion_Pearson(pred_ppg.view(1, -1), labels.view(1, -1)), the NaN guard of the time term (set to 0), the weights 0.2,
1.0, 1.0 and the frequency and HR terms are those of the archived class. The vendor tree must be on sys.path (train/
train_pc.py sets it when imported).
"""
from mcc_loss import MCCLoss
from src.st12c.rhythm_loss_author import RhythmFormer_Loss


class RhythmFormer_Loss_MCC(RhythmFormer_Loss):
    def __init__(self):
        super().__init__()
        self.criterion_Pearson = MCCLoss(Fs=30)
