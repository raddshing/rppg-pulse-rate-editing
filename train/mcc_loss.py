"""Maximum cross-correlation (MCC) loss of Gideon and Stent (ICCV 2021), for brief 02 7.04 (A6 item 2).

Source: Toyota Research Institute, RemotePPG, commit 38cfde203b38867caea37cb9bfb8314180803cb9,
iccv/src/losses/NegativeMaxCrossCov.py (SHA-256 d8467d878f54ff014147932730562a0a5c14cb3cd8e2c0b2a6b8c42a08806dc7) and
iccv/src/losses/NegativeMaxCrossCorr.py (SHA-256 5aa325afba4864b1c159202dee6d4e8fc13ce545561044a19888017d6b61bb8f),
MIT License, Copyright (c) 2021 Toyota Research Institute (TRI); the license text is reproduced at the end of this file.

The two author classes are copied below. Changes:
  1. Band. The author passes high_pass/low_pass in BPM (training defaults 40 and 250, iccv/src/parse_args.py) and keeps
     the cross-power bins with high_pass/60 <= f <= low_pass/60 Hz. Here: 36 and 198 BPM, i.e. 0.6-3.3 Hz, the band of
     our HR readout.
  2. Zero standard deviation. The author code is kept: no epsilon; a clip with std(pred) * std(label) = 0 gets the
     output 0, and the energy ratio is 1 where the cross-power is all zero. The output tensor is started as cov * 0
     instead of torch.zeros_like(cov). The values are the same; the difference is that a batch in which no clip has a
     nonzero standard deviation gives a loss of 0 with a zero gradient, where the author's tensor has no gradient graph
     and loss.backward() raises an error.
  3. Reduction and lag. MCCLoss takes torch.mean over the batch of the per-clip output, as the author training code
     (iccv/src/shared/train_utils.py, calculate_ppg_metrics). The lag of the maximum can be returned for tests; it does
     not change the value.
Loss = mean over clips of -MCC (the author's NegativeMaxCrossCorr output).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

tr = torch
BAND_BPM = (36.0, 198.0)  # 0.6-3.3 Hz


class NegativeMaxCrossCov(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super(NegativeMaxCrossCov, self).__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, preds, labels, return_lag=False):
        # Normalize
        preds_norm = preds - torch.mean(preds, dim=-1, keepdim=True)
        labels_norm = labels - torch.mean(labels, dim=-1, keepdim=True)

        # Zero-pad signals to prevent circular cross-correlation
        # Also allows for signals of different length
        # https://dsp.stackexchange.com/questions/736/how-do-i-implement-cross-correlation-to-prove-two-audio-files-are-similar
        min_N = min(preds.shape[-1], labels.shape[-1])
        padded_N = max(preds.shape[-1], labels.shape[-1]) * 2
        preds_pad = F.pad(preds_norm, (0, padded_N - preds.shape[-1]))
        labels_pad = F.pad(labels_norm, (0, padded_N - labels.shape[-1]))

        # FFT
        preds_fft = torch.fft.rfft(preds_pad, dim=-1)
        labels_fft = torch.fft.rfft(labels_pad, dim=-1)

        # Cross-correlation in frequency space
        X = preds_fft * torch.conj(labels_fft)
        X_real = tr.view_as_real(X)

        # Determine ratio of energy between relevant and non-relevant regions
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, X.shape[-1])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        zero_freqs = torch.logical_not(use_freqs)
        use_energy = tr.sum(tr.linalg.norm(X_real[:, use_freqs], dim=-1), dim=-1)
        zero_energy = tr.sum(tr.linalg.norm(X_real[:, zero_freqs], dim=-1), dim=-1)
        denom = use_energy + zero_energy
        energy_ratio = tr.ones_like(denom)
        for ii in range(len(denom)):
            if denom[ii] > 0:
                energy_ratio[ii] = use_energy[ii] / denom[ii]

        # Zero out irrelevant freqs
        X[:, zero_freqs] = 0.

        # Inverse FFT and normalization
        cc = torch.fft.irfft(X, dim=-1) / (min_N - 1)

        # Max of cross correlation, adjusted for relevant energy
        max_value, max_index = torch.max(cc, dim=-1)
        max_cc = max_value / energy_ratio

        if return_lag:
            return -max_cc, max_index, padded_N
        return -max_cc


class NegativeMaxCrossCorr(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super(NegativeMaxCrossCorr, self).__init__()
        self.cross_cov = NegativeMaxCrossCov(Fs, high_pass, low_pass)

    def forward(self, preds, labels, return_lag=False):
        denom = tr.std(preds, dim=-1) * tr.std(labels, dim=-1)
        cov, max_index, padded_N = self.cross_cov(preds, labels, return_lag=True)
        output = cov * 0.  # author: torch.zeros_like(cov); see change 2
        for ii in range(len(denom)):
            if denom[ii] > 0:
                output[ii] = cov[ii] / denom[ii]
        if return_lag:
            # cc[tau] = sum_t pred[t + tau] * label[t] (circular index of the padded signals); a prediction that leads the
            # label by d frames (pred[t] = label[t + d]) peaks at tau = -d
            tau = torch.where(max_index >= padded_N // 2, max_index - padded_N, max_index)
            return output, -tau
        return output


class MCCLoss(nn.Module):
    """Loss = mean over the batch of -MCC (author NegativeMaxCrossCorr output); pred and label have shape (batch, time)."""

    def __init__(self, Fs=30, band_bpm=BAND_BPM):
        super().__init__()
        self.ncc = NegativeMaxCrossCorr(Fs, band_bpm[0], band_bpm[1])

    def forward(self, preds, labels):
        return torch.mean(self.ncc(preds, labels))


LICENSE = """MIT License

Copyright (c) 2021 Toyota Research Institute (TRI)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
