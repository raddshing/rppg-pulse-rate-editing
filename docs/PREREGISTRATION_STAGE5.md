# Pre-registration: post-review confirmatory test evaluation (stage 5)

This document fixes, before any test data are opened, how the post-review training arms will be evaluated once on the held-out test data. It is committed to this repository together with `docs/PREREGISTRATION_STAGE5_MANIFEST.csv`, so that its date is fixed. Later changes are made only as dated amendments in separate files; this document and the manifest are not rewritten.

## 1. Background and chronology

**October evaluation.**
- The original arms (0), (i) and (ii) were trained on PURE together with other strategies.
- They were evaluated on MMPD-B and on all 192 evaluable MCD-rPPG recordings. The results are in the original manuscript.
- Its primary comparison (PhysNet (ii) vs (0) on MMPD-B) was committed on 2026-10-15, before training.

**MCD-rPPG split.**
- The split was made after the October evaluation and before any post-review analysis for this paper.
- It covers the subjects of the 789 MCD-rPPG recordings with a confirmed frame-to-PPG correspondence, divided into a development half and a test half.
- Subjects were stratified by whether they have a window with a reference HR of at least 100 bpm. Within each stratum, they were assigned alternately to the two halves in the SHA-256 order of their IDs.
- The split used reference signals and metadata only, not model outputs.
- Both subject lists are identified by hash in the manifest.

**Review and audit.**
- An external review pointed out that the pulse editor may encode the video–PPG polarity as a half-period delay. This would cause a phase error of about π(r − 1) between the injected pulse and the label when the heart rate is edited by a ratio r.
- We audited the editor on PURE only.
  - The delay τ is the lag of the maximum *signed* correlation, and the injection never flips sign.
  - The natural relation is predominantly negative: the maximum |correlation| is negative in 49 of 59 recordings.
  - τ follows the half period (slope −1.18 against the half period, both in frames). The lag of the maximum |correlation| does not depend on heart rate (slope +0.08).
- The same τ was also used to shift the labels of the time-resampling arm, by τ·(r − 1) source frames.

**Corrections and development.**
- The corrections were defined from the PURE audit alone: sign s = −1 and delay d = +2 frames. d is the median lag of the maximum |correlation| over the 49 negative-sign PURE recordings.
- The corrected arms were evaluated on development data only: MMPD-A and the development half of MCD-rPPG.

**Access to the test data since the October evaluation.**
- The work for this paper has not opened any file of MMPD-B. It has not opened any video, PPG, label or per-recording metadata file of a test-half subject of MCD-rPPG.
- The MCD-rPPG index table was read when selecting development-half recordings. It is a single file covering all subjects, with clinical columns. Rows of other subjects were discarded before any use.
- Both test sets are also held out as test data of another study by the same group. According to that study's records, no model has been run on either set since the October evaluation. Only reference signals and metadata have been read, for example to make and describe the MCD-rPPG split.
- No file of either test set will be opened before this document and the manifest are committed.

## 2. Arms

All arms are trained on the same PURE training split with seeds 20261131–20261133. The backbones are PhysNet, RhythmFormer, and FactorizePhys trained under the matched conditions.

| Arm | Description |
|---|---|
| (0) | no augmentation |
| (i) | time resampling, original label shift τ·(r − 1) ("delay-rule" resampling) |
| (ii) | pulse editing, original injection +g*(t − τ) ("delay-rule" editing) |
| (i-pc) | time resampling, label shift d·(r − 1); label samples that the shift moves outside the recording take the nearest edge value (at most 2.04 frames) |
| (ii-pc) | pulse editing with explicit polarity and delay, injection −g*(t − d) |

- The corrected arms use the same generation code, seeds and training settings as their originals. Only the injection term differs, or the label shift with the edge rule above. The checks of this correspondence are reported in the paper.
- The proposed method is (ii-pc) for all backbones. No backbone-specific choice of arm will be made.
- The checkpoints to be evaluated are identified by SHA-256 in the manifest. The manifest lists every checkpoint whose training had finished when this document was committed. Any other checkpoint is added by a dated amendment before any test data are opened.

## 3. Test data and evaluation

**Definitions**
- Window construction, reference verification, the high-HR definition, recording selection and the HR readout are those of the original manuscript. They are implemented in the October evaluation code, which this repository contains as of the commit that adds this document. They are used unchanged.
- **HR readout.** One rule for predictions and references, including the manuscript's preprocessing: the highest peak of a 512-point periodogram within the band. The primary band is 0.6–3.3 Hz.
- **MMPD-B windows.** Only verified windows (Q*) are used: no signal defect, and the three model-independent reference estimates agree (spectral peak, autocorrelation and median inter-beat interval). A window is high-HR if its reference HR is at least 100 bpm.
- **MCD-rPPG high-HR windows.** The window is gap-free, its three reference estimates agree as above, and its spectral-peak reference HR is at least 100 bpm.

**MMPD-B (primary).**
- Verified windows (Q*) in the primary band.
- Inference follows the documented input order and per-model random-number handling, as for the development data.

**MCD-rPPG test half (replication).**
- Only test-half subjects are used.
- The recording selection of the original manuscript is applied: front-view webcam video, confirmed frame-to-PPG correspondence, and at least one high-HR window as defined above.
- The random-number state is reset per recording.
- The development half is excluded because it was used for development.

**Run.**
- All arms, including (0), (i) and (ii), are evaluated in the same run under the same settings.
- For the original arms, the re-inferred MMPD-B predictions are compared with the stored October predictions. Any difference is reported.

**Metric.** High-HR MAE.

**Intervals.** 95% intervals from a crossed subject × seed bootstrap with 10,000 resamples. Contrasts are paired by seed.

## 4. Confirmatory contrasts (the only confirmatory analyses)
For each backbone, on high-HR MAE:
- C1: (ii-pc) − (0)
- C2: (ii-pc) − (ii)
- C3: (i-pc) − (i)
- C4: (ii-pc) − (i-pc)

**Decision statements**
- We claim that "corrected pulse editing improves on no augmentation for all three backbones" only if C1 has a 95% upper bound below 0 on MMPD-B for every backbone. MCD-rPPG is reported as replication.
- The alignment effects C2 and C3 are reported per backbone with their intervals. We state that alignment improves an arm only for a backbone and dataset where the upper bound is below 0.
- We claim pulse editing is superior to time resampling only for a backbone whose C4 upper bound is below 0 on MMPD-B. Otherwise we state that no difference was established.

## 5. Pre-specified sensitivity analyses
- **MCD-rPPG key-frame line.**
  - The HR search is repeated under two exclusion rules, each a separate sensitivity analysis.
    - October rule: the periodogram bins between 2.45 and 2.55 Hz are set to zero.
    - Main-lobe rule: the periodogram bins whose centre frequency lies between 2.40 and 2.60 Hz are set to zero. This covers the main lobe of a 2.5-Hz component in a 10-s window.
  - The key-frame-line fraction is reported for every arm. It is the share of windows read within 147.66–154.69 bpm.
  - A line-removal input procedure may be fixed on development data before unsealing. If it is, it is applied as an additional sensitivity analysis; otherwise it is omitted.
- **Narrow band.** HR is read in 0.75–2.5 Hz, as in the original manuscript.
- **Pre-written sentence.** If C2 for RhythmFormer on MCD-rPPG has a 95% lower bound above 0, we state: "On the development half, the alignment correction increased RhythmFormer's attraction to the 2.5-Hz key-frame line. The key-frame-line fractions and the results under both exclusion rules are shown alongside."

## 6. Exploratory (not confirmatory)
- All other arms, including (ii-inv), (ii-lag), (i-simple), label-distribution arms and earlier strategies.
- All other metrics, including within-range rank correlation, within-subject change, output phase and per-bin errors.
- All analyses on development data.

## 7. Commitments
- The test data are evaluated once.
- No model, arm, rule or metric is changed after unsealing.
- Any deviation from this document is reported with its reason.
- The manifest is extended only by dated amendments, made before any test data are opened.
- The October primary comparison (committed 2026-10-15) is reported as originally specified. The original arms are labelled as delay-rule arms.

## 8. Manifest
`docs/PREREGISTRATION_STAGE5_MANIFEST.csv` has the columns `kind, backbone, arm, seed, n, sha256`.

**`checkpoint` rows**
- One row per checkpoint file that the evaluation will load, with the SHA-256 of the file bytes.
- For the original arms, this is the file loaded in the October evaluation.

**`mcd_dev_subjects` and `mcd_test_subjects` rows**
- Each row holds the SHA-256 of the subject list of one half.
- The list is written as the subject IDs sorted as strings, one per line, in UTF-8, with LF line ends including the last line.
- `n` is the number of subjects.
- The lists themselves are released with the test results.
