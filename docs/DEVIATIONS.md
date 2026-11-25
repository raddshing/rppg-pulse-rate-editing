# Deviations, technical failures and the history of the test data

This file lists what was registered, what happened, when, and the consequence for the reader, in the format proposed by
Willroth and Atherton (2024), with one more column: the evidence (commits of this repository, files, and the numbers of
the internal operations reports in which each event was reported when it happened). Times are KST (UTC+9), year 2026.
The registered documents are `docs/PREREGISTRATION_STAGE5.md` and `docs/PREREGISTRATION_STAGE5_AMENDMENTS.md`
(entries A1–A8 in order of date); they are not changed by this file.

Section 1 is the deviation table prepared for an earlier, longer version of the manuscript, unchanged except that references to sections and
tables of the manuscript are replaced by the repository files that hold the values. Sections 2–6 add the deviations of
the post hoc block (entry of 2026-11-20, "A6"), every technical failure and re-run (A6 item 10), the reading of
A6 item 8(b), the registered evaluation on VIPL-HR (A7) and its sensitivity analysis (A8), and the history of the test data.

## 1. Test run and the analyses added after it

| Registered | What happened | When | Consequence | Evidence |
|---|---|---|---|---|
| Stage (a) records the frame rate and key-frame interval of the MCD-rPPG test recordings. | The stage-(a) code did not record them, which the rehearsal had not revealed. They were read from the bitstream headers, without decoding pixels, by a script outside the hashed code, after stage (a) and before stage (b). | 15–16 Nov | None on the analyses. The key-frame interval is 12 frames in all 385 recordings; one recording has its key-frame frequency outside the key-frame line. | commits 081f0f7, 3ec2e31; `tools/mcd_gop.py`; `docs/TEST_NOTES.md` (stage (a) and its correction) |
| Between stages (b) and (c), at most 1% of the readouts of any original-arm checkpoint may differ from the stored October predictions; otherwise stage (c) waits until the cause is established. | Each FactorizePhys checkpoint differed in 0.62–1.02% of its MCD-test readouts (234 in all), and one exceeded 1%. All differences lay in windows whose batch members differed between the October evaluation (batches across recordings) and the registered batching (per recording). A first re-run attempt started in the wrong software environment and made no prediction; the repeated re-run reproduced the stage-(b) predictions byte for byte. | 16 Nov | The stop was released with this cause before stage (c). The FactorizePhys predictions on MCD-test follow the registered batching; MMPD-B, PhysNet and RhythmFormer were unaffected. | commits 094ef73, 3ea17dd, 51feedd; `docs/TEST_STAGE_B_STOP_RELEASE.txt`; `docs/TEST_NOTES.md` (stage (b)) |
| Stage (c) is run once; after an execution error it is corrected without changing any analysis and repeated from its start. | The first run stopped in the MMPD-B analysis because its command did not pass the test-set input plan. The command was corrected and stage (c) repeated. | 16 Nov | Results are from the repetition, which reproduced every row that the first run had written; 56 of its 60 files were byte-identical, the others differing in run times, the log and rows the first run had not written. | commits 89b786b, 2595fd0; `docs/TEST_STAGE_C_RUN1_COMPARISON.csv`; `docs/TEST_NOTES.md` (stage (c)) |
| The input-filtering sensitivity analysis for RhythmFormer on MCD-rPPG is omitted, because the filtering failed development rule 2. | The filtered path remained in the stage commands, so stage (b) predicted the 15 confirmatory RhythmFormer checkpoints on filtered inputs and stage (c) analysed them. Found when the tables were read for the paper. | 16 Nov | Reported as not pre-registered; it put no contrast in another tier, and no statement is based on it. | commits b34e187, 40935b2; `results/test_run/MCD-test/sensitivity.csv` (analysis `blur15`) |
| --- | Audit of UBFC-rPPG and of the development splits of MMPD and MCD-rPPG with the PURE procedure (no model). | 17 Nov | Post hoc and descriptive; no confirmatory claim. | report 6.97; `results/audit/cross_corpus/` |
| --- | Recording-level summaries of the PURE audit (slopes with bootstrap intervals) and an export of the pool-item phases for the figure of the main text; the audit and the phase measurement themselves predate the registration. | 17 Nov | Descriptive; the slopes replace the context-level slopes of earlier drafts (−1.18 and +0.08), whose contexts are not independent. | report 6.97; `results/audit/pure_polarity_lag/`, `results/audit/pool_phase_120/` |
| Development-only diagnostics that need new training or inference and are completed after the last amendment without being listed in it are not part of the paper. | PhysNet trained with the loss of RhythmFormer (12 runs) had been completed before the last amendment but was not listed in it; its reading rules had been recorded in the project log before its results, not in the repository. | Reported 19 Nov | Exploratory, development data only. | report 7.01; `results/dev_diagnostics/keyframe_and_loss_swap_701/ra1_lossswap.csv` |
| --- | Descriptive summaries of existing audit exports: resultant length of the phases of the 120 pool items by target band, and correlation strength of the PURE recordings whose correlation of largest magnitude is positive. | 19 Nov | Descriptive; no model and no test data. | reports 6.97, 7.01; `results/audit/pool_phase_120/pc_items_phase_by_band.csv`, `results/audit/pure_slopes_701/` |
| --- | Correction of the prediction level of arm (0): an offset and an affine map fitted for each seed on the development windows (MMPD-A, 723; MCD-dev, 1,446) and applied to the stored test predictions of arm (0), with contrasts against the registered arms. | 19 Nov | Post hoc, no tiers. Stored predictions and reference HRs only; no training or inference. | report 7.01; `results/test_run/saved_output_analyses/level_shift_701/` |
| --- | Split of MMPD-B by activity for (ii-pc) − (i-simple), (ii-pc) − (i-pc) and (ii-pc) − (0), with interactions against the stationary recordings. | 19 Nov | Post hoc, no tiers. The rotation and talking recordings have high-HR windows from two subjects each. | report 7.01; `results/test_run/saved_output_analyses/motion_conditions_701/` |
| --- | Readout of the stored October test predictions of RhythmFormer trained with its authors' augmentation, an earlier strategy outside the registration, with the windows and the readout of the test run. | 19 Nov | Post hoc, no tiers. Predictions made without the inference guards; on MCD-test, recording-level scores on 95 recordings. | report 7.01; `results/test_run/saved_output_analyses/rhythmformer_builtin_augmentation_701/` |
| --- | Bootstrap over the ten subjects for the two slopes of the PURE audit. | 19 Nov | Descriptive; replaces the bootstrap over recordings. The intervals widen (−2.20 to −0.58 and −1.63 to +1.08) without changing the conclusion. | report 7.01; `results/audit/pure_slopes_701/da_slopes.csv` |
| --- | One-bin hit rate, half-HR rate and bin-balanced MAE of the arms of the test run, with contrasts. | 20 Nov | Post hoc, no tiers. Stored readouts and reference HRs only. | report 7.02; `results/test_run/saved_output_analyses/hit_rate_half_HR_balanced_702/` |
| --- | Signal-to-noise ratio of the stored output waveforms at the reference HR, with contrasts. | 20 Nov | Post hoc, no tiers. No re-inference; the HRs read from the waveforms equal the stored readouts. | report 7.03; `results/test_run/saved_output_analyses/output_snr_703/` |
| --- | Paired contrasts of the within-subject change on MCD-test, for all 149 subjects and for the 19 with a reference HR of at least 100 bpm after exercise. | 20 Nov | Post hoc, no tiers. | report 7.02; `results/test_run/saved_output_analyses/within_subject_pairs_702/` |
| --- | Median offset and LAD map as further corrections of the prediction level of arm (0). | 20 Nov | Post hoc, no tiers. | report 7.03; `results/test_run/saved_output_analyses/level_calibration_703/` |
| --- | HR of the labels presented in training, read like the test windows. | 20 Nov | Descriptive. Training data only; labels of removed files recomputed without making videos; no model. | report 7.02; `results/dev_diagnostics/label_distribution/training_labels_702/` |
| --- | Bonferroni-adjusted intervals for the 24 confirmatory contrasts. | 20 Nov | Post hoc sensitivity analysis; the registered tiers are not changed. | report 7.02; `results/test_run/saved_output_analyses/bonferroni_702/` |
| --- | Line-removal procedure of 12 November, computed before the inference guards, transferred without recomputation. | 20 Nov | Development data; this procedure is the one whose failure led the amendment of 15 November to omit line removal. | report 7.02; `results/dev_diagnostics/b9_recheck_702/` |
| --- | Harmonic phase check of the audit recordings. | 20 Nov | Post hoc and descriptive. Training and development data with the audit signals recomputed by the audit code; no model and no test data. Decision rules fixed before the results. | report 7.03; `results/audit/harmonic_phase_703/` |

## 2. Post hoc block (entry of 2026-11-20, A6): its two departures

The entry was committed (f64c68a, 20 Nov 14:50) after the test run and before any run it describes was started. Its
results on the test sets are post hoc and are reported with 95% intervals and without tiers. The entry itself names two
departures and says that both are listed in the deviation table:

| Registered | What happened | When | Consequence | Evidence |
|---|---|---|---|---|
| Item 9 of the first entry of 2026-11-15 (A4): development-only diagnostics that need new training or new inference, are completed after that date and are not listed there are not part of this paper. | The post hoc block (A6 items 1–3 and 9: 159 checkpoints, listed in `docs/A6_CHECKPOINTS_SHA256.csv`) is reported in the paper. | registered 20 Nov; runs 20–25 Nov | Reported as a separate block, post hoc, with intervals and without tiers; the registered test run, its tiers and statements are unchanged. | A6 entry (f64c68a); reports 7.04, 7.07l, 7.07n, 7.07x, 7.07y; `results/posthoc_704/`, `results/posthoc_P/`, `results/posthoc_R/` |
| The same entry: development-only diagnostics are never evaluated on the test sets. | The nine checkpoints of development rule 5 (PhysNet with standardized labels, arms (ii), (ii-lag), (ii-pc), three seeds) were evaluated by guarded inference on MMPD-B and the MCD-rPPG test half, as part of the setting PhysNet-Std. | 23 Nov | Their test results are post hoc and appear only within PhysNet-Std. | A6 items 1 and 5; `docs/A6_PREDICTION_DIGESTS.csv`; `results/posthoc_704/` |

Other facts of the block that a reader may need:

- Trainer check (A6 item 4). Arms (0), (i) and (ii) of PhysNet and FactorizePhys were retrained with the training code
  of the block with seeds 20261131–20261133 (18 runs). Initial weights, per-step losses, exposure counts and the model,
  optimizer and scheduler tensors equal those of the confirmatory runs; the checkpoint files differ only in metadata
  entries (run name, step, arm, pool manifest). The interactions of item 7 are therefore computed once. The 18 retrained
  checkpoints are not evaluated on the test sets or on VIPL-HR (A7 item 8). Reports 7.04b, 7.04 (trainer check, final).
- Pools. The pools of FactorizePhys (i) and (ii), and for the RhythmFormer runs of item 9 the pools (ii-pc) and (ii-lag),
  were no longer stored and were regenerated by the original generator; every item matched the SHA-256 of the
  confirmed manifest (2,384 of 2,384 items for each RhythmFormer arm). The second pool of item 3 was checked in the
  same way against its own manifest. Reports 7.04, 7.07i, 7.07x, 7.07y.
- Non-finite losses: none. Items 1–2: 75 runs (54 PhysNet runs, 301,320 steps; 21 FactorizePhys runs, 31,290 steps);
  item 3 (new seeds): 54 runs, 207,720 steps; item 9 (RhythmFormer-MCC): 21 runs, 93,870 steps. Reports 7.04 (status of
  23 Nov), 7.07; training logs.
- Repeated inference (A6 item 5): for each setting, one checkpoint was inferred twice on each test set; the 16 repeat
  passes have the same summary hash as the first passes (`docs/A6_PREDICTION_DIGESTS.csv`, rows "(repeat pass)").

## 3. Technical failures and re-runs (A6 item 10)

A6 item 10: a run that fails for a technical reason is repeated from its inputs and reported; a run whose loss becomes
non-finite is reported and not repeated with other settings (there was none). Every failure, stop and re-run of the
block is listed. Outputs of failed attempts were moved aside, not deleted.

| When | What happened | Consequence | Evidence |
|---|---|---|---|
| 20 Nov 14:51 | Trainer check (A6 item 4), first attempt: the data-loader worker could not find a script given by a relative path after the trainer changed its working directory; stopped before step 1. | Restarted with absolute paths; no step had been made. | reports 7.04b (interim), 7.04 (trainer check, final) |
| 20 Nov 15:56 | Trainer check, FactorizePhys (ii), seed 20261132, first attempt: the manifest of the regenerated pool had relative paths; stopped before step 1. | The three seeds of this arm were run again from the start with a copy of the manifest with absolute paths. | report 7.04 (trainer check, final) |
| 20 Nov | MMPD-A comparison of the trainer check, first attempt: the run name in the checkpoint index differed from the one stored in the checkpoint; stopped before inference. | No prediction was written; repeated after the index was corrected. | report 7.04 (trainer check, final) |
| 20 Nov 19:14 – 23 Nov 15:05 | The pre-training label check of the FactorizePhys-MCC runs (A6 item 2, 21 runs) stopped: the check tool read the reference SHA-256 from the rows of the (i) pool instead of the (i-pc) manifest. While the tool was fixed, it stopped twice more before training (an arm without a pool; arms whose reference is the October pool). No training step had been made; nothing else started for about 68 hours, which was noticed late. | None on the results: the labels loaded by the training code equal the confirmed (i-pc) manifest (2,384 of 2,384); the check of the earlier runs was repeated with the fixed tool and agreed. The 21 runs started on 23 Nov 15:11 and ended at 15:29. | report 7.04 (status of 23 Nov) |
| 23 Nov 15:30 | The pre-check of the 7.04 inference runner stopped it: the runner required A6 as the last amendment, while A7 had been committed. | No inference had started; corrected and relaunched at once. | report 7.07 |
| 23 Nov 16:30 – 19:51 | One-epoch check of RhythmFormer (new seeds, arm (0)): the data worker could not start (relative script path; the RhythmFormer backend changes the working directory) and the main process waited while holding the lock of GPU 1. | Stopped at 19:51; VIPL-HR stage (b1), the 7.04 MMPD-B inference and the GPU-1 lane of the new-seed runs waited behind it. Paths made absolute. | report 7.07d |
| 23 Nov 20:22, 20:32, 20:43 | The three RhythmFormer runs of arm (0) with the new seeds 20261148–20261150 were stopped by the progress guard of the operations tools at about step 1,650 of 4,470. The fault was in the guard (its first-step probe always failed), not in the runs. | No checkpoint had been saved; the three runs were repeated from their inputs (20:55–22:15, completed). About 31 GPU minutes lost. | report 7.07f |
| 23 Nov 23:32 – 24 Nov 00:15 | Batch preparation of the PhysNet new-seed runs did not pass: the operator's own first criterion required every label of the second pool to differ from the first pool, but 7 of 2,972 (ii)/(ii-pc) items have the same label by construction (the label resampling uses a rational ratio limited to denominator 64; the confirmed first pool has 15 such groups). | No training had started. The criterion was replaced by an exact rule (equal labels exactly when the rational ratios are identical; every target differs; (i) labels all differ; originals equal), which passed; the runs then started. | report 7.07i |
| 24 Nov 02:09 – 02:10 | The first three RhythmFormer-MCC runs (A6 item 9; arms (0), (i), (i-pc), seed 20261131) failed when the loss was built (recursion in the operator's wrapper that substitutes the MCC term). No training step. | Wrapper corrected (the criterion is replaced after the archived initialization; three-step smoke test: loss of the first step 32.7346 against 32.9908 of the archived loss on the same batch, same initial weights); the three runs were repeated from their inputs and completed. | report 7.07n |

Scheduling changes that did not fail and did not change any computation (handovers of waiting runners, more CPU workers
for pool regeneration, two inference lanes side by side) are recorded in the operations log and are not listed here.

## 4. Reading of A6 item 8(b)

| Registered | What happened | When | Consequence | Evidence |
|---|---|---|---|---|
| Item 8(b): the cost of misalignment is called absorbed if the MCC interval includes 0, "the same contrast with the negative Pearson loss lies below 0", and the lower bound of the loss interaction is above 0. | "Lies below 0" was read as the upper bound of the interval of the negative-Pearson contrast below 0 (interval reading), as for the other statements of the entry. The reading was fixed before the statements were computed; the point-estimate reading was computed alongside. | 24 Nov | Both readings give the same 7 absorbed cells of 12 for PhysNet, PhysNet-Std and FactorizePhys; no statement differs. Item 9 (conditional RhythmFormer-MCC runs) was triggered. | report 7.07l; `results/posthoc_704/test/statements.csv` (columns `kind`, `kind_if_point_reading`, `readings_differ`) |

## 5. VIPL-HR (A7) and its sensitivity analysis (A8)

| Registered | What happened | When | Consequence | Evidence |
|---|---|---|---|---|
| A7 item 5: the reference HR is read from the BVP; item 6: if the BVP rate cannot be fixed within 0.98–1.02 of the mean of gt_HR, the mean of gt_HR becomes the reference. | The median ratio was 0.979 at 60 Hz and 0.933 at the rate of each record, both outside 0.98–1.02. The fallback of item 6 applied: the mean of gt_HR is the reference and the BVP only verifies windows and decides whether they lie within the record. | 23 Nov (stage (a), 16:25–16:47) | The sensitivity analysis of item 5 (gt_HR as reference) coincides with the primary analysis; A8 was added in its place (row below). | commit 3de6c80 (`docs/A7_STAGE_A_SHA256.txt`); `results/vipl/stage_a/bvp_rate.json`, `results/vipl/stage_a/exclusions.csv`; report 7.07c |
| A7 items 4–5: windows are verified as for MMPD; the reference is the mean of gt_HR over the window (fallback of item 6). | In one recording pair (one subject, scene v6, the same session recorded by source1 and source3), the second window has a mean gt_HR of 255.0 bpm, while the HR read from its BVP is about 63 bpm; both windows passed the registered verification. | stage (a), 23 Nov | Kept as registered: both windows are among the 213 high-HR and the 55 windows at or above 120 bpm of the primary analysis. They are 2 of the 4 primary windows above 180 bpm that the signed-error figure does not show. No window was removed after the analysis. | `results/vipl/common_support.csv`; `results/figures/fig4_signed_error_by_bin_VIPL-HR.csv` |
| A7 item 8: secondary checkpoints are those of the entries of 2026-11-15 and 2026-11-20 that exist when A7 is committed. | The 54 new-seed checkpoints (A6 item 3) and the 21 RhythmFormer-MCC checkpoints (A6 item 9) were trained after the A7 commit and were not evaluated on VIPL-HR. The 18 retrained checkpoints of A6 item 4 were not evaluated (their tensors equal the confirmatory checkpoints). | A7 committed 23 Nov 15:29 | Table III shows "---" for RhythmFormer-MCC on VIPL-HR. | commit 29977ee; `docs/A7_SECONDARY_MANIFEST.csv` (84 checkpoints) |
| A7 items 3–4: windows within the video and the BVP record; every checkpoint must predict a window for it to be analysed. | The requirement of a prediction from every checkpoint removed 1,622 of 6,762 windows, all the last window of a recording, which the predictions of at least one backbone did not reach (PhysNet 1,614, RhythmFormer 1,076, FactorizePhys 1,076). | stage (c1), 23 Nov | 2,657 primary and 1,870 secondary windows analysed; the common support is the same in (c1) and (c2). | `results/vipl/common_support.csv`; reports 7.07h, 7.07p |
| --- (A8, added after stage (a)) | Sensitivity analysis with the BVP read at 60 Hz as the reference, for the primary checkpoints on the primary set, registered after stage (a) and before any prediction on VIPL-HR was analysed. Its code was committed after stage (c2), in the form published here, and reproduces the outputs byte for byte except start and end times. | registered 23 Nov 20:57; analysed 24 Nov | Values only, without tiers. | commits efa380c, d6ec294; `docs/A8_OUTPUT_SHA256.txt`; `results/vipl/A8_sensitivity/`; reports 7.07e, 7.07q |

## 6. History of the test data

| When | Data | What was done | Evidence |
|---|---|---|---|
| Aug 2026 | VIPL-HR | An earlier study of our group evaluated its own EfficientPhys, PhysNet and FactorizePhys models (trained on PURE or UBFC-rPPG) on VIPL-HR; its results were not consulted in designing this study. | A6 item 11; A7 |
| Oct 2026 | MMPD-B, MCD-rPPG test half | The delay-rule arms were evaluated (October evaluation); their results were known when the correction was designed, so the test results are a conditional confirmation. | manuscript, Section IV-D; A5 (commit 98df687) |
| Oct 2026 | VIPL-HR | Its reference signals and frame times were read, without decoding its videos or running any model, to plan the test of another study. | A6 item 11; A7 |
| 11 Nov | — | Stage-5 pre-registration committed before the test data were opened for it. | commit dbb4535 |
| 15 Nov 19:14 (renewed 20:10) | MMPD-B, MCD-test | Unsealing instruction; test stage (a) ended 22:33. | `docs/TEST_NOTES.md`; commit c6cf519 |
| 16 Nov | MMPD-B, MCD-test | Stage (b) 12:14–17:11 (including the unregistered filtered RhythmFormer predictions), the 1% stop and its release, stage (c) first run 18:22–18:30 (stopped) and repetition 19:59–21:49. | Section 1; commits 094ef73–40935b2 |
| 19–20 Nov | MMPD-B, MCD-test | Post hoc analyses of the stored test outputs and of the stored October predictions (Section 1, rows of 19–20 Nov); no inference. | reports 7.01–7.03 |
| 20 Nov 14:50 | — | A6 committed, before any of its runs. | commit f64c68a |
| 23 Nov 15:30 – 24 Nov 00:29 | MCD-test, MMPD-B (and MMPD-A, MCD-dev) | Guarded inference of the 84 checkpoints of A6 items 1–2 and rule 5 (MCD-test from 17:15, MMPD-B from 22:15:44 after VIPL-HR stage (b1)), with the repeat passes. | `docs/A6_PREDICTION_DIGESTS.csv`; report 7.07g |
| 23 Nov 16:25 – 24 Nov 04:46 | VIPL-HR | Stages (a), (b1), (c1), (b2), (c2) of A7; A8 registered after (a). | commits 3de6c80, 1b59c73, 126b0b7, f16ddef, 13802c0, efa380c |
| 24 Nov 07:53 – 23:23 | MMPD-B, MCD-test | Guarded inference of the 54 new-seed checkpoints (A6 item 3), with the repeat passes. | `docs/A6_PREDICTION_DIGESTS.csv`; reports 7.07v, 7.07x |
| 25 Nov 02:30 – 06:54 | MMPD-B, MCD-test (and MMPD-A, MCD-dev) | Guarded inference of the 21 RhythmFormer-MCC checkpoints (A6 item 9), with the repeat passes. | `docs/A6_PREDICTION_DIGESTS.csv`; report 7.07y |

No model of this study was trained on any test set, and every inference on a test set after 16 Nov used the guarded
evaluation path of the test run (commit 19ecf5d).
