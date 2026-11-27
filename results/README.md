# Aggregate results

Aggregate result tables of the stage-5 test run, the development diagnostics, the audits, the post hoc experiments
(amendment of 2026-11-20, A6), the registered evaluation on VIPL-HR (A7, A8) and the post hoc experiments and analyses
of A9 (amendment of 2026-11-26), with the data of the figures of the manuscript. Every file is a copy, byte for byte, of
an output of the analysis that produced it, except the four files marked otherwise in `MANIFEST.csv` (column `action`),
`figures/fig3b_posthoc_settings.csv` (two rows added; column `change`) and the files of `a9/`, which are the aggregate
CSVs attached to the A9 reports (their derivation in column `note`). No value was recomputed for this folder.

Not included: values per window, recording or subject; subject or recording identifiers (one exception, below);
predictions, reference signals, video or face-derived data (pools, masks, crops); checkpoints (their SHA-256 are in
`docs/`); run records with machine paths.

## Files

`MANIFEST.csv` gives, for every file: `path`, `bytes`, `sha256`, the instruction item (`item`), the analysis output it
was copied from (`source`) with that output's SHA-256 (`source_sha256`), the `action` (`copy`: byte for byte;
`copy-rows`: some rows left out, every kept line byte for byte; `copy-cols`: every row, two columns left out; `transcribe`: values of report tables written in long
form), the `change` (empty for copies), the entry of a hash list already committed in `docs/` that has the same SHA-256
(`published_hash_list`; these files can be checked against the hashes committed during the test run) and a `note`.

| Folder | Contents |
|---|---|
| `test_run/MMPD-B/`, `test_run/MCD-test/` | Outputs of test stage (c) (commit 2595fd0; the 30 files listed in `docs/TEST_STAGE_C_OUTPUTS_SHA256.txt` have the committed hashes): confirmatory contrasts with tiers, secondary and exploratory contrasts, arms, sensitivity analyses (narrow band, FSAM off, batch of one, notch variants, filtered RhythmFormer `blur15`), tier changes, MAE at or above 120 bpm, error by reference bin, key-frame line, constant predictors, benchmark tables, within-subject change (MCD-test, `m7.csv`), MMPD-B subgroups and attraction |
| `test_run/saved_output_analyses/` | Post hoc analyses of the stored test outputs (no inference): level corrections of arm (0) (`level_shift_701`, `level_calibration_703`), activity split of MMPD-B (`motion_conditions_701`), hit rate, half-HR rate and bin-balanced MAE (`hit_rate_half_HR_balanced_702`), output SNR (`output_snr_703`), within-subject pairs (`within_subject_pairs_702`), Bonferroni-adjusted intervals (`bonferroni_702`) and the stored October predictions of RhythmFormer with its authors' augmentation (`rhythmformer_builtin_augmentation_701`) |
| `dev_diagnostics/` | Development data only: the stage-(c) tables of rehearsal part 2 on MMPD-A and MCD-dev (`development_tables/`), development rules 1–6 of `docs/STAGE5_DEV_DIAGNOSTICS.md` (`rule1_...` to `rule6_...`), label distributions, key-frame line, the loss swap (`keyframe_and_loss_swap_701`), the line-removal procedure of 12 November (`b9_recheck_702`) and the output phase (`m9_output_phase`) |
| `audit/` | PURE polarity and lag audit (`pure_polarity_lag`, `pure_slopes_701`), audit of the other corpora (`cross_corpus`), harmonic phase check (`harmonic_phase_703`), phase of the 120 pool items (`pool_phase_120`) and editor validity (360-window check, sham fidelity, forehead-and-cheek comparison) |
| `posthoc_704/` | Settings PhysNet-Std, PhysNet-MCC, PhysNet-Std-MCC and FactorizePhys-MCC (A6 items 1–2): arms, contrasts, label and loss interactions, statements of A6 item 8, reproduction of the stored confirmatory values; `test/` (MMPD-B, MCD-test) and `dev/` (MMPD-A, MCD-dev) |
| `posthoc_P/` | New seeds 20261148–20261150 and second pool (A6 item 3): replication verdicts of item 8(d) (`replication_8d.csv`), arms and contrasts on the new and on all six seeds |
| `posthoc_R/` | RhythmFormer with the MCC loss (A6 item 9): arms, contrasts, loss interactions, statements, reproduction; `test/` and `dev/` |
| `vipl/` | VIPL-HR: stage (a) BVP-rate rule (`stage_a/bvp_rate.json`) and exclusion counts (`stage_a/exclusions.csv`), primary analysis (`primary_c1`, commit 126b0b7), secondary checkpoints (`secondary_checkpoints_c2`, commit 13802c0), A8 sensitivity analysis (`A8_sensitivity`, `docs/A8_OUTPUT_SHA256.txt`) and the common support (`common_support.csv`) |
| `a9/` | A9 (amendment of 2026-11-26): E1 (UBFC-rPPG training; `e1_physnet/`, `e1_rhythmformer/`, `e1_factorizephys/`), E2 and E3 (`e2e3/`), E4, E6 and E7 (`e467/`) and the stored-output analyses of item 8 (`item8/`): the aggregate CSVs attached to the reports, byte for byte; files, A9 items and prediction SHA-256 lists in `a9/README.md` |
| `figures/` | The data of the figures of the manuscript, as used for the figures |

## Columns

Most tables share these columns: `dataset`/`test_set`, `population` (window population: `Qstar` for MMPD,
`draft_rule` for MCD-rPPG, `primary` for VIPL-HR), `layer` (band: `HAR` 0.6–3.3 Hz, primary; `L0` 0.75–2.5 Hz,
narrow), `backbone`, `arm` or `strategy`, `name`/`contrast` (C1 = (ii-pc) − (0), C2 = (ii-pc) − (ii),
C3 = (i-pc) − (i), C4 = (ii-pc) − (i-pc)), `metric` (`MAE_high`, `MAE_low`, `MAE_all`, `hit_high`, ...), `point`
(estimate, bpm), `ci_lo`/`ci_hi` (95% percentile interval of the crossed bootstrap over subjects and seeds, B = 10,000,
seed 20261110), `windows` and `subjects` (support), `grade` (tier T1–T4, confirmatory contrasts only) and `values_only`
(no tier). Values are unrounded; the manuscript rounds them. Other columns are named after the quantity they hold.

## Exceptions recorded in MANIFEST.csv

- `figures/fig1_PURE_audit.csv` contains the PURE recording codes (the dataset's own codes), needed to reproduce
  Fig. 1(e); it is the only result file with recording codes.
- `audit/editor_validity/sham_summary.csv` keeps the rows with subject "all" only; the stage-output version is used,
  whose last digits differ from an earlier version only by re-writing (see its `change`).
- `audit/editor_validity/roi_combinations/st5_editor_gate.csv` (the October stage-5 editor gate: full editor ED2W-S,
  all-pixel notch with ROI injection ED2AN-S, additive variant ED2Wadd-S) keeps every row; its two last columns
  (`excluded_window`, `excluded_reason`, the same in all rows: one MMPD-A window ID and its reason) are left out.
- `vipl/stage_a/exclusions.csv` and `vipl/common_support.csv` are transcribed from report tables counted from the
  stage outputs; their values were checked against those outputs when they were written.
