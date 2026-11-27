# Pulse-rate editing — code, registration and aggregate results

This repository accompanies a manuscript in preparation on pulse–label alignment in heart-rate range augmentation for remote photoplethysmography. It contains the
generation, training and evaluation code that produced the reported models, as archived at the
project's commit `6ebbc9cd`; the stage-5 pre-registration, its amendments and the hash lists of every
stage of the test run; the code of the post hoc experiments and of the evaluation on VIPL-HR; the
aggregate results behind the tables and figures (`results/`); the evaluation bootstrap; and the MMPD
subject / validation-window lists.

Not included: MMPD-B and MCD-rPPG window lists, predictions or reference signals; values per window,
recording or subject; any face-pixel derivative (crops, frames, edited video, augmentation pools,
masks); model checkpoints (their SHA-256 are listed, see below); server paths and credentials. Data
must be obtained from the dataset providers under their licences. Window-level predictions and recording lists are left out because of the datasets' terms of use and
this repository's publication rule (aggregate values only).

Data notice. The identification codes that appear in this repository (the MMPD subjects in `lists/`,
the recording codes in the file names of the stage hash lists, the PURE recording codes in
`results/figures/fig1_PURE_audit.csv`) are the codes of the datasets themselves. The repository
contains no video, no physiological signal and no personal information.

## Layout

| Path | Contents |
|---|---|
| `src/st9e/` | Pulse editor (`editor.py`: notch, face mask, calibration, `edited`), time resampling (`generator.py`, `temporal.py`) and signal helpers |
| `src/st9g/` | PURE training chunks, target HR plan (`target_plan.py`) and pools for PhysNet / FactorizePhys: (i), (ii), (iv) |
| `src/st30g/` | Matched FactorizePhys chunks, targets and pools |
| `src/st12c/` | RhythmFormer chunks, pools and training |
| `src/st9t/`, `src/st30t/` | PhysNet and matched FactorizePhys training |
| `src/eval/`, `src/st11x/` | HR readout (`grid._prepare` + periodogram peak, bands HAR 0.6–3.3 Hz and L0 0.75–2.5 Hz; 300-sample windows) |
| `src/st5c/`, `src/st3/c4.py` | October MCD-rPPG window construction (`mcd_align.aligned_windows`: frame-to-PPG mapping, complete 10-s windows, gap rule, reference estimates) and MMPD signal-defect map (`c4.defect_map`), unchanged from `6ebbc9cd`, with the October modules they import (`src.st3.base`, `src.st3.c4_description`, `src.st3.media`, `src.st3.progress`, `src.st3.r3`, `src.st3.waves`, `src.st5c.avi_packets`, `src.st5c.base`, `src.st5c.signals`), unchanged; the subject list embedded in `src.st3.base` equals the A split in `lists/mmpd_subjects_A_B.csv`. Not included: `src.st3.r3_model` |
| `src/stage5/mmpd_background_flags.py` | Section 5 sensitivity analysis (amendment of 2026-11-11): per MMPD recording, background signal outside the PhysNet-view face box, Welch spectrum on the HR grid and the background-peak flag near 2.0 Hz and 3.0 Hz; the provider data root is read from `PULSE_EDIT_MMPD_ROOT` |
| `docs/` | Stage-5 pre-registration (`PREREGISTRATION_STAGE5.md`) and its manifest of checkpoint and MCD-rPPG subject-list SHA-256 |
| `src/st9/ops.py` | Public stand-in for the project's operational module (paths from environment variables, `GRID`, `BANDS`, guards) |
| `evaluation/` | Subject × seed crossed bootstrap (10,000 replicates, `default_rng(20261110)`) and the metric functions |
| `settings/` | Pool and training settings of the archived runs (server paths replaced by placeholders) |
| `lists/` | MMPD A/B subject lists; MMPD-A validation windows (`in_Qstar`, `high` flags; reference HR is recomputed by the code, not distributed) |
| `CHANGES.csv` | Every change relative to the archived sources |

## Results, deviations and the map to the manuscript (2026-11-25)

| Path | Contents |
|---|---|
| `results/` | Aggregate results: test run (`test_run/`), development diagnostics (`dev_diagnostics/`), audits (`audit/`), post hoc experiments of the amendment of 2026-11-20 (`posthoc_704/`, `posthoc_P/`, `posthoc_R/`), VIPL-HR (`vipl/`) and the data of the figures (`figures/`). `results/README.md` describes the folders and columns; `results/MANIFEST.csv` gives the size, SHA-256 and source of every file |
| `docs/DEVIATIONS.md` | Deviations from the registration, every technical failure and re-run of the post hoc block, the reading of its item 8(b), the deviations on VIPL-HR and the history of the test data, with the commits and reports that record them |
| `docs/PAPER_MAP.md` | Map from the tables, figures and numbers of the manuscript to the files of `results/` and `docs/` |
| `docs/A6_CHECKPOINTS_SHA256.csv` | SHA-256 of the 159 checkpoints of the post hoc block (amendment of 2026-11-20): settings of items 1 and 2 and rule 5 (84), new seeds of item 3 (54), RhythmFormer with the MCC loss of item 9 (21) |
| `docs/A6_PREDICTION_DIGESTS.csv` | One summary hash per checkpoint and data set of the predictions of these checkpoints (see below) |
| `tools/`, `train/` (OPS2 files listed in `docs/CODE_708_SHA256.csv`) | Code of the post hoc block (training wrappers, the MCC loss and its unit tests, pool generator of the second pool, checks, inference index, analyses), of the saved-output analyses, the development diagnostics, the audits and the figure data |
| `docs/CODE_708_SHA256.csv`, `docs/CODE_708_DIFF.txt` | For each of these files, the SHA-256 of the file as run and of the published file, and the lines that differ (machine paths only) |
| `config/paths.example.json` | Example of `config/paths.json`, which these tools read for paths outside the repository (October project, raw datasets, Python environments, ffmpeg) |

Checkpoint SHA-256: confirmatory checkpoints in `docs/PREREGISTRATION_STAGE5_MANIFEST.csv` and
`docs/PREREGISTRATION_STAGE5_MANIFEST_A1.csv`; exploratory checkpoints in
`docs/PREREGISTRATION_STAGE5_MANIFEST_A4.csv`; checkpoints existing at the VIPL-HR registration in
`docs/A7_SECONDARY_MANIFEST.csv`; post hoc checkpoints in `docs/A6_CHECKPOINTS_SHA256.csv`. The
checkpoint files themselves are not distributed.

Summary hash of the predictions (`docs/A6_PREDICTION_DIGESTS.csv`). For each checkpoint and data set
(and for the repeated inference of the determinism check, marked "repeat pass"), the prediction folder
holds one file per recording. For every file, a line `<path relative to the folder>  <SHA-256 of the
file>` (two spaces, lowercase hexadecimal) is written followed by a newline; the lines are sorted by
the relative path (byte order) and joined; the summary hash is the SHA-256 of the result (UTF-8).
Columns: `checkpoint_sha256`, `dataset`, `files`, `digest_sha256`. No recording code is listed.

The machine paths of the published OPS2 tools are read from `config/paths.json` (copy
`config/paths.example.json` and set the paths); paths inside the OPS2 work folder are taken relative to
the repository root, where these tools were run. The tools were not run again in this form; the
differences from the files as run are listed in `docs/CODE_708_DIFF.txt`.

Third-party code (rPPG-Toolbox at commit `b7500b8`, model definitions, the RhythmFormer author loss)
is not redistributed; obtain it from its authors at the stated commit.

## Stage-5 evaluation and exploratory-arm code (amendment A4)

Added in the A4 packaging commit at the same relative paths as in the evaluation workspace. The files are the ones run in
rehearsal part 2 of the stage-5 evaluation; files listed in `docs/A4_REDACTIONS.csv` differ only in machine-path values,
which are replaced by `<MACHINE_PATH>` here.

| Path | Contents |
|---|---|
| `eval5/` | Stage-5 test steps: stage 1 (`stage1_reference.py`, `mcd_recording_list.py`, `mcd_reference.py`), stage 2 inference (`stage2_predict.py`, `server.py`), the check between steps (b) and (c) (`stage3_check.py`), the analysis (`stage3_analyze.py`, `families.py`, `benchmark.py`), the start-up file of the face-detection worker (`face_guard/sitecustomize.py`) and the rehearsal-only comparisons |
| `eval5/drive/step.sh` | Driver; the name it is called by (`test_a.sh` ... `rehearsal_check.sh`) selects the step. Machine-specific locations are read from `local_paths*.sh`, which are not included |
| `run_test_b_locks.sh` | Launcher of test step (b): both GPUs until the step has ended |
| `b1/` | Development bootstrap (`boot.py`, identical to `evaluation/boot.py`), checkpoint roster and the development analysis used by the row check |
| `tools/` | Checkpoint index, re-run of differing recordings, row checks (`check_rows.py`, `check_rows_stage3.py`) and run comparisons |
| `train/`, `audit/` | Generation and training code of the exploratory arms ((i-simple), (ii-lag), label and processing arms) |
| `vendor/st8_6ebbc9cd/` | Archived source tree imported by the evaluation (`src/`, `configs/`): the October modules as at `6ebbc9cd` and the evaluation workspace's replacements of `src/mmpd_contract.py`, `src/st5c/base.py` and `src/st9/ops.py`. `third_party/rPPG-Toolbox` is not included |
| `docs/PREREGISTRATION_STAGE5_MANIFEST_A4.csv` | SHA-256 of the 48 exploratory checkpoints |
| `docs/A4_CODE_SHA256.txt`, `docs/A4_CODE_SHA256_THIRD_PARTY.txt` | SHA-256 of the code files as run (before redaction) and of the rPPG-Toolbox files run |
| `docs/A4_REDACTIONS.csv` | Lines whose machine-path values were replaced; SHA-256 of each file before and after |
| `docs/dev_guard_comparison/` | Summaries of the development comparisons of stored and guarded predictions (v1, v1_benchmark, v2, v2_notch): contrast cells, grade and rule changes, file and column summaries |

Before each test step the files to be run are checked in the evaluation workspace with
`sha256sum --strict --check docs/A4_CODE_SHA256.txt` and the third-party list.

## Injection mode of the editor

`edited(..., inject=None)` keeps the original injection `I' = I~[1 + m·A·v·g*(t − τ)]` (default).
`injection_waveform(gt, start, h0, h*, τ, injection)` returns the waveform to pass as `inject`:

* `'original'` (default): `None`, the original injection.
* `'polarity_explicit'`: `s · g*(t − d)` with `s = −1` and `d = +2` frames for every item.

The amplitude `A` is always computed from `g*(t − τ)`; chunk, target, notch, mask, colour vector,
rounding key and label are unchanged. Pool workers read `settings['injection']` (default `'original'`).
On synthetic input the default reproduces the archived editor's output bytes, and `'polarity_explicit'`
reproduces the bytes of the (ii-pc) pools.

## Random seeds

| Quantity | Generator |
|---|---|
| Paired target HR, arms (i)/(ii) | `default_rng([20261127, view, chunk, k]).uniform(70, 140)` |
| (iv) donor target | `default_rng([20261128, view, chunk, k])`: draw a donor chunk's reference HR, redraw while outside 70–140 (≤ 100 draws) |
| Matched FactorizePhys mixture targets | `default_rng([20261201, view, chunk, k])` |
| Stochastic rounding of edited frames | `default_rng([20261137, h])` (PhysNet/FactorizePhys, RhythmFormer), `default_rng([20261202, h])` (matched FactorizePhys); `h` = first 8 hex digits of SHA-256 of the item key |
| Training seeds | 20261131, 20261132, 20261133 (PhysNet five-seed strategies also 20261143, 20261144) |
| Bootstrap | `default_rng(20261110)`: subject indices `(10000, n_subjects)` then seed indices `(10000, n_seeds)` |

## Reproduction order

1. Download PURE, MMPD and (optionally) UBFC-rPPG from their providers. Set `PULSE_EDIT_ROOT` and
   `PULSE_EDIT_CACHE`, and place the rPPG-Toolbox checkout at `third_party/rPPG-Toolbox`.
2. Build the PURE training chunks (`st9g/originals.py`, `st30g/fpm_originals.py`, `st12c/rhythm_originals.py`).
3. Freeze the targets (`st9g/target_plan.py`, `st30g/prepare_A.py`) before any pool is generated.
4. Generate the pools (`st9g/pool.py`, `st30g/pool_A.py`, `st12c/rhythm_pool.py`); K = 4 items per chunk.
5. Train with the settings in `settings/` (batch: 2 originals + 2 pool items; fixed step counts:
   PhysNet 5,580, RhythmFormer 4,470, matched FactorizePhys 1,490; last epoch used).
6. Evaluate on the MMPD-A validation windows in `lists/` with the HR readout of `src/eval/grid.py`
   and the bootstrap in `evaluation/`.

Stage-5 test run and later analyses (the shell runners of the operations workspace are not included):

7. Test run (amendment A4): `eval5/drive/step.sh` called as `test_a.sh`, `test_b.sh` (via
   `run_test_b_locks.sh`), `test_check.sh`, `test_c.sh`; outputs are checked against
   `docs/TEST_STAGE_{A,B,C}_OUTPUTS_SHA256.txt`.
8. Post hoc block (amendment of 2026-11-20): pools of the second set (`train/gen_p2.py`), training
   (`train/train_704.py`, `train/train_704r.py`, `train/train_pc_pearson.py`, with `train/mcc_loss.py`
   and `train/rhythm_mcc.py`; checks in `tools/*_704.py`, `tools/p2_label_check.py`,
   `tools/mcc_unit_test*.py`), guarded inference with the evaluation code of commit `19ecf5d` on the
   checkpoint index built by `tools/infer_index_704*.py`, then `tools/analysis_704.py` (settings of
   items 1–2) and `tools/analysis_704b.py p|r` (items 3 and 9).
9. VIPL-HR (amendments A7, A8): `eval5/drive/step.sh` called as `vipl_a.sh`, `vipl_b1.sh`,
   `vipl_c1.sh`, `vipl_b2.sh`, `vipl_c2.sh`; then `eval5/vipl_a8.py`.
10. Saved-output analyses (`tools/posthoc701/`, `tools/posthoc702/`, `tools/posthoc703/`), development
    diagnostics (`tools/rule*.py`, `tools/blur_*.py`, `tools/frame_spectra.py`, `tools/keyframe_*.py`,
    `tools/replicate_*.py`, `tools/label_dist_0b.py`, `tools/mcd_dev_bpp.py`), audits
    (`tools/audit_cross*.py`, `tools/export_697.py`) and figure data (`tools/vipl_fig4.py`,
    `tools/report_*_tables.py`) read the stored outputs of these steps.

The archived scripts check the SHA-256 of their committed settings and sources; files changed for
this package (see `CHANGES.csv`) therefore differ from the hashes recorded in the archived settings.

## A9 experiments: execution paths (amendment of 2026-11-26)

The code run for the experiments of the amendment of 2026-11-26 (A9) is published at the paths it had in the operations
workspace. `docs/CODE_719_SHA256.csv` gives, for each file, the SHA-256 of the file as run and of the published file;
the published files differ only in machine paths, which are read from `config/paths.json` (keys in
`config/paths.example.json`) or derived from the repository root, and in one guard string (`docs/CODE_719_DIFF.txt`).
Files published earlier are used unchanged and are named here with their paths. The shell runners of the workspace are
not included; the commands below are the calls they made, without their scheduling, GPU locks and progress guards
(`python` = main environment, `python_repro` = repro environment of `config/paths.json`). `docs/A9_RUN_SETTINGS.csv`
lists the settings recorded by each of the 84 training runs (E1 63, E4 9, E6 3, E7 9). Pool manifests, chunk lists and
prediction folders stay on the server; their keys and SHA-256 are in `docs/A9_UBFC_POOLS.txt` and the `docs/A9_*_SHA256*`
lists.

| A9 item | Step | Entry point and command |
|---|---|---|
| 1 (E1, UBFC-rPPG) | Chunk lists and pool plans (targets with key 20261159) | `python -B tools/t709b_ubfc_manifests.py` → `cache/ubfc709/plan/<model>_originals.csv`, `<model>_items.csv` |
| 1 | Frame units | `python -B tools/t709b_frame_units.py` |
| 1c | Editor check (360 windows, notch f0/2f0 or f0/2f0/f*/2f*; (ii-pc) with d = 2 as on PURE) | `OPS2_UBFC_NOTCH=source` or `target python -B audit/a2_validity_ubfc.py`, then `python -B tools/t709b_validity_report.py`; editor `audit/editor_pc.py`, gate code `vendor/st8_6ebbc9cd/src/st9e/brange.py` |
| 1b | Originals and the (i) family: (i), (i-pc) with d = 1, (i-simple), one clip per item | `OPS2_WORKERS=3 OPS2_POOL_COMPRESS=delta-zlib python -B -s train/gen_ubfc.py <PhysNet\|FPm\|RhythmFormer> originals\|family_i <out_root>`; frames read by `tools/ubfc_video_reader.py` (`python_repro`); tensors checked by `python_repro -B -s tools/ubfc_tensor_check.py <manifest> ...` |
| 1b | The (ii) family: (ii), (ii-pc) with s = −1 and d = 1, (ii-lag) | `OPS2_WORKERS=4 OPS2_POOL_COMPRESS=delta-zlib python -B -s train/gen_ubfc_ii.py <backbone> original,polarity_ubfc,lag_shift source <out_root>` (RhythmFormer in two runs: `original,polarity_ubfc`, then `lag_shift`). The injection, with d = 1 (`D_UBFC`), is passed to `edited(..., inject=...)` of `audit/editor_pc.py`; the constant `D_FRAMES = 2` of that file is the PURE value and is not used by these pools |
| 1d | Training (63 runs) | `OPS2_DATA_WORKERS=3 CUDA_VISIBLE_DEVICES=<gpu> python_repro -B -s train/train_ubfc.py <backbone> <seed> <pool manifest>\|ORIGINAL <out_dir>`: a wrapper of `train/train_704.py` (published) that reads the UBFC originals (`train/ubfc_tensor.py`) |
| 1, 7 | Inference | Index: `python -B -s tools/infer_index_711.py <backbone> <dir>`. MMPD-B and the MCD-rPPG test half: `python_repro -B -s eval5/stage2_predict.py --dataset MMPD\|MCD ... --unseal --amendment-commit <A9 commit> --manifest <dir>/manifest.csv --checkpoint-index <dir>/checkpoint_index.csv --set exploratory --gpu <gpu> --out <dir>/<pass>` (the evaluation code of commit `19ecf5d`, compared by SHA-256 before each pass). VIPL-HR: `python_repro -B -s eval5/vipl.py b ... --manifest <dir>/manifest.csv --checkpoint-index <dir>/checkpoint_index.csv --set exploratory` (A7 code). Each pass is repeated for arm (ii-pc), seed 20261131 (`--arms '(ii-pc-ubfc)' --seeds 20261131`) and the two prediction folders are compared file by file |
| 1e, 9 | Analysis | `python -B -s tools/a9_eval/a9_groups.py vipl_readouts\|test_check\|analyze E1_<PhysNet\|FPm\|RhythmFormer>`; readouts of the PhysNet MCD-rPPG repeat passes `tools/a9_eval/repeat_readouts_717.py`; phase table of the (ii) items `tools/a9_eval/ubfc_phase_715.py`; report tables `tools/a9_eval/report_tables.py <run dir>` |
| 2, 3 (E2, E3) | Inference, analysis | Passes as above, on checkpoint indexes of the stored E2 and E3 checkpoints (their SHA-256 in `docs/A6_CHECKPOINTS_SHA256.csv` and A9 item 3; the tool that wrote these two indexes is not in the workspace); `python -B -s tools/a9_eval/a9_e2e3.py vipl_readouts\|test_check\|analyze`; values of the stored October predictions `tools/a9_eval/a9_e3_october.py` |
| 4 (E4) | Training (9 runs) | `python_repro -B -s train/train_e4.py <backbone> <seed> <out_dir>`: a wrapper of `train/train_704.py`; the two pool slots of each batch are filled with original chunks, drawn with probability 40/70 from the chunks of reference HR ≥ 100 bpm |
| 5 (E6) | Training (3 runs) | `OPS2_ARM=ii python_repro -B -s train/train_704.py RhythmFormer <seed> <second-pool (ii-pc) manifest> <out_dir>`; pool of `train/gen_p2.py` (both published) |
| 6 (E7) | Pools and training (9 runs) | `OPS2_WORKERS=<n> OPS2_POOL_COMPRESS=delta-zlib python -B -s train/gen_pool_frac.py FPm lag_frac_0.25,lag_frac_0.5,lag_frac_0.75 <out_root>`: the items, contexts, targets, notch, masks and amplitudes of the first FactorizePhys (ii-pc) pool, with the injection −g*(t − d − α(τ + T*/2 − d)) of `audit/editor_frac.py`; training `OPS2_ARM=ii python_repro -B -s train/train_704.py FPm <seed> <pool manifest> <out_dir>` |
| 4–7 | Index, inference, analysis | `python -B -s tools/infer_index_709.py E467 <dir>`; passes as above; `python -B -s tools/a9_eval/a9_groups.py vipl_readouts\|test_check\|analyze E467` |
| 8 | Stored-output analyses | `python -B -s tools/a9_8/vipl_bvp.py` (BVP records for 8c and 8g), then `python -B -s tools/a9_8/<a9_8ae\|a9_8b\|a9_8c\|a9_8d\|a9_8f\|a9_8g\|a9_8g_rec\|a9_8h>.py` (shared code `tools/a9_8/a9_common.py`) |

Aggregate results of these steps: `results/a9/` (see `results/a9/README.md`).

## Licence and terms

Code: MIT (`LICENSE`), except third-party code (everything under `vendor/` and the files listed in `LICENSE`), which keeps
the licences of its sources. Results and documents (`results/`, `docs/`, this README): CC BY 4.0.

Terms. "The operator" (`docs/DEVIATIONS.md`, `docs/TEST_NOTES.md`) and "OPS2" (file names, `docs/`, `results/MANIFEST.csv`)
denote the role that ran the post hoc experiments and the test run under the authors' written instructions; "OPS2 report
7.07c" and similar cite the operator's dated internal reports to the authors, which are not part of this repository.
