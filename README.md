# Pulse-rate editing — code and lists for reproduction (draft)

Draft public package for the manuscript "pulse-rate editing" (IEEE Access submission). It contains
the generation, training and evaluation code that produced the reported models, as archived at the
project's commit `6ebbc9cd`, plus the evaluation bootstrap and the MMPD subject / validation-window
lists. It is a local draft: it has not been published.

Not included: MMPD-B and MCD-rPPG window lists, predictions or reference signals; any face-pixel
derivative (crops, frames, edited video); model checkpoints; server paths and credentials. Data must
be obtained from the dataset providers under their licences.

## Layout

| Path | Contents |
|---|---|
| `src/st9e/` | Pulse editor (`editor.py`: notch, face mask, calibration, `edited`), time resampling (`generator.py`, `temporal.py`) and signal helpers |
| `src/st9g/` | PURE training chunks, target HR plan (`target_plan.py`) and pools for PhysNet / FactorizePhys: (i), (ii), (iv) |
| `src/st30g/` | Matched FactorizePhys chunks, targets and pools |
| `src/st12c/` | RhythmFormer chunks, pools and training |
| `src/st9t/`, `src/st30t/` | PhysNet and matched FactorizePhys training |
| `src/eval/`, `src/st11x/` | HR readout (`grid._prepare` + periodogram peak, bands HAR 0.6–3.3 Hz and L0 0.75–2.5 Hz; 300-sample windows) |
| `src/st5c/`, `src/st3/c4.py` | October MCD-rPPG window construction (`mcd_align.aligned_windows`: frame-to-PPG mapping, complete 10-s windows, gap rule, reference estimates) and MMPD signal-defect map (`c4.defect_map`), unchanged from `6ebbc9cd`. They import project modules that are not in this package: `src.st3.base`, `src.st3.c4_description`, `src.st3.media`, `src.st3.progress`, `src.st3.r3`, `src.st3.waves`, `src.st5c.avi_packets`, `src.st5c.base`, `src.st5c.signals`. `src.st5c.signals` is identical to the packaged `src/st9e/signals.py` except for its import line |
| `docs/` | Stage-5 pre-registration (`PREREGISTRATION_STAGE5.md`) and its manifest of checkpoint and MCD-rPPG subject-list SHA-256 |
| `src/st9/ops.py` | Public stand-in for the project's operational module (paths from environment variables, `GRID`, `BANDS`, guards) |
| `evaluation/` | Subject × seed crossed bootstrap (10,000 replicates, `default_rng(20261110)`) and the metric functions |
| `settings/` | Pool and training settings of the archived runs (server paths replaced by placeholders) |
| `lists/` | MMPD A/B subject lists; MMPD-A validation windows (`in_Qstar`, `high` flags; reference HR is recomputed by the code, not distributed) |
| `CHANGES.csv` | Every change relative to the archived sources |

Third-party code (rPPG-Toolbox at commit `b7500b8`, model definitions, the RhythmFormer author loss)
is not redistributed; obtain it from its authors at the stated commit.

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

The archived scripts check the SHA-256 of their committed settings and sources; files changed for
this package (see `CHANGES.csv`) therefore differ from the hashes recorded in the archived settings.
