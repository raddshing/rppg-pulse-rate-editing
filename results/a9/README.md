# A9 results

Aggregate results of the post hoc experiments and analyses of amendment A9 (2026-11-26, `docs/PREREGISTRATION_STAGE5_AMENDMENTS.md`). Every file is the aggregate CSV attached to the report of its experiment, byte for byte; no value was recomputed for this folder (`results/MANIFEST.csv` gives bytes and SHA-256). The arms and contrasts tables of E1 and of E4, E6 and E7 were read and written again for the reports (E4, E6 and E7 split by experiment; columns constant in a table left out), so some numbers differ from the analysis outputs in the last digit (largest difference 3.6e-15); column `note` of `results/MANIFEST.csv` gives the derivation of each file. The prediction SHA-256 lists in `docs/` were committed before the analyses. Lists per item, window or recording (pool item lists, the two excluded VIPL-HR windows of item 8a) stay on the server; the SHA-256 of the UBFC-rPPG pool manifests are in `docs/A9_UBFC_POOLS.txt`.

Intervals: 95% crossed bootstrap over subjects and seeds of the test run (B = 10,000, seed 20261110), seeds paired by index, no tiers, no correction for multiplicity. Contrasts: point = front arm minus back arm; MAE in bpm. Test sets: MMPD-B (Q*), the MCD-rPPG test half (draft rule) and the VIPL-HR primary set (wide band, HAR).

## `e1_physnet/`

A9 items: 1 (E1, PhysNet; 1c injection amplitude, 1e summary sentence over the three backbones), 7, 9 (repeat pass).

Prediction SHA-256 lists: docs/A9_E1_PhysNet_MMPD_B_SHA256.txt, docs/A9_E1_PhysNet_MCD_TEST_SHA256.txt, docs/A9_E1_PhysNet_VIPL_SHA256.txt; the two MCD-rPPG test-half repeat passes: docs/A9_E1_PhysNet_MCD_TEST_REPEAT1_SHA256.txt, docs/A9_E1_PhysNet_MCD_TEST_REPEAT2_SHA256.txt; checkpoints: docs/A9_E1_PhysNet_CHECKPOINTS_SHA256.csv.

| File | Content |
|---|---|
| `E1_PhysNet_arms.csv` | arms (0-ubfc), (i-simple-ubfc), (i-ubfc), (i-pc-ubfc), (ii-ubfc), (ii-pc-ubfc), (ii-lag-ubfc) on MMPD-B, the MCD-rPPG test half and VIPL-HR |
| `E1_PhysNet_contrasts.csv` | C1-C4, (ii-lag)-(ii-pc), (i-simple)-(0), (ii-pc)-(i-simple), all measures |
| `E1_PhysNet_descriptions.csv` | the contrasts for MAE_high, MAE_low and MAE_all beside the same contrast with PURE training (pure_point, pure_ci_lo, pure_ci_hi) and the A9 description |
| `E1_summary_sentences.csv` | A9 item 1e sentence per test set: number k of backbones whose C1 high-HR interval is below 0, and C1 of each backbone |
| `E1_injection_amplitude.csv` | injection amplitude A over all pool items of (ii-pc), PURE first pool and UBFC-rPPG pool, per backbone view (the same file in the three E1 folders) |
| `E1_UBFC_phase_error.csv` | absolute phase error at f* of the pool items and ratio r, UBFC-rPPG pools (d = 1) beside the first PURE FactorizePhys pool (d = 2) (the same file in two folders) |
| `E1_UBFC_phase_checks.csv` | checks of the phase table |
| `E1_PhysNet_MCD_test_repeat_files.csv` | repeat passes of (ii-pc-ubfc) seed 20261131 on the MCD-rPPG test half against the main pass (files, identical files, values differing, largest absolute difference) |
| `E1_PhysNet_MCD_test_repeat_readouts.csv` | step-(b) HR readouts (HAR and L0) of the repeat passes against the main pass |
| `E1_PhysNet_MCD_test_repeat_gpu_memory.csv` | GPU free memory during the repeat inferred again (1-s samples) |
| `checks.csv` | checks of the evaluation: identical prediction files of checkpoints inferred twice, step (b) of the test run reproduced on the stored readouts, VIPL-HR readout defects, and the further checks named in its rows |

## `e1_rhythmformer/`

A9 items: 1 (E1, RhythmFormer; 1c injection amplitude), 7.

Prediction SHA-256 lists: docs/A9_E1_RhythmFormer_MMPD_B_SHA256.txt, docs/A9_E1_RhythmFormer_MCD_TEST_SHA256.txt, docs/A9_E1_RhythmFormer_VIPL_SHA256.txt; checkpoints: docs/A9_E1_RhythmFormer_CHECKPOINTS_SHA256.csv.

| File | Content |
|---|---|
| `E1_RhythmFormer_arms.csv` | the same arms |
| `E1_RhythmFormer_contrasts.csv` | the same contrasts |
| `E1_RhythmFormer_descriptions.csv` | the same |
| `E1_injection_amplitude.csv` | injection amplitude A over all pool items of (ii-pc), PURE first pool and UBFC-rPPG pool, per backbone view (the same file in the three E1 folders) |
| `E1_UBFC_phase_error.csv` | absolute phase error at f* of the pool items and ratio r, UBFC-rPPG pools (d = 1) beside the first PURE FactorizePhys pool (d = 2) (the same file in two folders) |
| `E1_UBFC_phase_checks.csv` | checks of the phase table |
| `checks.csv` | checks of the evaluation: identical prediction files of checkpoints inferred twice, step (b) of the test run reproduced on the stored readouts, VIPL-HR readout defects, and the further checks named in its rows |

## `e1_factorizephys/`

A9 items: 1 (E1, FactorizePhys; 1c injection amplitude), 7.

Prediction SHA-256 lists: docs/A9_E1_FPm_MMPD_B_SHA256.txt, docs/A9_E1_FPm_MCD_TEST_SHA256.txt, docs/A9_E1_FPm_VIPL_SHA256.txt; checkpoints: docs/A9_E1_FactorizePhys_CHECKPOINTS_SHA256.csv.

| File | Content |
|---|---|
| `E1_FactorizePhys_arms.csv` | the same arms |
| `E1_FactorizePhys_contrasts.csv` | the same contrasts |
| `E1_FactorizePhys_descriptions.csv` | the same |
| `E1_injection_amplitude.csv` | injection amplitude A over all pool items of (ii-pc), PURE first pool and UBFC-rPPG pool, per backbone view (the same file in the three E1 folders) |
| `checks.csv` | checks of the evaluation: identical prediction files of checkpoints inferred twice, step (b) of the test run reproduced on the stored readouts, VIPL-HR readout defects, and the further checks named in its rows |

## `e2e3/`

A9 items: 2 (E2), 3 (E3), 7.

Prediction SHA-256 lists: docs/A9_E2E3_VIPL_SHA256.txt (E2 and E3 on VIPL-HR), docs/A9_E3_MMPD_B_SHA256.txt, docs/A9_E3_MCD_TEST_SHA256.txt.

| File | Content |
|---|---|
| `E2_P_arms.csv` | E2: arms of the new seeds 20261148-20261150 on VIPL-HR |
| `E2_P_contrasts.csv` | C1-C4 on the new seeds and on all six seeds |
| `E2_R_arms.csv` | E2: RhythmFormer with the MCC loss on VIPL-HR |
| `E2_R_contrasts.csv` | contrasts within RhythmFormer-MCC |
| `E2_R_interactions.csv` | loss interaction: contrast with the MCC loss minus the same contrast of confirmatory RhythmFormer |
| `E3_arms.csv` | E3: arm (0B) (RhythmFormer with its authors' augmentation) on the three test sets |
| `E3_contrasts.csv` | (ii-pc)-(0B), (i-simple)-(0B), (0)-(0B) |
| `E3_october_beside.csv` | E3 contrasts beside the values from the stored October predictions |
| `E3_october_beside_arms.csv` | E3 arms beside the October values |
| `checks.csv` | checks of the evaluation: identical prediction files of checkpoints inferred twice, step (b) of the test run reproduced on the stored readouts, VIPL-HR readout defects, and the further checks named in its rows |

## `e467/`

A9 items: 4 (E4), 5 (E6), 6 (E7), 7.

Prediction SHA-256 lists: docs/A9_E467_MMPD_B_SHA256.txt, docs/A9_E467_MCD_TEST_SHA256.txt, docs/A9_E467_VIPL_SHA256.txt.

| File | Content |
|---|---|
| `E4_arms.csv` | E4: arm (os), high-HR oversampling |
| `E4_contrasts.csv` | (os)-(0), (ii-pc)-(os), (i-simple)-(os) |
| `E6_arms.csv` | E6: RhythmFormer (ii-pc) on the second pool |
| `E6_contrasts.csv` | pool effect (ii-pc-pool2)-(ii-pc) and C1 on the second pool |
| `E7_arms.csv` | E7: FactorizePhys (ii-a), a = 0.25, 0.5, 0.75 |
| `E7_contrasts.csv` | (ii-a)-(ii-pc) |
| `E7_slope.csv` | slope of the high-HR MAE on a over five levels, with the A9 item 6 statement |
| `E7_phase_error.csv` | absolute phase error at f* of each level over the items of the first FactorizePhys (ii-pc) pool |
| `checks.csv` | checks of the evaluation: identical prediction files of checkpoints inferred twice, step (b) of the test run reproduced on the stored readouts, VIPL-HR readout defects, and the further checks named in its rows |

## `item8/`

A9 items: 8 (8a-8h).

Prediction SHA-256 lists: none: no new inference; item 8 analyses predictions that existed when A9 was committed (test run, post hoc block, VIPL-HR stages (c1) and (c2)).

| File | Content |
|---|---|
| `8a_contrasts.csv` | C1-C4 on VIPL-HR without the two windows of 255 bpm, beside the stage (c1) values |
| `8b_scene_contrasts.csv` | contrasts per VIPL-HR scene, primary set |
| `8b_scene_contrasts_secondary.csv` | the same, secondary set |
| `8c_snr.csv` | PhysNet arms: SNR of the predicted waveform at the reference HR |
| `8c_phase.csv` | PhysNet arms: phase of the prediction relative to the BVP |
| `8c_checks.csv` | HR read from the stored waveforms against the stored readouts |
| `8d_flow.csv` | recordings and windows removed at each exclusion step |
| `8d_windows.csv` | reference HR and window positions of included and excluded windows (aggregates) |
| `8e_agreement.csv` | windows whose mean gt_HR and 60-Hz BVP reference differ by more than 10 and 20 bpm |
| `8f_arms.csv` | subject-weighted MAE per arm |
| `8f_contrasts.csv` | subject-weighted C1-C4 |
| `8f_share.csv` | share of subjects whose high-HR MAE is lower with (ii-pc) than with (0) |
| `8f_checks.csv` | window-weighted C1-C4 on the same windows |
| `8g_arms.csv` | VIPL-HR arm-level MAE, hit rate and half-HR rate of every evaluated arm |
| `8g_checkpoints.csv` | the same per checkpoint |
| `8g_constant.csv` | oracle constant predictor (mean and median reference HR) |
| `8g_recording_arms.csv` | recording-level scores in the benchmark convention, per arm |
| `8g_recording_keys.csv` | the same per checkpoint |
| `8g_recording_constant.csv` | the same for the constant predictor |
| `8g_checks.csv` | MAE of the confirmatory arms recomputed against stage (c1) |
| `8h_harmonics.csv` | harmonic content 2f*/f* of the target waveform and 2f0/f0 of the contact PPG, first PURE (ii-pc) pool |
