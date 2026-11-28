# A10 results

Aggregate results of the post hoc experiment B1 of amendment A10 (2026-11-27, `docs/PREREGISTRATION_STAGE5_AMENDMENTS.md`; keys, SHA-256 and design table of the training sets in `docs/A10_UBFC_B1.txt`). Training: UBFC-rPPG chunks below 100 bpm (U-low; 31 subjects) with pool targets drawn from 100-140 bpm in the arms (0), (i-simple), (i) and (i-pc), and arm (0) on U-full-m (as many chunks as U-low, from the same 31 subjects), three backbones, seeds 20261131-20261133; heart rates of the training data in the frame units of item 1 of the entry of 2026-11-26. Every file is an aggregate CSV attached to the A10 report, byte for byte; no value was recomputed for this folder (`results/MANIFEST.csv` gives bytes and SHA-256 and, in column `note`, the derivation of each file). The arms, contrasts, descriptions and ratio tables join the analysis outputs of the three backbones and were read and written again for the report, so some numbers differ from those outputs in the last digit (largest difference 7.1e-15). Pool manifests, chunk lists and prediction folders stay on the server.

Intervals: 95% percentile intervals of the crossed bootstrap over subjects and seeds (B = 10,000, seed 20261110), seeds 20261131-20261133 paired by index, also between the U-low and U-full-m runs; no tiers and no correction for multiplicity (A10 item 9). Contrasts: point = front arm minus back arm; MAE in bpm. Test sets: MMPD-B (Q*), the MCD-rPPG test half (draft rule) and the VIPL-HR primary set (wide band, HAR).

Recovery ratio (A10 item 7d): reported with its percentile interval only in the cells whose denominator interval lies below 0; in the other cells the numerator and the denominator are reported without the ratio. For PhysNet on MMPD-B, 2 of the 10,000 replicates have a denominator of exactly 0 (ratio -inf); they are kept in the percentile interval, and leaving them out gives the same interval.

Expected share of labels at or above 100 bpm in a training batch (`docs/A10_UBFC_B1.txt`): augmented arms (2 originals of U-low + 2 pool items) 0.50; U-full-m (0) 0.344 (PhysNet list) and 0.367 (RhythmFormer and FactorizePhys list); U-low (0) 0.

Hash lists: the SHA-256 of the prediction files are in `docs/A10_B1_*_SHA256.txt` (backbone x test set, nine lists, committed before the analyses), those of the 45 checkpoints in `docs/A10_CHECKPOINTS_SHA256.csv` (committed before any evaluation). Repeat inference: for each backbone, the checkpoint (i-pc), seed 20261131 was inferred twice on each test set; the prediction files were identical in 9 of 9 backbone x test-set cells (`B1_determinism.csv`).

A10 items: 5 (training), 6 (evaluation), 7 (contrasts and recovery ratio), 8 (descriptions and summary sentences), 9 (statistics). Code: `tools/a10/` (README.md of the repository, section A10).

| File | Content |
|---|---|
| `B1_arms.csv` | arms (0-b1low), (i-simple-b1low), (i-b1low) and (i-pc-b1low) on U-low and (0-b1fullm) on U-full-m, per backbone and test set: high-HR, low-HR and all-window MAE, high-HR hit rate, signed errors and the other measures of the test run (A10 items 5, 6) |
| `B1_contrasts.csv` | the contrasts of A10 item 7, all measures: C3 = (i-pc) - (i), (i) - (i-simple), (i-simple) - (0), (i-pc) - (0) on U-low, and (0, U-full-m) - (0, U-low) |
| `B1_descriptions.csv` | the high-HR MAE contrasts with the fixed descriptions of A10 item 8a-8d |
| `B1_ratio.csv` | the recovery ratio of A10 item 7d: numerator, denominator and their intervals; the ratio and its percentile interval where the denominator interval lies below 0 (column denominator_interval_below_0); the check against the bootstrap function |
| `B1_ratio_parts_vs_contrasts.csv` | numerator and denominator of the ratio against the separately computed contrasts |
| `B1_summary_sentences.csv` | the summary sentences of A10 item 8e |
| `B1_determinism.csv` | repeat inference (A10 item 6b): files and identical files of the checkpoint inferred twice, per backbone and test set |
| `checks.csv` | checks of the report tables: windows equal over the 15 arm-backbone cells of each test set, ratio numerator and denominator equal to the bootstrap function and to the separate contrasts, repeat files identical |
