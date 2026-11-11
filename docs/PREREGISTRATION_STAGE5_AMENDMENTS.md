# Amendments to the stage-5 pre-registration

Each entry is dated and committed before any test data are opened. The original document, the original manifest and earlier entries are not rewritten.

- 2026-11-11: Added docs/PREREGISTRATION_STAGE5_MANIFEST_A1.csv with the SHA-256 of the two RhythmFormer (i-pc) checkpoints (seeds 20261132 and 20261133) that finished training after the original commit.
- 2026-11-11: Added the October modules imported by the code added in commit dbb4535 (src.st3.c4_description, src.st3.media, src.st3.progress, src.st3.r3, src.st3.waves, src.st5c.avi_packets, src.st5c.base, src.st5c.signals), unchanged from commit 6ebbc9cd. The evaluation is not changed.
- 2026-11-11: Clarification of Section 5. Both key-frame exclusion rules are applied to the standard HR readout of Section 3, and only the listed bins are set to zero. The October code for this analysis also removed the window mean before the periodogram. This is not part of either rule; its effect on the development half is reported in the paper.
