#!/bin/bash
# Stage-5 test steps (a), (b), (c) without command-line arguments (ACCESS_OPS2_BRIEF_02 6.67 다 1, 6.68 가-나).
# One file for the test and for its rehearsal on the development data; the name it is called by selects the run:
#   test_a.sh / test_b.sh / test_c.sh                  MMPD-B and the MCD-rPPG test half (sealed; see the seal check below)
#   rehearsal_a.sh / rehearsal_b.sh / rehearsal_c.sh   MMPD-A and the MCD-rPPG development half (rehearsal part 2)
#   test_check.sh / rehearsal_check.sh                 print the fixed arguments and list missing inputs (test: after the
#                                                      seal check only)
# Every argument of the evaluation code is fixed below (including the MMPD-B fixed population check, 435 recordings /
# 22 subjects). Machine-specific locations are read from files beside this one that are not committed (server paths):
#   local_paths.sh            Python environments, vendor tree, CPU set, unsealing marker, public-repository checkout
#   local_paths_rehearsal.sh  development data roots and output root (read only by the rehearsal)
#   local_paths_test.sh       test data roots, test-only inputs and output root (read only after the seal check passes)
# Seal check (6.68 가), before anything of the test is read: (1) the unsealing marker exists; OPS2 writes it by hand, outside
# the repository, only after the PI's unsealing instruction, with the time of that instruction and the A4 commit; (2) the A4
# commit is on the remote main (git ls-remote of the public repository equals the local main, and the A4 commit is one of
# its ancestors); (3) the PI's OpenTimestamps proof of the A4 commit is on main (docs/timestamps/A4_<7 hex>.txt.ots; 6.70 라 7).
# If any fails the run ends without opening anything. The unsealing arguments of the evaluation code are
# formed only after both pass. The rehearsal never reads local_paths_test.sh and names no test location.
# Inputs that exist for the test only after unsealing (6.69 가): step (a) makes or records them, step (b) checks them
# against that record before use (the rehearsal uses the development counterparts in the same places):
#   MMPD input plan       copied unchanged from its source (test: the October MMPD-B path's plan files; rehearsal: the
#                         MMPD-A plan), never remade; SHA-256 recorded at (a)
#   MCD-rPPG recordings   made at (a) by eval5/mcd_recording_list.py from the half's subject list in the manifest, with the
#                         code of the development half (front webcam, confirmed frame-to-PPG correspondence); number of
#                         recordings and SHA-256 recorded at (a)
#   A4 manifest           written at A4 packaging (SHA-256 of the 48 exploratory checkpoints); step (b) reads the original,
#                         A1 and A4 manifests (rehearsal: the draft A4 list of tools/rehearsal_index.py)
#   October predictions   of the original arms (MMPD and MCD-rPPG); only their SHA-256 is recorded at (a); they are read
#                         only by the comparison between (b) and (c)
# 1% rule (6.70 다): if the (b)-(c) comparison finds, for any checkpoint of either dataset, more than 1% of its windows with a
# readout different from the October prediction, it writes october_stop.json; step (c) then does not start until the cause
# is found and reported and, after the command's instruction to proceed, OPS2 writes the release note october_stop_release.txt
# beside it (lines instruction=<number>, instruction_time=<time>, cause=<summary>; 6.71 나).
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
W=$(cd "$HERE/../.." && pwd)                       # repository root (evaluation code in $W/eval5 here)
DOCS=$W/docs; [ -d "$DOCS" ] || DOCS=$W/pub/docs   # committed manifests (public layout / workspace layout)
EVAL=$HERE/..
NAME=$(basename "$0" .sh)
MODE=${NAME%_*}; STEP=${NAME##*_}
[ -f "$HERE/local_paths.sh" ] || { echo "local_paths.sh missing"; exit 2; }
. "$HERE/local_paths.sh"   # PY_REPRO PY_MAIN VENDOR CPUS UNSEAL_MARKER REPO REPO_REMOTE
export CUDA_VISIBLE_DEVICES=   # CPU steps see no GPU (as stage 2's workers); stage 2 gives its inference server the GPU explicitly

seal_check() {
  [ -f "$UNSEAL_MARKER" ] || { echo "sealed: no unsealing marker; nothing opened"; exit 2; }
  local t a4 remote local_head
  t=$(sed -n 's/^pi_unsealing_instruction=//p' "$UNSEAL_MARKER")
  a4=$(sed -n 's/^a4_commit=\([0-9a-f]\{40\}\)$/\1/p' "$UNSEAL_MARKER")
  [ -n "$t" ] && [ -n "$a4" ] || { echo "sealed: unsealing marker incomplete (instruction time, A4 commit); nothing opened"; exit 2; }
  remote=$(git -C "$REPO" ls-remote "$REPO_REMOTE" refs/heads/main 2>/dev/null | cut -f1)
  local_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)
  [ -n "$remote" ] && [ "$remote" = "$local_head" ] || { echo "sealed: remote main ($remote) is not the local main ($local_head); nothing opened"; exit 2; }
  git -C "$REPO" merge-base --is-ancestor "$a4" "$local_head" 2>/dev/null || { echo "sealed: A4 commit $a4 is not on main; nothing opened"; exit 2; }
  # 6.70 라 7: the PI's OpenTimestamps proof of the A4 commit is committed on main (docs/timestamps/A4_<7 hex>.txt.ots, as A3)
  git -C "$REPO" cat-file -e "HEAD:docs/timestamps/A4_${a4:0:7}.txt.ots" 2>/dev/null || { echo "sealed: no OpenTimestamps proof docs/timestamps/A4_${a4:0:7}.txt.ots on main; nothing opened"; exit 2; }
  A4_COMMIT=$a4
  echo "seal check passed: marker (PI unsealing instruction $t), A4 commit $a4 on remote main $remote"
}

# ------------------------------------------------------------------------------------------------ fixed arguments
CONF_MANIFESTS="$DOCS/PREREGISTRATION_STAGE5_MANIFEST.csv $DOCS/PREREGISTRATION_STAGE5_MANIFEST_A1.csv"
SEALED="$W/outputs/rehearsal/sealed_mmpd_b_subjects.txt $W/outputs/rehearsal/sealed_mcd_test_subjects.txt"  # ID lists for the seal guard of the code
GPU_MMPD=1; GPU_MCD=0          # fixed GPU per dataset pass (every checkpoint of a dataset on the same GPU, as in rehearsal part 1)
WORKERS_A=2; WORKERS_B=3; WORKERS_C=3
BLUR=1.5                        # RhythmFormer MCD-rPPG line-removal sensitivity (stage 2 --blur)
case $MODE in
  rehearsal)
    [ -f "$HERE/local_paths_rehearsal.sh" ] || { echo "local_paths_rehearsal.sh missing"; exit 2; }
    . "$HERE/local_paths_rehearsal.sh"   # MMPD_ROOT MCD_ROOT OUT REHEARSAL_STAGE1
    MMPD_SUBJECTS=$W/outputs/rehearsal/mmpd_a_subjects.txt
    MMPD_PLAN_SOURCE=$VENDOR/outputs; MMPD_PLAN_PREFIX=st30e_A       # the MMPD-A plan (draft input plan of the development data)
    MMPD_EXPECTED_RECORDINGS=220; MMPD_EXPECTED_SUBJECTS=11
    MCD_SUBJECTS=$W/outputs/rehearsal/mcd_dev_subjects.txt; MCD_LIST_ROW=mcd_dev_subjects
    EXPL_MANIFEST=$OUT/manifest_A4_draft.csv                         # written at step (b) by tools/rehearsal_index.py
    OCT_MMPD=$W/outputs/rehearsal/october_mmpd_a_predictions.csv     # stored October MMPD-A predictions, original arms
    OCT_MCD=$W/outputs/rehearsal/october_mcd_dev_predictions.csv      # stand-in: stored development readouts, original arms
    CONSTANTS=$OUT/constants
    UNSEAL=()
    ;;
  test)
    seal_check
    [ -f "$HERE/local_paths_test.sh" ] || { echo "local_paths_test.sh missing"; exit 2; }
    . "$HERE/local_paths_test.sh"   # MMPD_ROOT MCD_ROOT OUT DEV_CONSTANTS and, after unsealing: MMPD_PLAN_SOURCE
                                    # MMPD_PLAN_PREFIX (October MMPD-B plan), OCT_MMPD OCT_MCD (October test predictions)
    MMPD_SUBJECTS=$W/outputs/rehearsal/sealed_mmpd_b_subjects.txt
    MMPD_EXPECTED_RECORDINGS=435; MMPD_EXPECTED_SUBJECTS=22         # src/st11x/future.py mmpd_b_references (commit 6ebbc9cd)
    MCD_SUBJECTS=$W/outputs/rehearsal/sealed_mcd_test_subjects.txt; MCD_LIST_ROW=mcd_test_subjects
    EXPL_MANIFEST=$DOCS/PREREGISTRATION_STAGE5_MANIFEST_A4.csv
    CONSTANTS=$DEV_CONSTANTS                                        # constant predictors from the development data
    UNSEAL=(--unseal --amendment-commit "$A4_COMMIT" --repo "$REPO")
    ;;
  *) echo "call as test_<a|b|c|check>.sh or rehearsal_<a|b|c|check>.sh"; exit 2 ;;
esac
INPUTS=$OUT/inputs                       # step-(a) inputs of this run
MMPD_PLAN_DIR=$INPUTS                    # the copied plan
MCD_RECORDINGS=$INPUTS/mcd_recording_list.csv
VIEWS="FPm PhysNet RhythmFormer"
MISSING=""
need=("$MMPD_SUBJECTS" "$MCD_SUBJECTS" $CONF_MANIFESTS "${OCT_MMPD:-<October MMPD predictions: after unsealing>}" "${OCT_MCD:-<October MCD predictions: after unsealing>}")
for v in $VIEWS; do need+=("${MMPD_PLAN_SOURCE:-<MMPD plan source: after unsealing>}/${MMPD_PLAN_PREFIX:-}_${v}_input_plan.csv"); done
[ $MODE = test ] && need+=("$EXPL_MANIFEST")
[ "$STEP" != a ] && [ "$STEP" != check ] && need+=("$INPUTS/step_a_inputs.sha256" "$INPUTS/step_a_october.sha256")
for f in "${need[@]}"; do [ -f "$f" ] || MISSING="$MISSING $f"; done
if [ -n "$MISSING" ]; then
  echo "missing fixed inputs:$MISSING"
  [ "$STEP" = check ] || exit 2
fi
COMMON=(--sealed-subjects $SEALED --vendor "$VENDOR" "${UNSEAL[@]}")
MMPD=(--dataset MMPD --data-root "$MMPD_ROOT" --subjects "$MMPD_SUBJECTS" --plan-dir "$MMPD_PLAN_DIR" --plan-prefix "${MMPD_PLAN_PREFIX:-}")
MCD=(--dataset MCD --data-root "$MCD_ROOT" --subjects "$MCD_SUBJECTS" --recordings "$MCD_RECORDINGS")
CONF=(--manifest $CONF_MANIFESTS --set confirmatory --fsam-off --batch1)
EXPL=(--manifest "$EXPL_MANIFEST" --set exploratory)
R="nice -n 16 taskset -c $CPUS $PY_REPRO -B -s"
M="nice -n 16 taskset -c $CPUS $PY_MAIN -B -s"
mkdir -p "$OUT"
LOG=$OUT/driver_$NAME.log
exec > >(tee -a "$LOG") 2>&1
echo "== $NAME start $(date -Is); repository commit $(git -C "$W" log -1 --format=%H 2>/dev/null)"
echo "fixed: MMPD expected $MMPD_EXPECTED_RECORDINGS/$MMPD_EXPECTED_SUBJECTS, GPU MMPD $GPU_MMPD, GPU MCD $GPU_MCD, blur $BLUR"
rc=0
run() {  # LABEL COMMAND...: run, record a nonzero exit in rc, print the exit status
  local label=$1; shift; "$@"; local e=$?; [ $e = 0 ] || rc=1; echo "$label exit $e $(date -Is)"; }
case $STEP in
  a)
    mkdir -p "$INPUTS"
    for v in $VIEWS; do  # MMPD input plan copied unchanged (never remade)
      f=${MMPD_PLAN_PREFIX}_${v}_input_plan.csv
      run "MMPD plan copy $v" cp --preserve=timestamps "$MMPD_PLAN_SOURCE/$f" "$INPUTS/$f"
      run "MMPD plan copy $v byte check" cmp "$MMPD_PLAN_SOURCE/$f" "$INPUTS/$f"
    done
    run "MCD-rPPG recording list" $M "$EVAL/mcd_recording_list.py" "$VENDOR" "$MCD_ROOT" "$MCD_SUBJECTS" "$DOCS/PREREGISTRATION_STAGE5_MANIFEST.csv" \
      $MCD_LIST_ROW "$MCD_RECORDINGS" $WORKERS_A
    ( cd "$INPUTS" && sha256sum ${MMPD_PLAN_PREFIX}_*_input_plan.csv mcd_recording_list.csv > step_a_inputs.sha256 )
    sha256sum "$OCT_MMPD" "$OCT_MCD" > "$INPUTS/step_a_october.sha256"   # October predictions: SHA-256 only, not read here
    echo "step (a) inputs: MCD-rPPG recordings $(($(wc -l < "$MCD_RECORDINGS") - 1)); hashes:"; cat "$INPUTS/step_a_inputs.sha256" "$INPUTS/step_a_october.sha256"
    run "stage 1 MMPD" $R "$EVAL/stage1_reference.py" "${MMPD[@]}" --expected-recordings $MMPD_EXPECTED_RECORDINGS \
      --expected-subjects $MMPD_EXPECTED_SUBJECTS --manifest $CONF_MANIFESTS --out "$OUT/stage1" --workers $WORKERS_A "${COMMON[@]}"
    run "stage 1 MCD" $R "$EVAL/stage1_reference.py" "${MCD[@]}" --manifest $CONF_MANIFESTS --out "$OUT/stage1" --workers $WORKERS_A "${COMMON[@]}"
    if [ $MODE = rehearsal ]; then
      run "constants" $M "$EVAL/dev_constants.py" "$OUT/stage1" "$CONSTANTS"
      for ds in MMPD MCD; do  # 6.68 나: stage 1 again with the final code: every earlier file byte-identical?
        run "stage 1 $ds against the earlier rehearsal stage 1" $M "$W/tools/compare_hash_lists.py" "$REHEARSAL_STAGE1/$ds/stage1_hashes.txt" \
          "$OUT/stage1/$ds/stage1_hashes.txt" "$OUT/stage1_vs_earlier_$ds.csv"
      done
    fi
    ;;
  b)
    run "step (a) inputs unchanged" sh -c "cd '$INPUTS' && sha256sum --quiet -c step_a_inputs.sha256 && sha256sum --quiet -c step_a_october.sha256"
    [ $rc = 0 ] || { echo "step (a) inputs changed; stop"; exit 2; }
    run "checkpoint index" $M "$W/tools/rehearsal_index.py" "$OUT"   # checkpoint_index.csv (server paths, not committed); rehearsal: manifest_A4_draft.csv
    [ -f "$EXPL_MANIFEST" ] || { echo "missing exploratory manifest: $EXPL_MANIFEST"; exit 2; }
    # the two dataset lanes run side by side, each on its fixed GPU; within a lane the passes run one after the other
    ( rc=0
      run "stage 2 MMPD confirmatory" $R "$EVAL/stage2_predict.py" "${MMPD[@]}" "${CONF[@]}" "${COMMON[@]}" --gpu $GPU_MMPD --workers $WORKERS_B --checkpoint-index "$OUT/checkpoint_index.csv" --out "$OUT/stage2_conf"
      run "stage 2 MMPD exploratory" $R "$EVAL/stage2_predict.py" "${MMPD[@]}" "${EXPL[@]}" "${COMMON[@]}" --gpu $GPU_MMPD --workers $WORKERS_B --checkpoint-index "$OUT/checkpoint_index.csv" --out "$OUT/stage2_expl"
      if [ $MODE = rehearsal ]; then  # 6.68 나: second run (seed-20261131 confirmatory checkpoints, same GPU) for byte identity
        run "stage 2 MMPD determinism" $R "$EVAL/stage2_predict.py" "${MMPD[@]}" "${CONF[@]}" --seeds 20261131 "${COMMON[@]}" --gpu $GPU_MMPD --workers $WORKERS_B --checkpoint-index "$OUT/checkpoint_index.csv" --out "$OUT/stage2_determinism"
      fi
      exit $rc ) &
    LANE_MMPD=$!
    ( rc=0
      run "stage 2 MCD confirmatory" $R "$EVAL/stage2_predict.py" "${MCD[@]}" "${CONF[@]}" --blur $BLUR "${COMMON[@]}" --gpu $GPU_MCD --workers $WORKERS_B --checkpoint-index "$OUT/checkpoint_index.csv" --out "$OUT/stage2_conf"
      run "stage 2 MCD exploratory" $R "$EVAL/stage2_predict.py" "${MCD[@]}" "${EXPL[@]}" "${COMMON[@]}" --gpu $GPU_MCD --workers $WORKERS_B --checkpoint-index "$OUT/checkpoint_index.csv" --out "$OUT/stage2_expl"
      if [ $MODE = rehearsal ]; then
        run "stage 2 MCD determinism" $R "$EVAL/stage2_predict.py" "${MCD[@]}" "${CONF[@]}" --blur $BLUR --seeds 20261131 "${COMMON[@]}" --gpu $GPU_MCD --workers $WORKERS_B --checkpoint-index "$OUT/checkpoint_index.csv" --out "$OUT/stage2_determinism"
      fi
      exit $rc ) &
    LANE_MCD=$!
    wait $LANE_MMPD || rc=1
    wait $LANE_MCD || rc=1
    if [ $MODE = rehearsal ]; then
      # 6.67 다 2: seed-20261131 checkpoints of (0) and (ii-pc) on the other GPU (the two swaps run side by side)
      run "GPU swap MMPD (on GPU $GPU_MCD)" $R "$EVAL/stage2_predict.py" "${MMPD[@]}" "${CONF[@]}" --seeds 20261131 --arms "(0),(ii-pc)" "${COMMON[@]}" \
        --gpu $GPU_MCD --workers $WORKERS_B --checkpoint-index "$OUT/checkpoint_index.csv" --out "$OUT/stage2_gpu_swap" &
      SW1=$!
      run "GPU swap MCD (on GPU $GPU_MMPD)" $R "$EVAL/stage2_predict.py" "${MCD[@]}" "${CONF[@]}" --blur $BLUR --seeds 20261131 --arms "(0),(ii-pc)" "${COMMON[@]}" \
        --gpu $GPU_MMPD --workers $WORKERS_B --checkpoint-index "$OUT/checkpoint_index.csv" --out "$OUT/stage2_gpu_swap"
      wait $SW1 || rc=1
      for ds in MMPD MCD; do
        run "determinism comparison $ds" $M "$EVAL/compare_runs.py" "$OUT/stage2_conf/$ds" "$OUT/stage2_determinism/$ds" "$OUT/determinism_$ds.csv"
        run "GPU swap comparison $ds" $M "$EVAL/compare_runs.py" "$OUT/stage2_conf/$ds" "$OUT/stage2_gpu_swap/$ds" "$OUT/gpu_swap_$ds.csv"
      done
    fi
    # between (b) and (c): hashes, validity, October comparison (MMPD as before; MCD-rPPG technical, 6.67 다 5)
    run "check MMPD" $M "$EVAL/stage3_check.py" --dataset MMPD --stage1 "$OUT/stage1" --stage2 "$OUT/stage2_conf" "$OUT/stage2_expl" \
      --october-mmpd "$OCT_MMPD" --vendor "$VENDOR" --out "$OUT/check" --workers $WORKERS_C
    run "check MCD" $M "$EVAL/stage3_check.py" --dataset MCD --stage1 "$OUT/stage1" --stage2 "$OUT/stage2_conf" "$OUT/stage2_expl" \
      --october-mcd "$OCT_MCD" --vendor "$VENDOR" --out "$OUT/check" --workers $WORKERS_C
    run "re-run of differing MCD recordings" $M "$W/tools/rerun_differing.py" --check "$OUT/check/MCD" --stage2 "$OUT/stage2_conf/MCD" --out "$OUT/rerun_MCD" -- \
      "${MCD[@]}" "${CONF[@]}" --blur $BLUR --arms "(0),(i),(ii)" "${COMMON[@]}" --gpu $GPU_MCD --workers $WORKERS_B --checkpoint-index "$OUT/checkpoint_index.csv"
    ;;
  c)
    for ds in MMPD MCD; do  # 6.70 다: 1% rule of the (b)-(c) comparison (both datasets)
      if [ -f "$OUT/check/$ds/october_stop.json" ]; then
        rel="$OUT/check/$ds/october_stop_release.txt"   # 6.71 나: written by OPS2 only after the cause report and the instruction to proceed
        if [ ! -f "$rel" ] || ! grep -q '^instruction=..*' "$rel" || ! grep -q '^instruction_time=..*' "$rel" || ! grep -q '^cause=..*' "$rel"; then
          echo "1% rule: $ds has checkpoints with more than 1% windows of different readout ($OUT/check/$ds/october_stop.json);"
          echo "step (c) not started: it starts only with a release note (instruction=, instruction_time=, cause=) written after the"
          echo "cause report and the command's instruction to proceed"
          exit 3
        fi
        echo "1% rule: $ds released ($(grep -h '^instruction=' "$rel"); $(grep -h '^instruction_time=' "$rel"))"
      fi
    done
    for run in analysis analysis_rerun; do  # 6.67 다 6: the second run must reproduce every row of the first
      run "stage 3 MMPD $run" $M "$EVAL/stage3_analyze.py" --dataset MMPD --stage1 "$OUT/stage1" --check "$OUT/check" --stage2 "$OUT/stage2_conf" "$OUT/stage2_expl" \
        --tag automatic --vendor "$VENDOR" --out "$OUT/$run" --constants "$CONSTANTS/constants_MMPD.json" --plan-dir "$MMPD_PLAN_DIR" --plan-prefix "${MMPD_PLAN_PREFIX:-}"
      run "stage 3 MCD $run" $M "$EVAL/stage3_analyze.py" --dataset MCD --stage1 "$OUT/stage1" --check "$OUT/check" --stage2 "$OUT/stage2_conf" "$OUT/stage2_expl" \
        --tag halves --vendor "$VENDOR" --out "$OUT/$run" --constants "$CONSTANTS/constants_MCD.json"
    done
    for f in MMPD/automatic MCD/halves; do
      t=$(echo $f | tr / _)
      run "row check $f" $M "$W/tools/check_rows_stage3.py" "$OUT/analysis/$f" "$OUT/row_check_$t.csv" --confirmatory $CONF_MANIFESTS --exploratory "$EXPL_MANIFEST"
      run "second-run comparison $f" $M "$W/tools/compare_stage3_runs.py" "$OUT/analysis/$f" "$OUT/analysis_rerun/$f" "$OUT/rerun_compare_$t.csv"
    done
    if [ $MODE = rehearsal ]; then
      # 6.68 나: development results reproduced (October 760 windows and the MCD draft rule against the reported values)
      run "stage 3 MMPD October 760" $M "$EVAL/stage3_analyze.py" --dataset MMPD --stage1 "$OUT/stage1" --check "$OUT/check" --stage2 "$OUT/stage2_conf" "$OUT/stage2_expl" \
        --population "$VENDOR/outputs/st30a_new_A_reference.csv" --tag october760 --vendor "$VENDOR" --out "$OUT/analysis" --constants "$CONSTANTS/constants_MMPD.json" --plan-dir "$MMPD_PLAN_DIR"
      run "reported values MMPD-A 760" $M "$EVAL/rehearsal_compare_reported.py" "$OUT/analysis/MMPD/october760" MMPD-A validation "$OUT/compare_reported_MMPD760.csv"
      run "reported values MCD-dev" $M "$EVAL/rehearsal_compare_reported.py" "$OUT/analysis/MCD/halves" MCD-dev draft_rule "$OUT/compare_reported_MCD.csv"
      # 6.68 나: one exploratory checkpoint left out -> the confirmatory results are unchanged
      K=$(awk -F, 'NR>1 && $6=="exploratory"{print $1; exit}' "$OUT/stage2_expl/MMPD/checkpoints.csv")
      run "support test check" $M "$EVAL/stage3_check.py" --dataset MMPD --stage1 "$OUT/stage1" --stage2 "$OUT/stage2_conf" "$OUT/stage2_expl" \
        --exclude "$K=support test (6.48 마 1, 6.68 나)" --vendor "$VENDOR" --out "$OUT/check_support_test" --workers $WORKERS_C
      run "support test analysis" $M "$EVAL/stage3_analyze.py" --dataset MMPD --stage1 "$OUT/stage1" --check "$OUT/check_support_test" --stage2 "$OUT/stage2_conf" "$OUT/stage2_expl" \
        --tag automatic --vendor "$VENDOR" --out "$OUT/analysis_support_test" --constants "$CONSTANTS/constants_MMPD.json" --plan-dir "$MMPD_PLAN_DIR"
      run "support test comparison" $M "$W/tools/compare_confirmatory.py" "$OUT/analysis/MMPD/automatic/confirmatory.csv" \
        "$OUT/analysis_support_test/MMPD/automatic/confirmatory.csv" "$OUT/support_test.csv"
    fi
    ;;
  check)  # print the fixed arguments; inputs checked above
    echo "MMPD: ${MMPD[*]}"; echo "MCD: ${MCD[*]}"; echo "confirmatory: ${CONF[*]}"; echo "exploratory: ${EXPL[*]}"
    echo "common: ${COMMON[*]}"; echo "October MMPD: ${OCT_MMPD:-after unsealing} ($( [ -f "${OCT_MMPD:-}" ] && echo present || echo absent))"
    echo "October MCD: ${OCT_MCD:-after unsealing} ($( [ -f "${OCT_MCD:-}" ] && echo present || echo absent))"; echo "constants: $CONSTANTS"; echo "output: $OUT"
    ;;
  *) echo "unknown step $STEP"; exit 2 ;;
esac
echo "== $NAME end $(date -Is) result $([ $rc = 0 ] && echo ok || echo 'FAILED (see above)')"
exit $rc
