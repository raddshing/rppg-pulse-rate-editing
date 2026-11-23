#!/bin/bash
# VIPL-HR registered evaluation (amendment A7, item 11), stages (a), (b1), (c1), (b2), (c2) without command-line arguments;
# the name it is called by selects the stage: vipl_a.sh / vipl_b1.sh / vipl_c1.sh / vipl_b2.sh / vipl_c2.sh. Every argument
# of the evaluation code is fixed below. (a) windows, exclusions, references, flags and the BVP rate, without any model;
# (b1) predictions of the primary (confirmatory) checkpoints; (c1) the analyses of the primary checkpoints; (b2) predictions
# of the secondary checkpoints; (c2) the remaining analyses. No code is changed between stages.
# Machine-specific locations come from files beside this one that are not committed (server paths):
#   local_paths.sh        Python environments, vendor tree, CPU set, public-repository checkout and remote
#   local_paths_vipl.sh   VIPL_ZIPS (folder of the 14 original zips), VIPL_OUT (output root), A7_COMMIT (40 hex)
# Before anything is read: (1) the A7 entry is committed on the remote main (git ls-remote of the public repository equals
# the local main, A7_COMMIT is one of its ancestors and its amendments file holds the A7 heading); (2) every code file
# matches its SHA-256 in docs/A7_CODE_SHA256.txt of that commit; (3) the zips match docs/A7_VIPL_ZIPS_SHA256.txt
# (stage (a) only: a full read of the 14 zips). If any fails, the stage ends without opening the data. The evaluation code
# repeats check (1) itself (eval5/vipl.py data_guard). Each stage writes the hashes of its outputs; OPS2 commits them
# (docs/A7_STAGE_<A|B1|C1|B2>_SHA256.txt) before the next stage starts, and each stage checks that the hash file of the
# previous stage is committed on main and equals the outputs on disk.
set -u
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
W=$(cd "$HERE/../.." && pwd)
EVAL=$HERE/..
DOCS=$W/docs; [ -d "$DOCS" ] || DOCS=$W/pub/docs   # committed manifests (public layout / workspace layout)
NAME=$(basename "$0" .sh)
STEP=${NAME##*_}
. "$HERE/local_paths.sh"        # PY_REPRO PY_MAIN VENDOR CPUS REPO REPO_REMOTE
. "$HERE/local_paths_vipl.sh"   # VIPL_ZIPS VIPL_OUT A7_COMMIT
export CUDA_VISIBLE_DEVICES=
GPU=1; WORKERS=3; B=10000
fail() { echo "$1; nothing opened"; exit 2; }
remote=$(git -C "$REPO" ls-remote "$REPO_REMOTE" refs/heads/main | cut -f1)
local_head=$(git -C "$REPO" rev-parse HEAD)
[ -n "$remote" ] && [ "$remote" = "$local_head" ] || fail "remote main ($remote) is not the local main ($local_head)"
git -C "$REPO" merge-base --is-ancestor "$A7_COMMIT" "$local_head" || fail "A7 commit $A7_COMMIT is not on main"
git -C "$REPO" show "$A7_COMMIT:docs/PREREGISTRATION_STAGE5_AMENDMENTS.md" | grep -q 'Registered evaluation on VIPL-HR\.' || fail "no A7 entry at $A7_COMMIT"
git -C "$REPO" show "$A7_COMMIT:docs/A7_CODE_SHA256.txt" > "$VIPL_OUT/A7_CODE_SHA256.txt" || fail "no code hash list at A7"
( cd "$W" && sha256sum --quiet -c "$VIPL_OUT/A7_CODE_SHA256.txt" ) || fail "a code file differs from its SHA-256 at A7"
committed() {  # hash file name in docs/ on main
  git -C "$REPO" cat-file -e "HEAD:docs/$1" 2>/dev/null || fail "docs/$1 is not committed on main"
}
R="nice -n 16 taskset -c $CPUS $PY_REPRO -B -s"
M="nice -n 16 taskset -c $CPUS $PY_MAIN -B -s"
G=(--repo "$REPO" --remote "$REPO_REMOTE" --a7-commit "$A7_COMMIT")
LOG=$VIPL_OUT/driver_$NAME.log
exec > >(tee -a "$LOG") 2>&1
echo "== $NAME start $(date -Is); A7 $A7_COMMIT; code list $(sha256sum "$VIPL_OUT/A7_CODE_SHA256.txt" | cut -c1-16)"
case $STEP in
  a)
    git -C "$REPO" show "$A7_COMMIT:docs/A7_VIPL_ZIPS_SHA256.txt" > "$VIPL_OUT/A7_VIPL_ZIPS_SHA256.txt" || fail "no zip hash list at A7"
    ( cd "$VIPL_ZIPS" && sha256sum --quiet -c "$VIPL_OUT/A7_VIPL_ZIPS_SHA256.txt" ) || fail "a zip differs from its SHA-256"
    $R "$EVAL/vipl.py" a --zip-dir "$VIPL_ZIPS" --out "$VIPL_OUT/stage_a" --vendor "$VENDOR" --workers $WORKERS "${G[@]}"
    echo "stage (a) exit $? $(date -Is); hash file: $VIPL_OUT/stage_a/VIPL/stage_a_hashes.txt" ;;
  b1|b2)
    committed A7_STAGE_A_SHA256.txt
    git -C "$REPO" show HEAD:docs/A7_STAGE_A_SHA256.txt | cmp - "$VIPL_OUT/stage_a/VIPL/stage_a_hashes.txt" || fail "stage (a) outputs differ from the committed hashes"
    if [ $STEP = b1 ]; then
      s=confirmatory; MAN=("$DOCS/PREREGISTRATION_STAGE5_MANIFEST.csv" "$DOCS/PREREGISTRATION_STAGE5_MANIFEST_A1.csv")
    else
      committed A7_STAGE_B1_SHA256.txt; committed A7_STAGE_C1_SHA256.txt
      git -C "$REPO" show HEAD:docs/A7_STAGE_C1_SHA256.txt | cmp - "$VIPL_OUT/stage_c1/VIPL/analysis_hashes.txt" || fail "stage (c1) outputs differ from the committed hashes"
      git -C "$REPO" show "$A7_COMMIT:docs/A7_SECONDARY_MANIFEST.csv" > "$VIPL_OUT/A7_SECONDARY_MANIFEST.csv" || fail "no secondary manifest at A7"
      s=exploratory; MAN=("$DOCS/PREREGISTRATION_STAGE5_MANIFEST_A4.csv" "$VIPL_OUT/A7_SECONDARY_MANIFEST.csv")
    fi
    $R "$EVAL/vipl.py" b --zip-dir "$VIPL_ZIPS" --stage1 "$VIPL_OUT/stage_a" --out "$VIPL_OUT/stage_b_$s" --manifest "${MAN[@]}" \
      --checkpoint-index "$VIPL_OUT/checkpoint_index.csv" --set $s --vendor "$VENDOR" --gpu $GPU --workers $WORKERS "${G[@]}"
    echo "stage ($STEP) exit $? $(date -Is); hash file: $VIPL_OUT/stage_b_$s/VIPL/stage2_hashes.txt" ;;
  c1|c2)
    committed A7_STAGE_B1_SHA256.txt
    git -C "$REPO" show HEAD:docs/A7_STAGE_B1_SHA256.txt | cmp - "$VIPL_OUT/stage_b_confirmatory/VIPL/stage2_hashes.txt" || fail "stage (b1) outputs differ from the committed hashes"
    if [ $STEP = c1 ]; then
      PART=primary; SB=("$VIPL_OUT/stage_b_confirmatory")
    else
      committed A7_STAGE_B2_SHA256.txt
      git -C "$REPO" show HEAD:docs/A7_STAGE_B2_SHA256.txt | cmp - "$VIPL_OUT/stage_b_exploratory/VIPL/stage2_hashes.txt" || fail "stage (b2) outputs differ from the committed hashes"
      PART=remaining; SB=("$VIPL_OUT/stage_b_confirmatory" "$VIPL_OUT/stage_b_exploratory")
    fi
    $M "$EVAL/vipl_c.py" --part $PART --stage-a "$VIPL_OUT/stage_a" --stage-b "${SB[@]}" \
      --vendor "$VENDOR" --out "$VIPL_OUT/stage_$STEP" --b $B --workers $WORKERS
    echo "stage ($STEP) exit $? $(date -Is); hash file: $VIPL_OUT/stage_$STEP/VIPL/analysis_hashes.txt" ;;
  *) echo "call as vipl_a.sh, vipl_b1.sh, vipl_c1.sh, vipl_b2.sh or vipl_c2.sh"; exit 2 ;;
esac
