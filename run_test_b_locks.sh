#!/bin/bash
# Brief 02 6.72 나 (A4 item 4: both GPUs for stage (b) "until stage (b) has ended"): step (b) of the test
# (eval5/drive/test_b.sh, unchanged) with both GPU training locks taken before the driver starts and held, with no hand-back
# between passes, until the driver has exited. tools/gpu_rounds2.py yields a GPU while a flock process holds or waits for its
# lock, so the replicates, PhysNet-Std and every other coordinator job stop for the whole step; the coordinator resumes only
# after 5 quiet minutes once both locks are given back. The driver starts only when neither GPU has a compute process.
# The rehearsal's lock supervisor (run_rehearsal_b_locks.sh, GPU1 handed back between passes) is for the rehearsal only.
# Run only after step (a) of the test, i.e. after the A4 push, the PI's OpenTimestamps proof and the PI's unsealing
# instruction; otherwise the driver's seal check ends the run before anything of the test is opened.
# Events: cache/ops/test_b_locks.log; the driver's own output before its log starts (seal check): cache/ops/test_b_driver.out.
# Usage: run_test_b_locks.sh (no arguments)
cd <MACHINE_PATH> || exit 2
export DRIVER=./eval5/drive/test_b.sh
export EV=cache/ops/test_b_locks.log
export OUTF=cache/ops/test_b_driver.out
L0=cache/ops/gpu0_train.lock; L1=cache/ops/gpu1_train.lock
ev(){ echo "$(date -Is) $*" | tee -a "$EV"; }

ev "start: test step (b), both GPUs until stage (b) has ended (6.72 나); taking the GPU1 then the GPU0 training lock"
flock "$L1" flock "$L0" bash -c '
  ev(){ echo "$(date -Is) $*" | tee -a "$EV"; }
  ev "both GPU training locks held (GPU1, GPU0)"
  n=0
  while :; do  # every compute process on either GPU (this machine has GPU0 and GPU1 only) must have ended
    apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1); s=$?
    [ $s = 0 ] && [ -z "$apps" ] && break
    [ $((n % 60)) = 0 ] && ev "waiting: nvidia-smi exit $s, compute processes: $(echo "$apps" | tr "\n" ";")"
    n=$((n + 1)); sleep 10
  done
  ev "no compute process on either GPU; driver starts"
  "$DRIVER" >> "$OUTF" 2>&1
  rc=$?
  ev "driver ended (exit $rc)"
  exit $rc'
rc=$?
ev "both GPU training locks given back; end (exit $rc)"
exit $rc
