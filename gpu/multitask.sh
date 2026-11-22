#!/bin/bash
# The multi-task queue on a 1-GPU VM (steps: gpu/README.md). One job at a time, in the
# order below. A job with runs/<job>/job.done is skipped, and multitask.py skips each finished predictions file,
# so a restart goes on where it stopped. Start it detached, so keys in a terminal cannot stop it:
#   (setsid nohup ~/hackathon/gpu/multitask.sh round1 > ~/hackathon/runs/multitask.log 2>&1 < /dev/null &)
#   round1  mt_smoke, mt_all                                   the Round-1 minimum (~2.7 h of jobs)
#   full    round1, then mt_class and mt_all_seed1               (~5.6 h of jobs)
#   har     har_cot only, if the HAR-CoT run on a laptop fails   (needs gpu/setup_har.sh first)
#   seed0 | seed1 | seed2 | attend | refs   one plan per 1-GPU VM, run in parallel (each: its own smoke test, then
#           mt_all | mt_all_seed1 | mt_all_seed2 | mt_all_attend (--attend-all) | mt_class + mt_diag)
# Watch:  ~/hackathon/gpu/progress.sh   or   ~/hackathon/gpu/status.sh   (both only read files)
# Stop:   pkill -f gpu/multitask.sh; pkill -f 'multitask.py|run_har.py'
#
#   mt_smoke      all 4 tasks, 4 rows per task and split, 1 epoch, the base model on 4 rows: the whole code path
#   mt_all        all 4 tasks, 2 epochs, seed 0, the model saved (a restart scores without training again); test:
#                 the controls each use case asked for; validation: the controls of its go/no-go rule
#   mt_class      classification only, 2 epochs, seed 0: the class-only reference in this harness, with controls
#   mt_all_seed1  mt_all with seed 1, so the multi-task recipe has 2 runs
#   har_cot       OpenTSLM's released HAR-CoT model on real, shuffled and reversed windows (usecases/explanation/har/run_har.py)
cd ~/hackathon/flamingo || exit 1
export PATH="$HOME/.local/bin:$PATH"
PY=.venv/bin/python
RUNS=../runs
PLAN=${1:?usage: multitask.sh round1|full}
MIN_FREE_GB=${MIN_FREE_GB:-20}
LOCK=$RUNS/multitask.pid
log() { echo "$(date -u +%H:%M) $*"; }

# NOTE: a second start would put two 47 GB jobs on one 80 GB GPU, and both crash
if [ -f $LOCK ] && ps -p "$(cat $LOCK)" -o args= 2>/dev/null | grep -q gpu/multitask.sh; then
  log "not started: the queue already runs (pid $(cat $LOCK))"; exit 1
fi
JOBS='python.*(multitask|finetune_kernelsight|pilot_temporal|run_har)\.py'
if pgrep -f "$JOBS" > /dev/null; then
  log "not started: a job already runs:"; pgrep -af "$JOBS"; exit 1
fi
echo $$ > $LOCK
trap 'rm -f $LOCK' EXIT  # NOTE: a stale lock is harmless: its pid no longer runs gpu/multitask.sh
rm -f $RUNS/multitask.done

# NOTE: utils.OpenTSLM.load asks the Hub in each job; offline, the cached model loads without the network
if ls "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--opentime--tslm-nov26/snapshots/*/model.safetensors > /dev/null 2>&1; then
  export HF_HUB_OFFLINE=1
else
  log "opentime/tslm-nov26 is not cached (setup_vm.sh downloads it): the first job downloads it"
fi
log "plan $PLAN, HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"

fails=0
run() {  # run <job> <command...>: adds --out runs/<job>; writes runs/<job>/job.done when the command exits 0
  local name=$1 out=$RUNS/$1; shift
  if [ -f "$out/job.done" ]; then log "skip $name (done)"; return 0; fi
  local free; free=$(df -Pk "$RUNS" | awk 'NR == 2 {print int($4 / 1048576)}')
  if [ "$free" -lt "$MIN_FREE_GB" ]; then log "stop: $free GB free, $MIN_FREE_GB needed before $name"; exit 1; fi
  mkdir -p "$out"; (cd "$out" && pwd) > $RUNS/multitask.current
  log "start $name ($free GB free)"
  "$@" --out "$out" > "$out/train.log" 2>&1
  local code=$?  # NOTE: read it first: $(date) in log would reset it to 0
  log "end $name (exit $code)"
  if [ $code -eq 0 ]; then date -u > "$out/job.done"; fails=0; else fails=$((fails + 1)); fi
  if [ $fails -ge 2 ]; then log "stop: 2 jobs in a row failed"; exit 1; fi
  return $code
}

# NOTE: diagnosis replaces explanation in the main run (the headline use case: kernel optimization). To add explanation
# back: append ",explanation" to --tasks, "explanation=100" to --val-rows, "--test-rows explanation=100", and
# "explanation=real+shuffled+reversed" / "explanation=real" to the two control lists.
ALL="--tasks diagnosis,temporal,classification,segmentation --add-train temporal/train_reversed.jsonl:trend
  --val-rows classification=300,temporal=300,segmentation=150,diagnosis=100
  --controls diagnosis=real+shuffled,temporal=real+shuffled+reversed,classification=real+shuffled+swapped+reversed,segmentation=real+shuffled+swapped+reversed
  --val-controls diagnosis=real,temporal=real+shuffled+reversed,classification=real+swapped,segmentation=real+shuffled"
# shellcheck disable=SC2086  # NOTE: $ALL is split into words on purpose
case $PLAN in
  round1|full)
    run mt_smoke $PY -u multitask.py $ALL --limit 4 --epochs 1 --before 4 || { log "stop: the smoke test failed"; exit 1; }
    run mt_all $PY -u multitask.py $ALL --epochs 2 --before 20 --save
    if [ "$PLAN" = full ]; then
      run mt_class $PY -u multitask.py --tasks classification --epochs 2 --val-controls real,swapped
      run mt_all_seed1 $PY -u multitask.py $ALL --epochs 2 --seed 1 --save
    fi ;;
  # NOTE: parallel plans, one per 1-GPU VM; each job has its own runs/<job> folder,
  # so pull.sh can copy all VMs into one runs/ folder
  seed0|seed1|seed2|attend|refs)
    run "mt_smoke_$PLAN" $PY -u multitask.py $ALL --limit 4 --epochs 1 --before 4 || { log "stop: the smoke test failed"; exit 1; }
    case $PLAN in
      seed0) run mt_all $PY -u multitask.py $ALL --epochs 2 --before 20 --save ;;
      seed1) run mt_all_seed1 $PY -u multitask.py $ALL --epochs 2 --seed 1 --save ;;
      seed2) run mt_all_seed2 $PY -u multitask.py $ALL --epochs 2 --seed 2 --save ;;
      attend) run mt_all_attend $PY -u multitask.py $ALL --epochs 2 --attend-all --save ;;
      refs)
        run mt_class $PY -u multitask.py --tasks classification --epochs 2 --val-controls real,swapped
        run mt_diag $PY -u multitask.py --tasks diagnosis --epochs 2 --val-rows diagnosis=100 \
          --controls diagnosis=real+shuffled --val-controls diagnosis=real ;;
    esac ;;
  har)
    if [ -f ../data/har_cot/har_cot_test_cot.csv ] && [ -d ../OpenTSLM/src ]; then
      run har_cot uv run ../usecases/explanation/har/run_har.py --csv ../data/har_cot/har_cot_test_cot.csv --opentslm ../OpenTSLM
    else
      log "stop: run gpu/setup_har.sh first (needs ../data/har_cot/har_cot_test_cot.csv and ../OpenTSLM)"; exit 1
    fi ;;
  *) log "unknown plan $PLAN (round1, full, seed0, seed1, seed2, attend, refs or har)"; exit 1 ;;
esac
log "queue done"
date -u > $RUNS/multitask.done
