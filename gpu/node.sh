#!/bin/bash
# One multi-GPU node, every lane at once, one per GPU, all ending at about the same time, so no GPU idles while the
# node bills. Plan and costs: gpu/MULTITASK.md, "One node".
#   ssh <host> "(setsid nohup bash -c 'bash ~/hackathon/gpu/setup_vm.sh > ~/setup.log 2>&1 && ~/hackathon/gpu/node.sh 4' \
#     < /dev/null > /dev/null 2>&1 &)"
#   4 GPUs  0 mt_all (seed 0, base model too)  1 mt_all_seed1  2 mt_all_attend (answers read all 24 counters)
#           3 mt_smoke, mt_class, mt_diag, then the heavy controls of mt_all from its saved model
#   8 GPUs  0-4 seeds 0-4, 5 mt_all_attend, 6 smoke + mt_class + heavy controls of mt_all, 7 mt_diag + heavy of attend
# Light controls = what every run needs (real answers; temporal reversed for the pair score). Heavy = the other
# controls, needed once per recipe. A failed smoke test stops every lane.
# Stop: pkill -f gpu/node.sh; pkill -f multitask.py
cd ~/hackathon/flamingo || exit 1
export PATH="$HOME/.local/bin:$PATH"
PY=${PY:-.venv/bin/python}
GPUS=${1:-4}
RUNS=../runs
LOG=$RUNS/node8.log
log() { echo "$(date -u +%H:%M) $*" >> $LOG; }

if pgrep -f 'python.*multitask\.py' > /dev/null; then echo "not started: a job already runs"; exit 1; fi
rm -f $RUNS/multitask.done $RUNS/node8.stop
if ls "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--opentime--tslm-nov26/snapshots/*/model.safetensors > /dev/null 2>&1; then
  export HF_HUB_OFFLINE=1
fi

BASE="--tasks diagnosis,temporal,classification,segmentation --add-train temporal/train_reversed.jsonl:trend
  --val-rows classification=300,temporal=300,segmentation=150,diagnosis=100"
LIGHT="--controls diagnosis=real,temporal=real+reversed,classification=real,segmentation=real
  --val-controls diagnosis=real,temporal=real+reversed,classification=real,segmentation=real"
HEAVY="--controls diagnosis=shuffled,temporal=shuffled,classification=shuffled+swapped,segmentation=shuffled+swapped
  --val-controls temporal=shuffled,classification=swapped"
FULL="--controls diagnosis=real+shuffled,temporal=real+shuffled+reversed,classification=real+shuffled+swapped,segmentation=real+shuffled+swapped
  --val-controls diagnosis=real,temporal=real+shuffled+reversed,classification=real+swapped,segmentation=real"

job() {  # job <gpu> <name> <logname> <args...>: one multitask.py run on one GPU
  local gpu=$1 name=$2 logname=$3; shift 3
  mkdir -p "$RUNS/$name"; log "start $name on GPU $gpu"
  # shellcheck disable=SC2068
  CUDA_VISIBLE_DEVICES=$gpu $PY -u multitask.py $@ --out "$RUNS/$name" > "$RUNS/$name/$logname.log" 2>&1
  local code=$?  # NOTE: read it first: $(date) in log would reset it
  log "end $name (exit $code)"
  [ $code -eq 0 ] && date -u > "$RUNS/$name/$logname.done"
  return $code
}
smoke() {  # the whole code path on 4 rows per task; a failure stops every lane
  job $1 mt_smoke train $BASE --limit 4 --epochs 1 --before 4 \
    || { log "stop: the smoke test failed; every lane stops"; touch $RUNS/node8.stop; pkill -f 'python.*multitask\.py'; exit 1; }
}
saved() {  # wait until <name> has saved its model (its first scored file appears after the save) or has ended
  until [ -s "$RUNS/$1/predictions_diagnosis_validation_real.jsonl" ] || grep -q "end $1 " $LOG 2>/dev/null; do sleep 60; done
  [ -d "$RUNS/$1/model" ]
}

# shellcheck disable=SC2086
case $GPUS in
  4)  # GPU 0-2: main recipe seeds 0-1, read-all-counters variant; GPU 3: smoke, references, heavy controls of mt_all
    job 0 mt_all train $BASE $LIGHT --epochs 2 --before 20 --save &
    job 1 mt_all_seed1 train $BASE $LIGHT --epochs 2 --seed 1 --save &
    job 2 mt_all_attend train $BASE $LIGHT --epochs 2 --attend-all --save &
    ( smoke 3
      job 3 mt_class train --tasks classification --epochs 2 --val-controls real,swapped
      job 3 mt_diag train --tasks diagnosis --epochs 2 --val-rows diagnosis=100 \
        --controls diagnosis=real+shuffled --val-controls diagnosis=real
      saved mt_all && job 3 mt_all heavy $BASE $HEAVY --save ) &
    wait ;;
  8)  # GPU 0-4: seeds 0-4; GPU 5: read-all-counters; GPU 6-7: smoke, references, heavy controls of both recipes
    job 0 mt_all train $BASE $LIGHT --epochs 2 --before 20 --save &
    for s in 1 2 3 4; do job $s mt_all_seed$s train $BASE $LIGHT --epochs 2 --seed $s --save & done
    job 5 mt_all_attend train $BASE $LIGHT --epochs 2 --attend-all --save &
    ( smoke 6
      job 6 mt_class train --tasks classification --epochs 2 --val-controls real,swapped
      saved mt_all && job 6 mt_all heavy $BASE $HEAVY --save ) &
    ( job 7 mt_diag train --tasks diagnosis --epochs 2 --val-rows diagnosis=100 \
        --controls diagnosis=real+shuffled --val-controls diagnosis=real
      saved mt_all_attend && job 7 mt_all_attend heavy $BASE $HEAVY --attend-all --save ) &
    wait ;;
  *) echo "usage: node.sh 4|8"; exit 1 ;;
esac
if [ -f $RUNS/node8.stop ]; then
  log "stopped: read runs/mt_smoke/train.log, fix, delete runs/node8.stop and start again"
  date -u > $RUNS/multitask.done; exit 1
fi

# NOTE: two processes wrote mt_all (and mt_all_attend) at once; a rerun with every file complete loads no model and
# only scores again, so metrics.json then holds all controls
# shellcheck disable=SC2086
specs=("mt_all --before 20"); [ "$GPUS" = 8 ] && specs+=("mt_all_attend --attend-all")
for spec in "${specs[@]}"; do
  set -- $spec; name=$1; shift
  CUDA_VISIBLE_DEVICES=0 $PY -u multitask.py $BASE $FULL "$@" --save --out "$RUNS/$name" > "$RUNS/$name/rescore.log" 2>&1
  log "rescore $name (exit $?)"
done
log "node done: delete the node in the console"
date -u > $RUNS/multitask.done
