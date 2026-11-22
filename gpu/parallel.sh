#!/bin/bash
# SUPERSEDED (2026-11-21) by multitask.sh and MULTITASK.md; the ops review advises against the 8-GPU node.
# For a VM with 8 GPUs: start 8 jobs at once, one per GPU, then write runs/parallel.done when all have ended.
# Every job saves its results and model in ~/hackathon/runs/<job>/ and is skipped if already complete, so a restart
# is safe. Leftover jobs of overnight.sh can run later on one GPU: overnight.sh skips the jobs that are complete.
# Start it detached, so keys in a terminal cannot stop it:
#   (setsid nohup ~/hackathon/gpu/parallel.sh > ~/hackathon/runs/parallel.log 2>&1 < /dev/null &)
# Watch it:  ~/hackathon/gpu/status.sh
# Stop it:   pkill -f parallel.sh; pkill -f pilot_temporal.py; pkill -f finetune_kernelsight.py

cd ~/hackathon/flamingo || exit 1
# NOTE: a second job would put two 47 GB runs on one 80 GB GPU, and both crash
if pgrep -f 'python.*(multitask|finetune_kernelsight|pilot_temporal|run_har)\.py' > /dev/null; then
  echo "not started: a job already runs"; exit 1
fi
PY=.venv/bin/python
RUNS=../runs
rm -f $RUNS/parallel.done

# gpu | job | key that a complete metrics.json has | script and arguments
JOBS=(
  "0|pilot5_trend|shuffled_trend_pairs|pilot_temporal.py --train 0 --validation 0 --trend 4000 --before"
  "1|pilot5_trend_seed1|shuffled_trend_pairs|pilot_temporal.py --train 0 --validation 0 --trend 4000 --seed 1"
  "2|class_seed1|after_shuffled|finetune_kernelsight.py --data ../kernelsight.npz --seed 1 --skip-before"
  "3|class_seed2|after_shuffled|finetune_kernelsight.py --data ../kernelsight.npz --seed 2 --skip-before"
  "4|class_seed3|after_shuffled|finetune_kernelsight.py --data ../kernelsight.npz --seed 3 --skip-before"
  "5|class_seed4|after_shuffled|finetune_kernelsight.py --data ../kernelsight.npz --seed 4 --skip-before"
  "6|pilot3ab_seed1|shuffled|pilot_temporal.py --peak 2000 --segments --seed 1"
  "7|pilot3ab_seed2|shuffled|pilot_temporal.py --peak 2000 --segments --seed 2"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r gpu name done_key command <<< "$job"
  out=$RUNS/$name
  if grep -q "\"$done_key\"" "$out/metrics.json" 2>/dev/null; then echo "$(date -u +%H:%M) skip $name (complete)"; continue; fi
  mkdir -p "$out"
  echo "$(date -u +%H:%M) start $name on GPU $gpu"
  # shellcheck disable=SC2086  # NOTE: the command is split into words on purpose
  (CUDA_VISIBLE_DEVICES=$gpu $PY -u $command --out "$out" > "$out/train.log" 2>&1;
   code=$?; echo "$(date -u +%H:%M) end $name (exit $code)") &
done
wait
echo "$(date -u +%H:%M) all jobs ended"
date -u > $RUNS/parallel.done
