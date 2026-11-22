#!/bin/bash
# SUPERSEDED (2026-11-21) by multitask.sh and MULTITASK.md. Kept working for the old pilot jobs.
# The overnight queue, on the GPU VM. Runs one job at a time (one H100 holds one run) and skips a job whose results are
# already complete, so a restart after a stop goes on where it ended. Every job saves its results and its model in
# ~/hackathon/runs/<job>/. Start it detached, so keys in a terminal cannot stop it:
#   (setsid nohup ~/hackathon/gpu/overnight.sh > ~/hackathon/runs/overnight.log 2>&1 < /dev/null &)
# Stop it on purpose:  pkill -f overnight.sh; kill $(cat "$(cat ~/hackathon/runs/overnight.current)/run.pid")
#
# Order: the trend pilot first (the result that matters most), then the two recipes take turns, so a short night
# still gives some seeds of each. The trend seeds run last, and only if the trend pilot works.
#   pilot5_trend       class questions plus trend pairs (forward and reversed), with the base model scored too ~50 min
#   class_seed<N>      finetune_kernelsight.py, the class task only (seed 0 = runs/flamingo)                    ~40 min
#   pilot3ab_seed<N>   pilot_temporal.py with the pilot 3ab recipe (seed 0 = runs/pilot3ab)                     ~60 min
#   pilot4_balanced    pilot 2 questions plus peak_side balanced for each range of N, one run                   ~45 min
#   pilot5_trend_seed<N>  only if pilot5_trend works (see trend_works)                                           ~50 min

cd ~/hackathon/flamingo || exit 1
# NOTE: a second job would put two 47 GB runs on one 80 GB GPU, and both crash
if pgrep -f 'python.*(multitask|finetune_kernelsight|pilot_temporal|run_har)\.py' > /dev/null; then
  echo "not started: a job already runs"; exit 1
fi
PY=.venv/bin/python
RUNS=../runs

run() {  # run <name> <key that a complete metrics.json has> <script and arguments...>
  local name=$1 done_key=$2; shift 2
  local out=$RUNS/$name
  if grep -q "\"$done_key\"" "$out/metrics.json" 2>/dev/null; then echo "$(date -u +%H:%M) skip $name (complete)"; return; fi
  mkdir -p "$out"; (cd "$out" && pwd) > $RUNS/overnight.current
  echo "$(date -u +%H:%M) start $name"
  $PY -u "$@" --out "$out" > "$out/train.log" 2>&1
  local code=$?  # NOTE: read it first: $(date) would reset it to 0
  echo "$(date -u +%H:%M) end $name (exit $code)"
}

trend_works() {  # pair score at least 0.10 above the shuffled control and above the base model
  python3 -c "
import json, sys
m = json.load(open('$RUNS/pilot5_trend/metrics.json'))
a, s, b = m['after_trend_pairs'], m['shuffled_trend_pairs'], m['before_trend_pairs']
print(f'trend pairs: after {a:.3f}, shuffled {s:.3f}, before {b:.3f}')
sys.exit(0 if a >= s + 0.10 and a >= b + 0.10 else 1)" 2>/dev/null
}

TREND="pilot_temporal.py --train 0 --validation 0 --trend 4000 --before"
run pilot5_trend shuffled_trend_pairs $TREND
for seed in 1 2; do
  run class_seed$seed after_shuffled finetune_kernelsight.py --data ../kernelsight.npz --seed $seed --skip-before
  run pilot3ab_seed$seed shuffled pilot_temporal.py --peak 2000 --segments --seed $seed
done
run pilot4_balanced shuffled pilot_temporal.py --peak 2000 --peak-balanced
for seed in 3 4; do
  run class_seed$seed after_shuffled finetune_kernelsight.py --data ../kernelsight.npz --seed $seed --skip-before
  run pilot3ab_seed$seed shuffled pilot_temporal.py --peak 2000 --segments --seed $seed
done
if trend_works; then
  for seed in 1 2; do run pilot5_trend_seed$seed shuffled_trend_pairs $TREND --seed $seed; done
else
  echo "$(date -u +%H:%M) trend pilot did not work (or did not finish): no trend seeds"
fi
echo "$(date -u +%H:%M) queue done"
date -u > $RUNS/overnight.done
