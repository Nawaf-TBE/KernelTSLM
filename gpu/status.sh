#!/bin/bash
# One line per run in ~/hackathon/runs: running or not, questions trained, and whether its results are complete.
# Safe to stop with Ctrl-C: it only reads files.  watch -n 30 ~/hackathon/gpu/status.sh
cd ~/hackathon/runs || exit 1
printf "%-20s %-8s %8s  %s\n" run state trained results
for d in */; do
  d=${d%/}
  [ -f "$d/run.pid" ] || continue
  kill -0 "$(cat "$d/run.pid")" 2>/dev/null && state=RUNNING || state=stopped
  n=$(wc -l < "$d/losses.jsonl" 2>/dev/null | tr -d ' ')
  r=$(python3 -c "
import json, sys
m = json.load(open('$d/metrics.json'))
if 'complete' in m:  # multitask.py
    sys.exit(print('complete' if m['complete'] else '-'))
keys = {'shuffled_trend_pairs'} if m['args'].get('trend') else {'shuffled'} if 'number_ranges' in m else {'after_shuffled'}  # pilot, else fine-tune
print('complete' if keys <= set(m) else '-')" 2>/dev/null || echo -)
  [ "$r" = - ] && grep -q Traceback "$d/train.log" 2>/dev/null && r=CRASHED
  printf "%-20s %-8s %8s  %s\n" "$d" "$state" "${n:-0}" "$r"
done
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
[ -f multitask.done ] && echo "queue: done ($(cat multitask.done))"
