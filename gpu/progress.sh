#!/bin/bash
# Progress of one run (default: the run that multitask.sh runs now). Safe to stop with Ctrl-C:
# it only reads files.
#   ~/hackathon/gpu/progress.sh                         # the current run of the queue
#   ~/hackathon/gpu/progress.sh ~/hackathon/runs/mt_all
R=$HOME/hackathon/runs
f=${1:-$(cat "$R/multitask.current" 2>/dev/null)}
[ -d "$f" ] || { echo "no run yet"; exit 0; }
echo "run: $(basename "$f")"
kill -0 "$(cat "$f/run.pid" 2>/dev/null)" 2>/dev/null && echo "RUNNING (pid $(cat "$f/run.pid"))" || echo "STOPPED"
echo "trained $(wc -l < "$f/losses.jsonl" 2>/dev/null || echo 0) questions"
[ -s "$f/losses.jsonl" ] && tail -100 "$f/losses.jsonl" | python3 -c "import sys, json; v = [json.loads(l)['loss'] for l in sys.stdin]; print('loss, mean of last 100:', round(sum(v) / len(v), 3))"
grep -E '^\{"(epoch"|training"|class_test|after|shuffled)|Traceback' "$f/train.log" 2>/dev/null | cut -c1-150
grep '"summary"' "$f/train.log" 2>/dev/null | tail -1 | cut -c1-600
echo "last answer: $(grep evaluated "$f/train.log" 2>/dev/null | tail -1 | cut -c1-160)"
[ -f "$R/multitask.log" ] && { echo "--- queue"; tail -4 "$R/multitask.log"; }
