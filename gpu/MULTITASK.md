# Multi-task GPU plan (1× H100, on demand, 4.50 USD/h)

One fine-tune of OpenTSLM-Flamingo on 4 use cases, 6,158 rows per epoch: classification 1,509, temporal 2,925 (480
of them reversed trend copies), segmentation 862, diagnosis 862. Then answers with controls on validation and test.
Code: `flamingo/multitask.py`. Queue: `gpu/multitask.sh`.

## Jobs, in order

| Job | Plan | Time | USD |
|---|---|---|---|
| VM, upload, `setup_vm.sh` | both | 15 min | 1.1 |
| `mt_smoke`: 4 rows per task, 1 epoch | round1 | 8 min | 0.6 |
| `mt_all`: 2 epochs, model saved; training 65 min, answers 83 min | round1 | 2.5 h | 11.3 |
| `mt_class`: classification only, 2 epochs | full | 30 min | 2.2 |
| `mt_all_seed1`: `mt_all` with seed 1 | full | 2.4 h | 10.9 |
| `har_cot`: EXP's HAR-CoT check | Mac, 0 USD | 1.5 h | 0 |

**Round-1 minimum** (`round1`): 2.9 h, 13 USD. **Full** (`full`): 5.8 h, 26 USD.

## Where the times come from

- Training, fit to 3 real epochs (class 583 s, pilot 1 1,356 s, pilot 2 1,244 s): 0.185 s + 0.204 s per 1,000
  tokens per row. 24-Channel rows (~950 tokens): 0.38 s. Temporal rows (~160 tokens): 0.22 s. Epoch: ~32 min.
- Answers, fit to the file times of 52 pilot test files (residual 0.025 s per row): 0.128 s per 1,000 prompt tokens
  + 0.052 s per output token. Per row: class 0.26 s, temporal 0.24 s, segmentation 1.1 s (19 tokens), explanation
  3.7 s (69 tokens).
- `mt_all` answers: test 60 min, validation 18 min, base model 5 min (20 rows per task).
- Largest risk: a segmentation answer at its 160-token limit takes 8.4 s, not 1.1 s.

## Kill rules

- `mt_smoke` fails, or 2 jobs fail in a row: the queue stops. Read `train.log`; stop the VM.
- 20 min after "start mt_all", `status.sh` must show at least 1,800 trained questions. If not, check `nvidia-smi`,
  then kill.
- Epoch 1 takes more than 50 min (`progress.sh`): kill.
- `results.validation.segmentation.real.tokens_mean` above 40 in `runs/mt_all/metrics.json`: the test passes run
  long. Skip `mt_all_seed1`.
- A job runs longer than 1.5× its time: kill it. A restart of `mt_all` loads the saved model and does not train again.
- Start a job only with its cost + 5 USD of credit left.
- Delete the VM after the last pull. A stopped VM still bills its disk.

## Primary result (fixed before the run)

Use case: GPU programmers who optimize kernels ("Talk to Your GPU"). **Diagnosis is the headline task:** "What
limits this kernel, and what should I try?" Main metric: limiter accuracy on test (`usecases/diagnosis/score.py`),
next to its baselines: majority .402, workload class -> limiter .570, text-only LR on the printed mean/std .938. The
labels come from rules on the counters, not from experts; say so. Diagnosis REPLACES explanation in `mt_all`.
Temporal reasoning (below) is the check that answers come from the trace; classification is the anchor;
segmentation is a secondary output.

- **Primary metric: paired consistency.** On an original–reversed pair whose correct answer changes, the share of
  pairs answered right in BOTH conditions (TMP's `pairs`: trend 306 pairs, the 46 hard trend pairs, peak 56,
  iteration 21). A model that ignores order scores exactly 0. Read on validation first: GO if the pair score is at
  least .10 with an interval that excludes 0, or z clustered by Problem is at least 2. Test claims need z >= 2.5.
- **Every control is scored against the answer for the input as given.** Each row stores the answer recomputed
  on the reversed and the shuffled input. Where the shuffled input has no defined answer (`controls.shuffled` is
  null, e.g. a busy-period number), the score is the **agreement with the original answer**, not accuracy: high
  agreement there means the model ignores order. Label it that way in every table.
- **No swapped control for temporal.** Its questions name a Bin and a counter that do not carry over to the
  donor's data. Swapped is used only where the question is the same for every Snapshot (classification,
  segmentation), and the donor's own answer is the correct answer for the donor's series.
- `multitask.py` scores temporal first (real, reversed, shuffled), so a stop midway still leaves the primary result.

## What to compare (on the Mac, from the pulled files)

1. Classification: `mt_all` against `mt_class`. Same harness, rows, epochs and seed. Use balanced accuracy and
   macro-F1, with `paired_difference` in `usecases/classification/score.py` (Problem clusters). Older reference:
   `runs/flamingo` .897 / .717 (3 epochs, 1 run). Text-only LR: .911 / .751.
2. Does it read the series? `swapped`: own answer against donor answer (`swap_read` in `extras`).
3. Does it use time order? `shuffled` and `reversed` against `real`, per task: temporal `pairs_reversed` (an
   order-blind model scores 0) and `paired_shuffled`; segmentation `reversed`; explanation shuffled order precision.
4. Base model against fine-tuned: `before`.
5. Go/no-go on validation only (each `SPEC.md`). Read the test once. Write "1 run" or "2 runs (mean, spread)".
   Report every job.

## Commands

```
# 0. Mac: check rows and scorers (no GPU; it ends with "no CUDA GPU")
cd ~/hackathon/flamingo && ~/hackathon/OpenTSLM/.venv/bin/python multitask.py \
  --tasks classification,temporal,segmentation,explanation --add-train temporal/train_reversed.jsonl:trend \
  --out ../runs/mt_check; rm -r ../runs/mt_check
# 1. Nebius console: 1x H100, on demand, disk 150 GiB or more. Then:
HOST=<user>@<vm-ip>
~/hackathon/gpu/upload.sh $HOST
ssh -t $HOST 'tmux new -s setup "bash ~/hackathon/gpu/setup_vm.sh; bash"'   # ~10 min; Ctrl-b d to leave
# 2. Start "round1" (smoke test + one complete fine-tune). Run "full" (class-only reference + seed 1) only after
#    the results are checked and the submission materials are done, and only with 20 USD or more of credit left.
ssh $HOST '(setsid nohup ~/hackathon/gpu/multitask.sh round1 >> ~/hackathon/runs/multitask.log 2>&1 < /dev/null &)'
# 3. Mac: copy the results every 5 min until the queue is done
caffeinate -i ~/hackathon/gpu/pull.sh $HOST --loop
# 4. Watch (read only)
ssh $HOST '~/hackathon/gpu/status.sh; ~/hackathon/gpu/progress.sh'
# 5. After "queue done": delete the VM in the console
```

A restart (after a stop, a crash, or "round1" → "full") is step 2 again; finished work is skipped. Then step 3
again, with the new IP if it changed. To stop on purpose:
`ssh $HOST "pkill -f gpu/multitask.sh; pkill -f 'multitask.py|run_har.py'"`.

## HAR-CoT

It runs on the Mac (EXP). GPU fallback: `hf auth login` on the VM (Gemma 3 270M is gated), then
`setup_har.sh` (OpenTSLM at 2968f4b, the HAR-CoT zip, a uv environment with torch 2.6+, transformers <4.58,
open-flamingo 2.0.1), then step 2 with `har`. H100 time: not measured (Mac: 4.4 s per window).

## Parallel: five 1-GPU VMs (90 USD credit)

An 8-GPU node bills all 8 GPUs while the longest job runs (~36 USD/h × ~3.2 h ≈ 115 USD), so use separate
1-GPU VMs. Each VM runs one plan; each job writes its own `runs/<job>`, so one Mac folder takes them all.

| VM | Plan | Jobs | Time | USD | What it gives |
|---|---|---|---|---|---|
| 1 | `seed0` | smoke, `mt_all` (+ base model) | ~3.0 h | ~14 | the main run; its saved model is the `ask.py` demo |
| 2 | `seed1` | smoke, `mt_all_seed1` | ~2.9 h | ~13 | run 2 of the recipe |
| 3 | `refs` | smoke, `mt_class`, `mt_diag` | ~1.7 h | ~8 | does mixing the tasks help or hurt? |
| 4 | `attend` | smoke, `mt_all_attend` | ~3.0 h | ~14 | answers read all 24 counters, not only the last one |
| 5 | `seed2` | smoke, `mt_all_seed2` | ~2.9 h | ~13 | run 3: mean and spread over 3 runs |

Total ~62 USD, ~3 h wall clock, ~28 USD left for one rerun. With a smaller GPU quota, start the plans in this
order: seed0, seed1, refs, attend, seed2 (a VM can run a second plan after its first one; finished jobs are skipped).
Choose between recipes (attend or not) on VALIDATION scores only.

```
# 1. Nebius console: up to 5 VMs, each 1x H100, on demand, disk 150 GiB or more. Then on the Mac:
H0=<user>@<ip0>; H1=<user>@<ip1>; H2=<user>@<ip2>; H3=<user>@<ip3>; H4=<user>@<ip4>
# 2. Upload, then setup + queue in one detached command per VM (keys in a terminal cannot stop it)
start() { ~/hackathon/gpu/upload.sh "$1" && ssh "$1" "(setsid nohup bash -c 'bash ~/hackathon/gpu/setup_vm.sh > ~/setup.log 2>&1 && ~/hackathon/gpu/multitask.sh $2 >> ~/hackathon/runs/multitask.log 2>&1' < /dev/null > /dev/null 2>&1 &)"; }
start $H0 seed0; start $H1 seed1; start $H2 refs; start $H3 attend; start $H4 seed2
# 3. Watch (read only): setup first (~10-15 min), then the queue
for H in $H0 $H1 $H2 $H3 $H4; do echo "== $H"; ssh $H 'tail -1 ~/setup.log; tail -2 ~/hackathon/runs/multitask.log 2>/dev/null'; done
# 4. Mac: copy results from every VM every 5 min until each queue is done (logs: runs/logs/<ip>.log)
mkdir -p ~/hackathon/runs/logs
for H in $H0 $H1 $H2 $H3 $H4; do caffeinate -i ~/hackathon/gpu/pull.sh $H --loop > ~/hackathon/runs/logs/pull_${H#*@}.txt 2>&1 & done
# 5. Optional demo on VM 1 when mt_all is done (~10 min): record a screen video of real answers
ssh -t $H0 'cd ~/hackathon/flamingo && .venv/bin/python ask.py ../runs/mt_all/model'
# 6. Delete each VM in the console as soon as its queue says "queue done" (a stopped VM still bills its disk)
```

## 8-GPU node (preferred with 90 USD): `gpu/node8.sh`

One node, one setup, 8 lanes that end at about the same time, so no GPU idles while the node bills (~36 USD/h).

| GPU | Lane | Ends after (est.) |
|---|---|---|
| 0 | `mt_all`: main recipe, seed 0, base model too, model saved | ~110 min |
| 1-4 | `mt_all_seed1..4`: seeds 1-4 (5 runs in all, like the 5-seed baselines) | ~105 min |
| 5 | `mt_all_attend`: answers read all 24 counters | ~110 min |
| 6 | `mt_smoke` (8 min; a failure stops every lane), `mt_class`, then the heavy controls of `mt_all` from its saved model | ~110 min |
| 7 | `mt_diag`, then the heavy controls of `mt_all_attend` | ~110 min |

Light controls (every run): real answers, temporal reversed (the pair score). Heavy controls (once per recipe):
diagnosis/temporal/classification/segmentation shuffled, classification/segmentation swapped. A final rescore puts
both halves in one metrics.json. Each job has a time cap (140 min; heavy 60 min).

Cost: setup ~15 min + ~115 min ≈ 2 h 10 min ≈ 78 USD at 36 USD/h; caps bound it near 90. Node: 8x H100, on demand,
disk 300 GiB or more (8 saved models of 13.7 GB). Delete it as soon as runs/node8.log says "node done".

```
H=<user>@<node-ip>
~/hackathon/gpu/upload.sh $H
ssh $H "(setsid nohup bash -c 'bash ~/hackathon/gpu/setup_vm.sh > ~/setup.log 2>&1 && ~/hackathon/gpu/node8.sh' < /dev/null > /dev/null 2>&1 &)"
ssh $H 'tail -1 ~/setup.log; cat ~/hackathon/runs/node8.log; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'
caffeinate -i ~/hackathon/gpu/pull.sh $H --loop      # ends when the node writes runs/multitask.done
ssh -t $H 'cd ~/hackathon/flamingo && .venv/bin/python ask.py ../runs/mt_all/model'   # optional demo video, ~10 min
```

## One node, 4 GPUs (chosen plan): `gpu/node.sh 4`

Supersedes the 8-GPU section above (`node8.sh` is now `node.sh`; `node.sh 8` still runs that layout). No time caps.

| GPU | Lane | Ends after (est.) |
|---|---|---|
| 0 | `mt_all`: all 4 tasks, seed 0, base model too, model saved (demo) | ~110 min |
| 1 | `mt_all_seed1` | ~105 min |
| 2 | `mt_all_attend`: answers read all 24 counters | ~110 min |
| 3 | `mt_smoke` (a failure stops every lane), `mt_class`, `mt_diag`, then the heavy controls of `mt_all` | ~120 min |

Cost: 4 GPUs × (setup ~15 min + ~120 min + ~10 min until deleted) ≈ 9.7 GPU-h ≈ 44 USD at 4.50 USD/GPU-h.
Node: 4x H100, on demand, disk 200 GiB or more (3 saved models). Delete it when runs/node8.log says "node done".

```
H=<user>@<node-ip>
~/hackathon/gpu/upload.sh $H
ssh $H "(setsid nohup bash -c 'bash ~/hackathon/gpu/setup_vm.sh > ~/setup.log 2>&1 && ~/hackathon/gpu/node.sh 4' < /dev/null > /dev/null 2>&1 &)"
ssh $H 'tail -1 ~/setup.log; cat ~/hackathon/runs/node8.log; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'
caffeinate -i ~/hackathon/gpu/pull.sh $H --loop
```
