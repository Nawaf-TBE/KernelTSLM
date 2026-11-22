# Overnight GPU runs (superseded)

**Now: [MULTITASK.md](MULTITASK.md)** (multitask.sh: all use cases in one fine-tune, with controls). The text below
describes the older single-task queues (overnight.sh, parallel.sh). pull.sh now waits for multitask.sh by default;
pass `--loop overnight` for the old queue.

The trend pilot first, then five runs of each LLM recipe (the baselines have 5 seeds), the class test with shuffled
Bins, and the balanced before-or-after pilot. About 8.5 hours on one H100, about 38 USD at 4.50 USD per hour; about
10 hours and 45 USD if the trend pilot works and its 2 extra seeds run (estimates).

| Job | Script | What it gives |
|---|---|---|
| `pilot5_trend` | `pilot_temporal.py --train 0 --validation 0 --trend 4000 --before` | Is a counter's trend increasing or decreasing? Each question also comes reversed in time with the opposite answer, so a model that ignores order gets exactly one of each pair right. Scored before training, after, and with shuffled Bins, on 150 test pairs |
| `class_seed1` to `class_seed4` | `finetune_kernelsight.py --seed N --skip-before` | Class-only fine-tune; seed 0 is `runs/flamingo` |
| `pilot3ab_seed1` to `pilot3ab_seed4` | `pilot_temporal.py --peak 2000 --segments --seed N` | Pilot 3ab recipe; seed 0 is `runs/pilot3ab` |
| `pilot4_balanced` | `pilot_temporal.py --peak 2000 --peak-balanced` | before-or-after questions where Bin N alone tells nothing (text-only rule: 0.504) |
| `pilot5_trend_seed1`, `seed2` | the trend pilot with `--seed N` | Only if the trend pilot works: its pair score is at least 0.10 above the shuffled control and above the base model |

Every class score now has a second score with the Bins shuffled (`after_shuffled`, `class_test_shuffled`): if it
falls, the model uses time order for the class task. Every job saves its model in `runs/<job>/model` (13 GB).

## Steps

Replace `<host>` with `<user>@<vm-ip>` of the VM.

1. From the Mac, send the files (about 20 MB):

       ~/hackathon/gpu/upload.sh <host>

2. Set up the VM (skips what is there; about 5 minutes on a new VM):

       ssh <host> 'bash ~/hackathon/gpu/setup_vm.sh'

3. Start the queue, detached (keys in a terminal cannot stop it):

       ssh <host> '(setsid nohup ~/hackathon/gpu/overnight.sh > ~/hackathon/runs/overnight.log 2>&1 < /dev/null &)'

4. On the Mac, copy the results every 5 minutes until the queue is done:

       ~/hackathon/gpu/pull.sh <host> --loop

5. Watch it (only reads files; Ctrl-C is safe):

       ssh <host> '~/hackathon/gpu/progress.sh'

6. When `runs/overnight.done` is on the Mac, stop the VM in the Nebius console (not with `shutdown` inside the VM).
   The models stay only on the VM disk.

A restart of step 3 skips the jobs that are complete. To stop on purpose:

    ssh <host> 'pkill -f overnight.sh; kill $(cat "$(cat ~/hackathon/runs/overnight.current)/run.pid")'

## On one VM with 8 GPUs (about 75 minutes)

`parallel.sh` runs 8 jobs at once, one per GPU: `pilot5_trend` (with the base model scored), `pilot5_trend_seed1`,
`class_seed1` to `class_seed4`, `pilot3ab_seed1` and `pilot3ab_seed2`. The pilot 3ab jobs take about 60 minutes, so
rent the node until `parallel.done` appears, not for a fixed hour. Disk: at least 150 GB (8 models of 13 GB, the cache).

    ~/hackathon/gpu/upload.sh <host>
    ssh <host> 'bash ~/hackathon/gpu/setup_vm.sh'
    ssh <host> '(setsid nohup ~/hackathon/gpu/parallel.sh > ~/hackathon/runs/parallel.log 2>&1 < /dev/null &)'
    ~/hackathon/gpu/pull.sh <host> --loop parallel      # on the Mac
    ssh <host> 'watch -n 30 ~/hackathon/gpu/status.sh'  # one line per job; Ctrl-C is safe

The jobs left out (`pilot4_balanced`, `pilot3ab_seed3` and `seed4`, `pilot5_trend_seed2`) can run later on one GPU
with `overnight.sh`, which skips the jobs that are complete.

## Rules for the report, fixed before the runs

Every job goes into the results, also the ones that look bad. Seeds are reported as mean and spread, never as the best
seed. A choice between settings uses validation scores, never test scores.
