# Running the fine-tune on a GPU

`multitask.sh` runs `flamingo/multitask.py` as a queue of jobs on one H100 (80 GB). Each job writes `runs/<job>/`,
and a restarted queue skips finished jobs and finished prediction files. The published runs came from two plans:

| Plan | Jobs | Time on one H100 |
|---|---|---|
| `seed0` | smoke test, then `mt_all`: all four tasks, 2 epochs, seed 0, base model scored too, model saved | ~2.5 h |
| `refs` | smoke test, then `mt_class` and `mt_diag`: one task each, same recipe | ~1.7 h |

Other plans: `seed1`, `seed2` (more seeds of `mt_all`), `attend` (answers read all 24 counters), `round1` / `full`
(the same jobs in sequence on one machine) and `har` (the HAR-CoT check of `usecases/explanation/`, which also runs
on a laptop). Each plan starts with a 4-row smoke test; two failed jobs in a row stop the queue.

## Steps

The scripts copy the repository to `~/hackathon` on the GPU machine. From the repository root:

```sh
HOST=<user>@<gpu-ip>
gpu/upload.sh $HOST                     # code, kernelsight.npz and the use-case rows (~40 MB)
ssh $HOST "(setsid nohup bash -c 'bash ~/hackathon/gpu/setup_vm.sh > ~/setup.log 2>&1 && \
  ~/hackathon/gpu/multitask.sh seed0 >> ~/hackathon/runs/multitask.log 2>&1' < /dev/null > /dev/null 2>&1 &)"
gpu/pull.sh $HOST --loop                # copies results to runs/ every 5 min until the queue is done
```

`setup_vm.sh` installs uv, a Python 3.12 environment from `flamingo/requirements.txt`, and downloads
`opentime/tslm-nov26` (13.7 GB). On the GPU machine, `status.sh` lists every run and `progress.sh` follows the
current one; both only read files.

`close_when_done.sh <host> <instance-id> <job>...` waits for the queue, pulls the results, checks that each job's
`metrics.json` is complete and only then deletes the instance with the Nebius CLI.
