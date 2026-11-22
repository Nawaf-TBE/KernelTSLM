# KernelTSLM

**Live demo:** https://kerneltslm.m7m2od.com · **Video:** https://www.youtube.com/watch?v=fsTSLqM1wMU ·
**TimeNet connector:** [OpenTSLM/TimeNet#835](https://github.com/OpenTSLM/TimeNet/pull/835)

KernelTSLM reads the hardware counters of a GPU kernel and tells you, in plain words, what is slowing it down.

While a kernel runs, an NVIDIA H100 records 24 counters over time: how busy the tensor cores are, how much DRAM
bandwidth is in use, how often warps stall at a barrier, and so on. A performance engineer reads these to answer
questions like "why is this kernel slow?" and "what should I try next?". That takes experience. We fine-tuned one
time-series language model (OpenTSLM-Flamingo on Gemma 3 4B) to answer four of those questions from the counters,
and to state numbers you can check against the trace.

Built for the TimeNet Hackathon 2026, Round 1, by [m7m2od](https://github.com/m7m2od) and
[Nawaf-TBE](https://github.com/Nawaf-TBE).

## An example

This is the model's answer for a test kernel it never saw in training (a matmul with an activation, KernelBench
Level 2 problem 29, in fp32):

> FMA pipe is 50% of peak while busy. The GPU is idle 35% of the active span, across 33 busy periods. Tensor
> active is 0% of peak while busy. Limiter: FMA pipe. Try: Use bf16 or TF32 so the tensor cores run.

Each sentence is a measurement you can look up in the counters. The reference answer for this kernel also says
FMA pipe.

## The demo

Open https://kerneltslm.m7m2od.com, or `site/index.html` locally; it needs no server. Six test kernels play one after another. For each, the
page draws the 24 counters like the leads of an ECG, then shows the model's reading next to the reference answer.
Below the demo, the same page explains the use cases and shows the results and findings.

The live demo page lets you run a test kernel, watch its counters fill in, and ask the model about it in your own words (the model's recorded answers, replayed at their recorded speed).

## Terms

The dataset and this README use a few words in a fixed sense. [GLOSSARY.md](GLOSSARY.md) has the full list.

- **Snapshot**: one profiled run of one kernel, 24 counters by 512 time steps.
- **Bin**: one of those 512 time steps. Bins are labelled with the kind of work running in them, or idle.
- **Problem**: one kernel at one shape. Its Variants differ only in data type (fp32, bf16), seed or launch
  options, so they look almost the same.
- **Limiter**: the part of the GPU that caps the kernel's speed. The model picks one of nine: launch gaps, low SM
  utilization, memory bandwidth, tensor cores, FMA pipe, ALU pipe, XU pipe, barrier stalls, latency.
- **Workload type**: the kind of work, such as matmul, attention, conv, softmax, reduction or normalization
  (11 types occur in the data).

## The four questions

One model answers all four. Each has its own folder under `usecases/`.

| Question | What the model writes | Example |
|---|---|---|
| What limits this kernel, and what should I try? (`diagnosis`) | 1 to 3 checked facts, then `Limiter: <limiter>.` and `Try: <step>` | the answer above |
| What ran when? (`segmentation`) | a timeline in 16-Bin steps | `0 idle; 224 attention; 288 idle` |
| Timing questions (`temporal`) | one option | "While Barrier stalls is active, is it increasing or decreasing?" decreasing |
| What kind of work is this? (`classification`) | the workload type | matmul |

The reference answers come from fixed rules applied to the counters and labels (each task's `SPEC.md` gives the
rules). They are not written by experts, and the model learns to give the same answers in words.

## How the model works

OpenTSLM-Flamingo (`opentime/tslm-nov26`) puts a time-series encoder in front of Gemma 3 4B. Each counter goes in
as its own series, with a short text line giving its mean and standard deviation, followed by the question. We
fine-tune the series encoder, the cross-attention layers that connect it to the language model, and the token
embeddings; the rest of Gemma stays frozen.

The main run, `runs/mt_all`, trains on all four tasks at once: 6,158 questions per epoch, 2 epochs, seed 0. Training
took 64 minutes on one H100. The code is `flamingo/multitask.py`, and `gpu/` has the scripts that ran it.

## Results

All scores are on held-out test kernels from Problems the model never saw in training (291 test Snapshots), from
one training run. Balanced accuracy averages the accuracy of each class, so rare classes count as much as common
ones.

### Diagnosis: what limits this kernel, and what should I try?

| Model | Right limiter | Notes |
|---|---|---|
| Always guess the most common limiter | 0.402 | |
| Guess the limiter from the workload type | 0.570 | |
| Logistic regression on each counter's mean and std | 0.938 | the averages carry much of the answer |
| Fine-tuned on diagnosis only (`runs/mt_diag`) | 0.780 | |
| **Fine-tuned on all four tasks (`runs/mt_all`)** | **0.787** | balanced 0.503; right next step 0.698; 82.5% of stated facts check out |
| Base model before fine-tuning | no valid answers | |

The logistic regression scores higher: it gets the limiter and the next step right 0.938 of the time, but it
states no facts. The model writes the facts behind its answer, and 82.5% of them match the counters.

### Workload type

11 classes; accuracy / balanced accuracy.

| Model | Score | Runs |
|---|---|---|
| Logistic regression on counter averages | 0.911 / 0.751 | 1 |
| CNN (`L1TemporalCNN`, see [CNN baseline](#cnn-baseline)) | 0.910 / 0.715 | 5 seeds |
| MiniRocket | 0.940 / 0.811 | 5 seeds |
| Fine-tuned on workload type only (`runs/mt_class`) | 0.890 / 0.731 | 1 |
| **Fine-tuned on all four tasks** | **0.887 / 0.747** | 1 |

MiniRocket, a standard time-series classifier, is the strongest model built only for this task.

### Timeline: what ran when

The model gets 0.897 of Bins right (a baseline that reads only the text lines gets 0.883). Busy or idle is right
for 0.919 of Bins, and the workload type for 0.826.

### Does the model read the trace?

A model could score well from the mean and std in the text lines alone and ignore the series. We tested this by
changing the series and keeping the text the same. In four earlier one-epoch pilots (`runs/pilot*`) the model
did not read the timeline; in the main run it does.

| Test | Real trace | Changed trace |
|---|---|---|
| Ask about a counter's trend on a trace and on the same trace reversed in time, where the right answer flips. Score: both answers right. A model that ignores time order gets exactly 0. | 0.709 [0.585, 0.817] | 0 by design |
| Timing questions on the real trace and on the same trace with its Bins shuffled | 0.803 | 0.465 (paired z by Problem 9.5) |
| Workload type when the series comes from another kernel of a different type but the text line is this kernel's own | 0.887 | 0.378 |

The brackets are 95% bootstrap intervals over Problems. Full numbers are in `runs/mt_all/metrics.json`, under
`results` and `extras`.

## Findings

1. **The diagnosis points at real speedups.** The dataset has the same kernels in fp32 and bf16, so we can check
   the advice "use bf16" against measured run times. fp32 kernels that the rules call memory-bound run 1.68x faster
   in bf16 (median over 68 pairs from 34 Problems, IQR 1.17 to 1.87). ALU-bound ones run 1.01x faster (18 pairs).
   Source: `runs/usecases/diagnosis/sanity.json`.
2. **The official split leaks.** In the released split, Variants of one kernel sit in both train and test: 120 of
   the 160 official test Snapshots share a Problem with official training (84 of 401 Problems cross). MiniRocket
   scores 0.999 on the official split, 0.995 when the official train set is cut to our size, and 0.940 on our split,
   which keeps each Problem on one side. Our TimeNet connector adds this grouped split.
3. **One model for four tasks costs nothing.** Diagnosis scores 0.787 when trained with the other tasks and 0.780
   alone. Workload balanced accuracy is 0.747 with the other tasks and 0.731 alone.
4. **A released time-series language model explains things its input does not show.** We ran OpenTSLM's released
   HAR-CoT model (`OpenTSLM/gemma-3-270m-pt-har-flamingo`, which names human activities from motion sensors) on 400
   test windows, 50 per class. Its accuracy is 0.58 on real input, 0.265 on time-shuffled input and 0.578 on
   reversed input, so its answers do depend on time order. But its explanations call the motion periodic on 46.5%
   of shuffled windows (49.5% of real ones), and shuffled input has no periodicity. This is why our diagnosis
   states facts that can be checked against the counters. Source: `runs/usecases/explanation/har/metrics.json`.

## Data and split

KernelSight (`williamhtan/kernelsight`, Apache-2.0) has 1,444 kernel traces from one H100: 24 counters by 512
Bins each, with a workload label per Bin. The kernels are GEMMs, attention, convolutions and the fused KernelBench
problems.

We wrote the TimeNet connector that loads KernelSight
([OpenTSLM/TimeNet#835](https://github.com/OpenTSLM/TimeNet/pull/835)). We split the Snapshots 862 / 291 / 291
into train, validation and test, grouped by Problem, so no Problem appears on two sides (see finding 2).

`kernelsight.npz` is the exported dataset; `export_kernelsight.py` writes it from a TimeNet checkout. Every script
in this repository reads it. `temporal_qa/labels_cache.npz` holds the per-Bin labels in the same order.

## How each task is built

Each folder in `usecases/` has the same four files:

- `SPEC.md`: the design, the baselines and the known risks
- `build.py`: writes the question rows to `runs/usecases/<task>/{train,validation,test}.jsonl`
- `score.py`: the scorer
- `baselines.py`: non-LLM baselines for that task

| Task | Train rows |
|---|---|
| diagnosis | 862 |
| temporal | 2,925 |
| classification | 1,509 (each rare class repeated to 100 rows) |
| segmentation | 862 |

Every row also stores the right answer for the reversed and the shuffled trace, so the tests in
[Does the model read the trace?](#does-the-model-read-the-trace) score each changed input against its own answer.
`usecases/explanation/` holds an earlier explanation task and the HAR-CoT test behind finding 4.

## CNN baseline

Nawaf-TBE built a counter-only residual CNN (`L1TemporalCNN`, in `src/kernel_xray/phase3/`) that labels the
workload of every Bin. On the official validation split, over 33,416 labelled Bins, it reaches macro-F1 0.946 and
accuracy 95.9% (unweighted model, frozen at epoch 7; the class-weighted model scores 0.943).

An independent CPU replay reproduces every saved prediction. Bins at the boundary between two kinds of work are
harder (63.9% right), and the trace-bootstrap interval for macro-F1 is about [0.831, 0.969].

The CNN row in the workload table uses the same architecture, trained on our grouped split, with its per-Bin
outputs averaged into one class per Snapshot.

More detail: [results](docs/cnn-baseline.md), [method](docs/phase3_cnn_baseline.md),
[independent audit](docs/phase3_cnn_independent_audit.md). Figures, metrics and checksums are in `results/cnn/`.
[docs/README.md](docs/README.md) lists every document.

## Repository layout

```text
README.md, GLOSSARY.md, LICENSE
kernelsight.npz         the dataset as exported from TimeNet
export_kernelsight.py   writes kernelsight.npz
baselines.py            workload-type baselines: logistic regression, nearest neighbour, MiniRocket, CNN

usecases/<task>/        the four tasks (and explanation/): SPEC.md, build.py, score.py, baselines.py
flamingo/               the model: multitask.py (main fine-tune), ask.py (ask your own questions),
                        walkthrough.py (marimo notebook), utils.py (model code), earlier single-task scripts
temporal_qa/            question templates and the per-Bin label cache used by the row builders
gpu/                    scripts that run the fine-tune on a rented H100
runs/                   every result: mt_all (main run), mt_class, mt_diag, baselines, use-case rows,
                        flamingo (first class-only run), pilot* (one-epoch pilots), temporal_qa, temporal_baselines
experiments/            temporal_order/: the question set and per-Bin baselines that led to the time-order tests
site/                   the demo page

src/kernel_xray/        CNN package: dataset loader, visualization, L1TemporalCNN
scripts/, tests/        CNN data pipeline, training, audit and verification; pytest suite
configs/, results/cnn/  CNN configuration; CNN figures, metrics and provenance
docs/                   CNN method and audit, TimeNet connector notes
```

## Reproduce

You need [uv](https://docs.astral.sh/uv/). These run on a laptop from the files in the repository:

```sh
uv run usecases/<task>/build.py                                                 # question rows (seconds)
uv run baselines.py                                                              # workload-type baselines (~5 min)
uv run --no-project --with numpy python experiments/temporal_order/temporal_questions.py   # timing question set
```

The rebuilt rows and question sets match the committed files byte for byte. Logistic-regression scores can move
by one test kernel between platforms: the diagnosis limiter scores 0.938 on macOS, where the committed results
were made, and 0.942 on Linux.

The fine-tune needs one H100 with 80 GB; [gpu/README.md](gpu/README.md) has the steps. To try the trained model on
your own questions, run `flamingo/ask.py` with the model that the main run saves on the GPU machine.

Two parts need the raw KernelSight release: the run-time check behind finding 1 (`usecases/diagnosis/build.py`
skips it without the data) and the CNN pipeline with its tests. The release is 86 MB:

```sh
hf download williamhtan/kernelsight --repo-type dataset --local-dir data/kernelsight_v4
ln -s data/kernelsight_v4/kernels data/kernelsight_v4/splits .     # paths the CNN code expects
uv run --with pytest python -m pytest -q                            # CNN tests
```

## Limits

- The diagnosis labels come from fixed rules on the counters (`usecases/diagnosis/SPEC.md`), not from experts.
- Counter averages already carry much of the answer: logistic regression gets the limiter right 0.938 of the time.
- Rare limiters are weak (balanced limiter accuracy 0.503), and some test classes have a single Problem.
- All traces come from one H100 and short, single-kernel runs. Each recipe was trained once.

## Team

- [m7m2od](https://github.com/m7m2od): the multi-task fine-tune, the four use cases and their baselines, the
  time-order tests, the HAR-CoT test, the demo page, and the TimeNet KernelSight connector
  ([OpenTSLM/TimeNet#835](https://github.com/OpenTSLM/TimeNet/pull/835)).
- [Nawaf-TBE](https://github.com/Nawaf-TBE): the CNN baseline and its independent audit, dataset validation, the
  loader and visualization in `src/kernel_xray/`, and the TimeNet conversion and memory studies (`docs/timenet_*`).

## Credits and licenses

The code is MIT licensed ([LICENSE](LICENSE)).

- Data: KernelSight by williamhtan (Apache-2.0).
- Model: OpenTSLM-Flamingo (`opentime/tslm-nov26`).
- `flamingo/utils.py` comes from the AWS sample `opentslm-finetune-arfbench`, with marked changes. Its Flamingo
  layers keep the MIT notices of OpenFlamingo and flamingo-pytorch.
- Photo in `site/`: 极客湾Geekerwan, via Wikimedia Commons, CC BY 3.0, cropped.
