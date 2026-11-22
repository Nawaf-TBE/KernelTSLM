# Kernel X-Ray: GPU kernel performance diagnosis from hardware counters

TimeNet Hackathon 2026, Round 1. Kernel X-Ray reads the 24 hardware counters an NVIDIA H100 records while a
kernel runs, and answers four questions a performance engineer asks: what limits the kernel and what to try next,
where the time goes, what ran when, and what kind of work it is. One fine-tuned time-series language model
(OpenTSLM-Flamingo on Gemma 3 4B) answers all four, in plain language, with numbers that can be checked.

**Demo:** open `site/index.html` in a browser. Six test kernels the model never saw play one after another, each
drawn as chart leads with the model's reading, followed by the use cases, results and findings.

## Results (held-out test kernels, grouped split, one training run)

Main run `runs/mt_all`: one model, four tasks, seed 0, 2 epochs, 64 min on one H100.

**Diagnosis: "What limits this kernel, and what should I try?"** (291 test kernels)

| Model | Limiter accuracy | Notes |
|---|---|---|
| Always the most common limiter | 0.402 | |
| Limiter from the workload type | 0.570 | |
| Logistic regression on counter averages (mean, std) | 0.938 | the averages carry much of the answer |
| Fine-tuned, diagnosis only (`runs/mt_diag`) | 0.780 | |
| **Fine-tuned, all four tasks (`runs/mt_all`)** | **0.787** | balanced 0.503; next step right 0.698; 82.5% of stated findings check out |
| Base model, before fine-tuning | 0 valid answers | |

**Workload type** (11 classes, 291 test kernels; accuracy / balanced accuracy)

| Model | Score | Runs |
|---|---|---|
| Logistic regression on averages | 0.911 / 0.751 | 1 |
| CNN (`L1TemporalCNN`, see [CNN baseline](#cnn-baseline)) | 0.910 / 0.715 | 5 seeds |
| MiniRocket | 0.940 / 0.811 | 5 seeds |
| Fine-tuned, workload only (`runs/mt_class`) | 0.890 / 0.731 | 1 |
| **Fine-tuned, all four tasks** | **0.887 / 0.747** | 1 |

**Timeline ("what ran when", 16-Bin steps):** 0.897 of Bins right (text-only baseline 0.883); busy/idle per Bin
0.919; workload per Bin 0.826.

**Does the model read the timeline?** Yes, in this run (it did not in four earlier one-epoch pilots, `runs/pilot*`):

| Control | Real trace | Control trace |
|---|---|---|
| Trend asked on a trace and on the same trace reversed, both right (order-blind model: exactly 0) | 0.709 [0.585, 0.817] | 0 by design |
| Timing questions, real vs time-shuffled trace | 0.803 | 0.465 (paired z by Problem 9.5) |
| Workload type with another kernel's series but this kernel's summary text | 0.887 | 0.378 |

Intervals: 95% bootstrap over Problems. Full numbers: `runs/mt_all/metrics.json` (`results`, `extras`).

## Findings

1. **The diagnosis points at real speedups.** The dataset holds the same kernels in fp32 and bf16. fp32 kernels
   that the rules diagnose as memory-bound run 1.68x faster in bf16 (median, 68 pairs from 34 Problems, IQR
   1.17-1.87); ALU-bound ones run 1.01x (18 pairs). Measured from kernel timing. `runs/usecases/diagnosis/sanity.json`.
2. **The release split leaks.** Variants (dtype, seed, launch options) of one kernel sit in train and test:
   120 of 160 official test Snapshots share a Problem with official training (84 of 401 Problems cross).
   MiniRocket scores 0.999 on the official split, 0.995 with the official train set cut to our size, and 0.940 on
   our grouped split. Our TimeNet connector (OpenTSLM/TimeNet PR #835) adds the grouped split.
3. **One model for four tasks costs nothing.** Diagnosis 0.787 multi-task vs 0.780 alone; workload balanced
   accuracy 0.747 vs 0.731 alone.
4. **A released TSLM's explanations do not follow its input.** OpenTSLM's released HAR-CoT model
   (`OpenTSLM/gemma-3-270m-pt-har-flamingo`, 400 test windows, 50 per class): accuracy 0.58 on real input, 0.265 on
   time-shuffled input, 0.578 reversed, so its answers use order. But its texts still call the motion periodic on
   46.5% of shuffled windows (49.5% on real ones), although shuffled input has no periodicity. This is why our
   diagnosis states checkable facts. `runs/usecases/explanation/har/metrics.json`.

## Data and split

KernelSight (`williamhtan/kernelsight`, Apache-2.0): 1,444 kernel traces from an H100, 24 counters x 512 time
Bins, with per-Bin workload labels. Read through our TimeNet connector (PR #835). Split 862 / 291 / 291 Snapshots,
grouped by Problem (kernel motif and shape without dtype, seed and launch options), so 0 Problems cross.
`kernelsight.npz` is the exported dataset (`export_kernelsight.py` writes it) and `temporal_qa/labels_cache.npz`
holds the per-Bin labels the question builders use. Terms (Snapshot, Problem, Bin, ...) are defined in
[GLOSSARY.md](GLOSSARY.md).

## The four tasks (`usecases/<task>/`, rows in `runs/usecases/<task>/`)

| Task | Answer the model writes | Train rows |
|---|---|---|
| diagnosis | 1-3 checked facts, `Limiter: <limiter>.`, `Try: <step>` | 862 |
| temporal | one option: trend, busy period, longest idle gap, peak before/after Bin N | 2,925 |
| classification | the workload class | 1,509 |
| segmentation | a timeline, e.g. `0 idle; 224 attention; 288 idle` | 862 |

Each folder has `SPEC.md` (design, baselines, risks), `build.py` (writes the rows), `score.py` (scorer) and
`baselines.py`. Every row also stores the correct answer for the time-reversed and the shuffled trace.
`usecases/explanation/` holds the HAR-CoT check behind finding 4.

## CNN baseline

A counter-only residual CNN (`L1TemporalCNN`, `src/kernel_xray/phase3/`) that labels the primary workload of every
Bin. On the official validation split, over 33,416 labeled Bins: macro-F1 0.946, accuracy 95.9% (unweighted,
frozen at epoch 7; class-weighted 0.943). An independent CPU replay matches every saved prediction; boundary Bins
are harder (63.9%) and the trace-bootstrap interval for macro-F1 is about [0.831, 0.969]. The same architecture,
trained on our grouped split with its Bin logits averaged into one class per Snapshot, is the CNN row in the
workload table above.

Details: [results](docs/cnn-baseline.md), [method](docs/phase3_cnn_baseline.md),
[independent audit](docs/phase3_cnn_independent_audit.md), figures and provenance in `results/cnn/`.

## Repository layout

```text
kernelsight.npz         the dataset as exported from TimeNet (export_kernelsight.py writes it)
baselines.py            workload-type baselines: logistic regression, nearest neighbour, MiniRocket, CNN

usecases/<task>/        the four tasks (plus explanation/): SPEC.md, build.py, score.py, baselines.py
flamingo/               OpenTSLM-Flamingo: multitask.py (main fine-tune), ask.py (interactive demo),
                        walkthrough.py (marimo notebook), utils.py (model code), earlier single-task scripts
temporal_qa/            question templates and the per-Bin label cache used by the row builders
gpu/                    scripts that run the fine-tune on a rented H100 (README inside)
runs/                   every result: mt_all (main run), mt_class, mt_diag, baselines, use-case rows,
                        flamingo (class-only run), pilot* (one-epoch pilots), temporal_qa, temporal_baselines
experiments/            temporal_order/: the question set and per-Bin baselines that motivated the time controls
site/                   demo page (index.html)

src/kernel_xray/        CNN baseline package: dataset loader, visualization, L1TemporalCNN
scripts/, tests/        CNN data pipeline, training, audit and verification; pytest suite
configs/, results/cnn/  CNN configuration; CNN figures, metrics and provenance
docs/                   CNN method and audit, TimeNet connector notes (index: docs/README.md)
```

## Reproduce

Everything below runs from `kernelsight.npz` and the files in the repository:

```sh
uv run --no-project --with numpy,scikit-learn usecases/<task>/build.py      # use-case rows (seconds)
uv run baselines.py                                                          # workload-type baselines (~5 min)
uv run --no-project --with numpy python experiments/temporal_order/temporal_questions.py
```

The rebuilt rows and question sets match the committed files exactly. Logistic-regression scores can move by one
test kernel between platforms (diagnosis limiter: 0.938 on macOS, where the committed results were made; 0.942 on
Linux).

The fine-tune needs one H100; see [gpu/README.md](gpu/README.md). Two parts need the raw KernelSight release:
the kernel timing in `usecases/diagnosis/build.py` and the CNN pipeline and its tests. Download it once:

```sh
hf download williamhtan/kernelsight --repo-type dataset --local-dir data/kernelsight_v4
ln -s data/kernelsight_v4/kernels data/kernelsight_v4/splits .              # paths the CNN code expects
uv run --with pytest python -m pytest -q                                     # CNN tests
```

## Limits

- Diagnosis labels come from fixed rules on the counters (`usecases/diagnosis/SPEC.md`), not from experts.
- Counter averages already carry much of the answer (logistic regression 0.938 on the limiter).
- Rare limiters are weak (balanced limiter accuracy 0.503); some test classes have a single Problem.
- One H100, short single-kernel traces; one training run per recipe.

## Team

- [m7m2od](https://github.com/m7m2od): multi-task fine-tune, the four use cases and their baselines, the
  temporal controls, the HAR-CoT check, the demo page, and the TimeNet KernelSight connector
  ([OpenTSLM/TimeNet#835](https://github.com/OpenTSLM/TimeNet/pull/835)).
- [Nawaf-TBE](https://github.com/Nawaf-TBE): CNN baseline and its independent audit, dataset validation, loader and
  visualization (`src/kernel_xray/`), and the TimeNet conversion and memory studies (`docs/timenet_*`).

## Credits and licenses

Code: MIT ([LICENSE](LICENSE)).

- Data: KernelSight by williamhtan (Apache-2.0). Model: OpenTSLM-Flamingo (`opentime/tslm-nov26`).
- `flamingo/utils.py` comes from the AWS sample `opentslm-finetune-arfbench`, with marked changes; its Flamingo
  layers keep the MIT notices of OpenFlamingo and flamingo-pytorch.
- Photo in `site/`: 极客湾Geekerwan, via Wikimedia Commons, CC BY 3.0, cropped.
