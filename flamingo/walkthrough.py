import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fine-tuning a time-series language model on GPU kernel profiles

    **What this notebook shows.** A GPU kernel leaves a trail in its hardware counters. KernelSight records
    24 counters (Channels) over 512 time steps (Bins) for each profiled kernel run (Snapshot). We fine-tune
    **OpenTSLM-Flamingo** (Gemma 3 4B plus a time-series encoder) to read the 24 Channels of a Snapshot and
    name its **Dominant L1 class**: matmul, attention, conv, softmax, and 7 more.

    We compare the model with five baselines on the **grouped split**, where all Variants of a Problem stay in
    one partition. The official split puts Variants of one Problem in train and in test, and section 4 shows
    how much that inflates every score.

    The data comes from the TimeNet build of `williamhtan/kernelsight`. The training recipe is the AWS example
    `opentslm-finetune-arfbench`, with its `utils.py` unchanged.
    """)
    return


@app.cell(hide_code=True)
def _(BASELINES, NAMES, OUT, metric_table, mo, read_json):
    _run, _base = read_json(OUT / "metrics.json") or {}, read_json(BASELINES)
    if "after" in _run and _base:
        _rows = {"**OpenTSLM-Flamingo, fine-tuned**": _run["after"], "OpenTSLM-Flamingo, before fine-tuning": _run["before"]}
        _rows |= {NAMES[k]: v for k, v in _base["grouped"].items()}
        _glance = mo.md("### Results at a glance (test set, grouped split)\n\n" + metric_table(_rows))
    else:
        _glance = mo.callout(mo.md("**Results:** the full run (section 9) has not finished yet. This box fills in when it ends."), kind="info")
    _glance
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this notebook

    - Read from top to bottom. Each section has one idea.
    - **Guess first** means: stop and write down your guess before you run or open the next part.
    - Steps that use the GPU have a **Run button**. They never start by themselves.
    - Each GPU step saves its result in `runs/flamingo/`. If the notebook restarts, the results stay.
    """)
    return


@app.cell
def _():
    import gc
    import inspect
    import json
    import os
    import random
    import subprocess
    import sys
    from collections import Counter
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download

    from finetune_kernelsight import MIN_PER_CLASS, evaluate, kernelsight_examples, oversampled, record_losses
    from utils import OpenTSLM, OpenTSLMFlamingo, chosen_option, train
    from utils import _autocast as autocast
    return (
        Counter,
        MIN_PER_CLASS,
        OpenTSLM,
        OpenTSLMFlamingo,
        Path,
        SimpleNamespace,
        autocast,
        chosen_option,
        evaluate,
        gc,
        hf_hub_download,
        inspect,
        json,
        kernelsight_examples,
        mo,
        np,
        os,
        oversampled,
        plt,
        random,
        record_losses,
        subprocess,
        sys,
        torch,
        train,
    )


@app.cell
def _(mo):
    _root = mo.notebook_dir().parent  # the project folder: kernelsight.npz and runs/ are here
    DATA = _root / "kernelsight.npz"
    BASELINES = _root / "runs/baselines/metrics.json"
    OUT = _root / "runs/flamingo"  # the full run writes here
    WALK = OUT / "walkthrough"  # the GPU steps of this notebook write here
    WALK.mkdir(parents=True, exist_ok=True)
    BASE_MODEL = "hf://opentime/tslm-nov26"
    return BASELINES, BASE_MODEL, DATA, OUT, WALK


@app.cell
def _(json, mo, os):
    NAMES = {
        "majority": "baseline: always the most common class",
        "logistic_regression": "baseline: logistic regression (mean and std of each Channel)",
        "nearest_neighbor": "baseline: nearest neighbor",
        "minirocket": "baseline: MiniRocket",
        "cnn": "baseline: CNN (kernel-xray)",
    }
    PARTS = ["series encoder", "perceiver", "cross-attention layers", "word table (token embeddings)"]


    def read_json(path):
        return json.loads(path.read_text()) if path.exists() else None


    def read_jsonl(path):
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


    def short(path):
        """kernels/<Motif>/_out/<Variant>/input/tensor_input.npz -> <Motif>/<Variant>"""
        return path.removeprefix("kernels/").removesuffix("/input/tensor_input.npz").replace("/_out/", "/")


    def metric_table(rows):
        """A Markdown table with one row per model; with the seed std and the 95% interval when they exist."""
        lines = ["| model | accuracy | balanced accuracy | macro-F1 |", "|---|---|---|---|"]
        for name, m in rows.items():
            cells = []
            for key in ("accuracy", "balanced_accuracy", "macro_f1"):
                cell = f"{m[key]:.3f}"
                if m.get("seeds", 1) > 1:
                    cell += f" ± {m[key + '_std']:.3f}"
                if "interval_95" in m:
                    cell += " ({:.2f}–{:.2f})".format(*m["interval_95"][key])
                cells.append(cell)
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines)


    def run_alive(holder, out):
        """True while the full run's process lives."""
        if holder.get("process") is not None:
            return holder["process"].poll() is None
        try:
            os.kill(int((out / "run.pid").read_text()), 0)
            return True
        except (OSError, ValueError):
            return False


    def no_model():
        return mo.callout(mo.md("Load a model first (section 6 or 11)."), kind="warn")
    return NAMES, PARTS, metric_table, no_model, read_json, read_jsonl, run_alive, short


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Setup
    """)
    return


@app.cell
def _(BASELINES, DATA, OUT, mo, torch):
    _gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    _checks = {
        "GPU": f"{_gpu}, {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB" if _gpu
        else "none: the GPU steps (sections 6–9 and 11) cannot run here",
        "data: `kernelsight.npz`": "found" if DATA.exists() else "**missing**",
        "baselines: `runs/baselines/metrics.json`": "found" if BASELINES.exists() else "**missing**",
        "full run: `runs/flamingo/metrics.json`": "found" if (OUT / "metrics.json").exists() else "not started",
    }
    mo.md("\n".join(["| check | result |", "|---|---|", *[f"| {k} | {v} |" for k, v in _checks.items()]]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. One Snapshot

    A **Snapshot** is one profiled run of one GPU kernel: 24 **Channels** (hardware counters), each with 512
    **Bins** (time steps). Pick a test Snapshot. The name shows its Motif and its Variant.

    **Guess first:** look at the 24 plots. Which workload is it: matmul, attention, conv, softmax, ...?
    Then open the answer below the plots.
    """)
    return


@app.cell
def _(DATA, kernelsight_examples, np):
    split = kernelsight_examples(dict(np.load(DATA)))  # {"train" | "validation" | "test": [(release path, Example), ...]}
    return (split,)


@app.cell
def _(mo, short, split):
    _options = {f"{i:03d}  {short(path)}": i for i, (path, _) in enumerate(split["test"])}
    snapshot = mo.ui.dropdown(_options, value=next(iter(_options)), searchable=True, label="Test Snapshot")
    snapshot
    return (snapshot,)


@app.cell
def _(plt, snapshot, split):
    _path, _example = split["test"][snapshot.value]
    _fig, _axes = plt.subplots(6, 4, figsize=(12, 10), sharex=True)
    for _ax, _series in zip(_axes.flat, _example.series):
        _ax.plot(_series.values, linewidth=0.8)
        _ax.set_title(_series.name, fontsize=7)
        _ax.tick_params(labelsize=6)
    _fig.suptitle("24 Channels × 512 Bins", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo, snapshot, split):
    mo.accordion({"Show the answer": mo.md(f"Dominant L1 class: **{split['test'][snapshot.value][1].answer}**")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The data, and the matmul trap

    The **grouped split** puts all Variants of a Problem (all dtypes, seeds and shapes of one computation) in
    one partition. So a test Snapshot never has a near-copy in train.

    - **train**: the model learns from these.
    - **validation**: after each epoch, the run measures the loss here and keeps the best epoch.
    - **test**: scored once, at the end. Nothing is tuned on it.
    """)
    return


@app.cell
def _(Counter, mo, oversampled, split):
    _counts = {name: Counter(e.answer for _, e in part) for name, part in split.items()}
    train_examples = oversampled([e for _, e in split["train"]])
    _repeated = Counter(e.answer for e in train_examples)
    _rows = [
        f"| {c} | {_counts['train'][c]} | {_repeated[c]} | {_counts['validation'][c]} | {_counts['test'][c]} |"
        for c in sorted(_counts["train"], key=lambda c: -_counts["train"][c])
    ]
    _total = f"| **total** | {len(split['train'])} | {len(train_examples)} | {len(split['validation'])} | {len(split['test'])} |"
    mo.md("\n".join(["| Dominant L1 class | train | train, with repeats | validation | test |", "|---|---|---|---|---|", *_rows, _total]))
    return (train_examples,)


@app.cell(hide_code=True)
def _(MIN_PER_CLASS, inspect, mo, oversampled):
    mo.vstack([
        mo.md(f"""
    **What you see:** 509 of the 862 training Snapshots (59%) are matmul. A model that always answers
    "matmul" is right 58% of the time on test, and it knows nothing. The loss rewards this shortcut, so a
    model can learn it first and stop there.

    **What we do:** in training, we repeat the Snapshots of each rare class until the class has at least
    {MIN_PER_CLASS}. The 7 softmax Snapshots, for example, each appear about 14 times in one epoch. Validation
    and test stay as they are.
    """),
        mo.accordion({"Show the code (from finetune_kernelsight.py)": mo.md(f"```python\n{inspect.getsource(oversampled)}```")}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The bar to beat

    The baselines come from `baselines.py`. They use the same test Snapshots and the same scoring as the
    language model. **Balanced accuracy** is the mean of the accuracy for each class, so "always matmul"
    gets only 1 of 11 classes right: 0.091. The numbers in brackets are 95% intervals over the test
    Snapshots. "±" is the spread over 5 seeds.

    **Guess first:** which baseline wins? A linear model on two numbers per Channel, or a CNN that reads all
    512 Bins?
    """)
    return


@app.cell
def _(BASELINES, NAMES, metric_table, mo, read_json):
    _base = read_json(BASELINES)
    if _base:
        _out = mo.md(metric_table({NAMES[k]: v for k, v in _base["grouped"].items()}))
    else:
        _out = mo.callout(mo.md("`runs/baselines/metrics.json` is missing: run `uv run baselines.py` first."), kind="warn")
    _out
    return


@app.cell(hide_code=True)
def _(BASELINES, NAMES, mo, read_json):
    _base = read_json(BASELINES)
    _lines = ["| balanced accuracy | grouped split | official split | official split, train cut to 862 |", "|---|---|---|---|"]
    for _k in ("logistic_regression", "nearest_neighbor", "minirocket", "cnn"):
        _lines.append(f"| {NAMES[_k]} | " + " | ".join(f"{_base[s][_k]['balanced_accuracy']:.3f}" for s in ("grouped", "official", "official_862")) + " |")
    mo.md("**Why the grouped split?** The same models on the official split:\n\n" + "\n".join(_lines) + """

    On the official split, the models look almost perfect, because Variants of the same Problem sit in train
    and in test. The last column cuts the official train to our 862 Snapshots: the scores stay high. So the
    gap comes from the leak, not from more training data.
    """) if _base else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. What the model reads

    The model does not see pictures. It reads **text** with one `<image>` placeholder for each Channel. At
    each placeholder, it looks at that Channel's numbers through new layers (section 6).

    Before that, each Channel is scaled to mean 0 and standard deviation 1. The sentence after each
    placeholder gives the original mean and standard deviation, so the model does not lose the scale.
    The answer options come first, in a shuffled order. The question comes last.

    This uses the real `prepare()` from `utils.py`. It needs only three limits from the model's
    `config.json`, so it runs without the GPU.
    """)
    return


@app.cell
def _(BASE_MODEL, OpenTSLMFlamingo, Path, SimpleNamespace, hf_hub_download, json, mo, snapshot, split):
    _config = json.loads(Path(hf_hub_download(BASE_MODEL.removeprefix("hf://"), "config.json")).read_text())
    _encoder, _resampler = _config["encoder_config"], _config["resampler_config"]
    _limits = SimpleNamespace(patch_size=_encoder["patch_size"], max_values=_encoder["max_patches"] * _encoder["patch_size"],
                              max_media=_resampler["max_media"])
    _path, _example = split["test"][snapshot.value]
    _prompt, _values, _ = OpenTSLMFlamingo.prepare(_limits, _example.series, _example.evaluated_question())
    mo.vstack([
        mo.md(f"The Snapshot from section 2. The model reads this text and {_values.shape[0]} Channels of "
              f"{_values.shape[1]} values. The right answer is **{_example.answer}**."),
        mo.plain_text(_prompt.replace(" <image>", "\n<image>").replace("answer_options:", "\n\nanswer_options:")),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The model: a frozen reader and a trained translator

    - **Gemma 3 4B** is a reader of words. It is pretrained, and fine-tuning does **not** change its layers.
    - **The translator** turns numbers into something Gemma can read. It has three trained parts:
      - the **series encoder** cuts each Channel into short patches of 4 values and turns each patch into a vector,
      - the **perceiver** summarizes each Channel into a fixed set of vectors,
      - the **cross-attention layers** are new "side windows" between Gemma's layers. Through them, Gemma
        looks at the summaries at each `<image>`.
    - The **word table** (token embeddings) also trains. Gemma uses the same table to write its answer.

    **Guess first:** which part has the most weights? And what share of the model trains?
    """)
    return


@app.cell
def _(mo):
    load_button = mo.ui.run_button(label="Load the base model (GPU; the first time downloads 14 GB)")
    load_button
    return (load_button,)


@app.cell
def _():
    holder = {"model": None, "name": None, "process": None}  # the model in GPU memory, and the full run's process
    return (holder,)


@app.cell
def _(BASE_MODEL, OpenTSLM, PARTS, WALK, holder, json, load_button, mo, read_json, torch):
    if load_button.value:
        with mo.status.spinner("Loading the model ..."):
            holder["model"], holder["name"] = OpenTSLM.load(BASE_MODEL).model, "the base model"
        _model = holder["model"]
        _counts = {name: sum(p.numel() for p in module.parameters()) for name, module in zip(PARTS, _model.trained_modules())}
        _frozen = sum(p.numel() for p in _model.parameters()) - sum(_counts.values())
        _memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        (WALK / "weights.json").write_text(json.dumps({"Gemma layers (frozen)": _frozen, **_counts, "gpu_gb": _memory}))
    _weights = read_json(WALK / "weights.json")
    if _weights:
        _memory = _weights.pop("gpu_gb")
        _total = sum(_weights.values())
        _rows = [f"| {name} | {n / 1e6:,.1f} M | {100 * n / _total:.1f}% | {'no' if 'frozen' in name else 'yes'} |" for name, n in _weights.items()]
        _out = mo.md("\n".join(["| part | weights | share | trains? |", "|---|---|---|---|", *_rows,
                                f"| **total** | {_total / 1e6:,.1f} M | | |", "", f"GPU memory in use after loading: {_memory:.1f} GB."]))
    else:
        _out = mo.md("_Not loaded yet: click the button._")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What you see:** about 70% of the weights sit in Gemma's layers, and they stay frozen. The
    cross-attention layers are the largest trained part: one block after every second Gemma layer. The word
    table is next, because Gemma knows about 262,000 tokens. The series encoder and the perceiver are tiny:
    reading the numbers needs few weights; connecting them to Gemma needs many.

    ## 7. Before training

    The base model has seen many time series, but never a GPU counter. **Guess first:** before any
    fine-tuning, how many of these 5 Snapshots does it get right? Does it even answer with one of the options?
    """)
    return


@app.cell
def _(mo):
    before_button = mo.ui.run_button(label="Ask the model about 5 test Snapshots")
    before_button
    return (before_button,)


@app.cell
def _(WALK, before_button, evaluate, holder, mo, no_model, read_jsonl, short, split):
    _five = [next(pair for pair in split["test"] if pair[1].answer == c) for c in ("matmul", "attention", "conv", "softmax", "pooling")]
    _note = None
    if before_button.value:
        if holder["model"] is None:
            _note = no_model()
        else:
            evaluate(holder["model"], _five, WALK / "before.jsonl")
    _rows = read_jsonl(WALK / "before.jsonl")
    _table = mo.md("\n".join(["| Snapshot | right answer | the model's text | counted as |", "|---|---|---|---|",
                              *[f"| {short(r['path'])} | {r['answer']} | `{r['output'][:80]}` | {r['prediction'] or '(no option)'} |" for r in _rows]]))
    mo.vstack([x for x in (_note, _table if _rows else mo.md("_Not run yet._")) if x is not None])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How an answer counts:** the evaluation keeps the text after the last "answer:", lowercases it, and
    strips periods. Then the text must equal one option exactly. "matmul." counts as matmul. "It is a
    matmul" counts as no option.

    ## 8. Mini-run: the real training loop, small

    This is the same `train()` that the full run uses, on 32 training Snapshots (with repeats), for 2 epochs.
    One **step** averages 8 questions and then changes the weights once, so the run has 8 steps.
    Before the run, we copy one weight tensor from each part. After the run, we check which ones changed.

    **Guess first:** does the loss go down in only 8 steps? Do the frozen Gemma weights change?
    """)
    return


@app.cell
def _(mo):
    mini_button = mo.ui.run_button(label="Run the mini fine-tune (about 1–2 minutes)")
    mini_button
    return (mini_button,)


@app.cell
def _(PARTS, WALK, holder, json, mini_button, mo, no_model, plt, random, read_json, read_jsonl, record_losses, split, train, train_examples):
    _note = None
    if mini_button.value:
        if holder["model"] is None:
            _note = no_model()
        else:
            _model, _rng = holder["model"], random.Random(0)
            _trained = _model.trained_modules()
            _trained_ids = {id(p) for module in _trained for p in module.parameters()}
            _picks = {name: next(module.parameters()) for name, module in zip(PARTS, _trained)}
            _picks["Gemma layers (frozen)"] = next(p for p in _model.parameters() if id(p) not in _trained_ids)
            _before = {name: p.detach().flatten()[:100_000].float().cpu().clone() for name, p in _picks.items()}
            _epochs = []
            record_losses(_model, WALK / "mini_losses.jsonl")
            train(_model, _rng.sample(train_examples, 32), _rng.sample([e for _, e in split["validation"]], 16),
                  epochs=2, accumulation=8, seed=0, log=_epochs.append)
            _changes = {name: float((p.detach().flatten()[:100_000].float().cpu() - _before[name]).abs().max()) for name, p in _picks.items()}
            (WALK / "mini_run.json").write_text(json.dumps({"epochs": _epochs, "changes": _changes}))
            holder["name"] = "the base model after the mini-run"
    _losses, _result = [r["loss"] for r in read_jsonl(WALK / "mini_losses.jsonl")], read_json(WALK / "mini_run.json")
    if _losses and _result:
        _fig, _ax = plt.subplots(figsize=(8, 3))
        _ax.plot(_losses, ".", alpha=0.4, label="loss of one question")
        _steps = [sum(_losses[i:i + 8]) / len(_losses[i:i + 8]) for i in range(0, len(_losses), 8)]
        _ax.plot([i * 8 + 3.5 for i in range(len(_steps))], _steps, "o-", label="mean of one step (8 questions)")
        _ax.set_xlabel("training question")
        _ax.set_ylabel("loss")
        _ax.legend()
        _fig.tight_layout()
        _table = "\n".join(["| part | largest change of a weight | changed? |", "|---|---|---|",
                            *[f"| {name} | {change:.2e} | {'yes' if change > 0 else 'no'} |" for name, change in _result["changes"].items()]])
        _val = ", ".join(f"epoch {e['epoch']}: {e['val_loss']:.3f}" for e in _result["epochs"])
        _shown = mo.vstack([_fig, mo.md(_table), mo.md(f"Validation loss (16 Snapshots): {_val}.")])
    else:
        _shown = mo.md("_Not run yet._")
    mo.vstack([x for x in (_note, _shown) if x is not None])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What you see:** the loss of single questions jumps around, because some questions are harder than
    others. The mean of each step is the signal to watch. Only the trained parts changed. Gemma's weight is
    exactly the same, because the training never gives its layers to the optimizer. With only 8 steps on
    32 Snapshots, the model does not learn the task yet. The mini-run shows the mechanics, not the result.

    ## 9. The full run

    The full run does the same as the mini-run, at full size: 1,509 questions in each epoch (with repeats),
    3 epochs, and it keeps the epoch with the lowest validation loss. It runs as a separate program, so you
    can close the browser while it runs. First, it frees the model from this notebook, so the two do not
    fill the GPU at once.

    The run has 4 stages: (1) answers before training, on all 291 test Snapshots, (2) training,
    (3) save the model, (4) answers after training.

    You can also start the run in a terminal, with the command in `finetune_kernelsight.py`. Then do not
    click the button: the plot below watches that run too.
    """)
    return


@app.cell
def _(mo):
    full_button = mo.ui.run_button(label="Start the full run (1–2 hours, in the background)", kind="warn")
    full_button
    return (full_button,)


@app.cell
def _(DATA, OUT, full_button, gc, holder, mo, read_json, run_alive, subprocess, sys, torch):
    _note = None
    if full_button.value:
        if run_alive(holder, OUT):
            _note = mo.callout(mo.md("The full run is already running."), kind="info")
        elif "after" in (read_json(OUT / "metrics.json") or {}):
            _note = mo.callout(mo.md("The full run has finished. To run it again, move `runs/flamingo/` away first."), kind="info")
        else:
            holder["model"], holder["name"] = None, None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            holder["process"] = subprocess.Popen(
                [sys.executable, "finetune_kernelsight.py", "--data", str(DATA), "--out", str(OUT)],
                cwd=mo.notebook_dir(), stdout=(OUT / "train.log").open("w"), stderr=subprocess.STDOUT,
                start_new_session=True,  # NOTE: the run survives a restart of this notebook
            )
            _note = mo.callout(mo.md(f"Started (process {holder['process'].pid}). Watch it below."), kind="success")
    _note
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(options=["30s", "1m", "5m"], default_interval="1m", label="Refresh the plot")
    refresh
    return (refresh,)


@app.cell
def _(OUT, holder, json, mo, plt, read_json, read_jsonl, refresh, run_alive, train_examples):
    refresh  # NOTE: this reference lets the refresh element re-run the cell
    _metrics = read_json(OUT / "metrics.json") or {}
    _losses = [r["loss"] for r in read_jsonl(OUT / "losses.jsonl")]
    _log = (OUT / "train.log").read_text().splitlines() if (OUT / "train.log").exists() else []
    _records = []
    for _line in _log:
        try:
            _records.append(json.loads(_line))
        except json.JSONDecodeError:  # NOTE: other output, or a line the run is still writing
            pass
    _records = [r for r in _records if isinstance(r, dict)]
    _epochs = [r for r in _records if "val_loss" in r]
    _total = len(train_examples) * _metrics.get("args", {}).get("epochs", 3)
    if not _log:
        _status = "Not started."
    elif "after" in _metrics:
        _status = "**Done.** The results are in section 10."
    elif not run_alive(holder, OUT):
        _status = "**Stopped before the end.** The last lines of `runs/flamingo/train.log`:\n\n```\n" + "\n".join(_log[-15:]) + "\n```"
    elif "before" not in _metrics:
        _done = [r["evaluated"] for r in _records if "evaluated" in r]
        _status = f"Stage 1 of 4: answers before training ({_done[-1] if _done else 0} of 291)."
    elif len(_losses) < _total:
        _status = f"Stage 2 of 4: training, {len(_losses)} of {_total} questions ({100 * len(_losses) / _total:.0f}%)."
    else:
        _status = "Stages 3–4 of 4: saving the model, then answers after training."
    _shown = [mo.md(_status)]
    if _losses:
        _fig, _ax = plt.subplots(figsize=(9, 3.5))
        _ax.plot(_losses, ",", alpha=0.3, color="gray", label="loss of one question")
        _window = 64
        _smooth = [sum(_losses[max(0, i - _window):i + 1]) / len(_losses[max(0, i - _window):i + 1]) for i in range(len(_losses))]
        _ax.plot(_smooth, label=f"training loss (mean of the last {_window} questions)")
        _ax.plot([e["epoch"] * len(train_examples) for e in _epochs], [e["val_loss"] for e in _epochs], "o", color="red", label="validation loss (end of epoch)")
        _ax.set_yscale("log")
        _ax.set_xlabel("training question")
        _ax.set_ylabel("loss")
        _ax.legend()
        _fig.tight_layout()
        _shown.append(_fig)
    mo.vstack(_shown)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **How to read the curve:**

    - **It learns:** the training loss goes down, and the red validation points go down too.
    - **It stalls:** the training loss stays flat. The model does not find a pattern, or the learning rate is too small.
    - **It overfits:** the training loss keeps going down, but the validation loss goes up. The model
      remembers the training Snapshots instead of learning the classes. The run keeps the epoch with the
      lowest validation loss, so an overfit last epoch does not reach the results.

    ## 10. Results

    **Guess first:** does the fine-tuned model beat MiniRocket? Which classes are hardest for it?
    """)
    return


@app.cell
def _(BASELINES, NAMES, OUT, metric_table, mo, read_json):
    _run, _base = read_json(OUT / "metrics.json") or {}, read_json(BASELINES) or {"grouped": {}}
    if "after" in _run:
        _rows = {"**OpenTSLM-Flamingo, fine-tuned**": _run["after"], "OpenTSLM-Flamingo, before fine-tuning": _run["before"]}
        _rows |= {NAMES[k]: v for k, v in _base["grouped"].items()}
        _classes = sorted(_run["after"]["per_class_f1"])
        _models = {"fine-tuned": _run["after"], **{k: _base["grouped"][k] for k in ("minirocket", "cnn", "logistic_regression") if k in _base["grouped"]}}
        _per_class = "\n".join([
            "| F1 for each class (test Snapshots) | " + " | ".join(_models) + " |", "|---|" + "---|" * len(_models),
            *[f"| {c} | " + " | ".join(f"{m['per_class_f1'].get(c, 0):.2f}" for m in _models.values()) + " |" for c in _classes],
        ])
        _invalid = _run["after"]["invalid"]
        _out = mo.vstack([mo.md(metric_table(_rows)), mo.md(f"{_invalid} of {_run['after']['n']} answers named no option."), mo.md(_per_class)])
    else:
        _out = mo.md("_Not ready: the full run has not finished._")
    _out
    return


@app.cell
def _(OUT, np, plt, read_jsonl):
    _rows = read_jsonl(OUT / "predictions_after.jsonl")
    if _rows:
        _truth = sorted({r["answer"] for r in _rows})
        _columns = _truth + sorted({r["prediction"] or "(no option)" for r in _rows} - set(_truth))
        _matrix = np.zeros((len(_truth), len(_columns)), dtype=int)
        for _r in _rows:
            _matrix[_truth.index(_r["answer"]), _columns.index(_r["prediction"] or "(no option)")] += 1
        _fig, _ax = plt.subplots(figsize=(7, 6))
        _ax.imshow(np.log1p(_matrix), cmap="Blues")
        for (_i, _j), _n in np.ndenumerate(_matrix):
            if _n:
                _ax.text(_j, _i, _n, ha="center", va="center", fontsize=8)
        _ax.set_xticks(range(len(_columns)), _columns, rotation=60, ha="right", fontsize=8)
        _ax.set_yticks(range(len(_truth)), _truth, fontsize=8)
        _ax.set_xlabel("the model's answer")
        _ax.set_ylabel("right answer")
        _ax.set_title("Fine-tuned model, test Snapshots")
        _fig.tight_layout()
        _shown = _fig
    else:
        _shown = None
    _shown
    return


@app.cell
def _(OUT, mo, read_jsonl, short):
    _wrong = [r for r in read_jsonl(OUT / "predictions_after.jsonl") if r["prediction"] != r["answer"]]
    if _wrong:
        _lines = ["| Snapshot | right answer | the model's text |", "|---|---|---|"]
        _lines += [f"| {short(r['path'])} | {r['answer']} | `{r['output'][:60]}` |" for r in _wrong[:20]]
        _shown = mo.md(f"**Read the raw answers before you trust a metric.** {len(_wrong)} wrong answers; the first 20:\n\n" + "\n".join(_lines))
    else:
        _shown = None
    _shown
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Your turn

    Load the fine-tuned model, pick any test Snapshot, and ask it. Wait until the full run has finished:
    the run saves the model before its last stage.
    """)
    return


@app.cell
def _(mo):
    trained_button = mo.ui.run_button(label="Load the fine-tuned model (GPU)")
    trained_button
    return (trained_button,)


@app.cell
def _(OUT, OpenTSLM, holder, mo, run_alive, trained_button):
    _note = None
    if trained_button.value:
        if run_alive(holder, OUT) or not (OUT / "model").exists():
            _note = mo.callout(mo.md("Wait until the full run has finished."), kind="warn")
        else:
            with mo.status.spinner("Loading the fine-tuned model ..."):
                holder["model"], holder["name"] = OpenTSLM.load(str(OUT / "model")).model, "the fine-tuned model"
    _note
    return


@app.cell
def _(mo, short, split):
    ask_snapshot = mo.ui.dropdown({f"{i:03d}  {short(path)}": i for i, (path, _) in enumerate(split["test"])},
                                  value=f"000  {short(split['test'][0][0])}", searchable=True, label="Test Snapshot")
    ask_button = mo.ui.run_button(label="Ask: which workload class dominates?")
    mo.hstack([ask_snapshot, ask_button], justify="start")
    return ask_button, ask_snapshot


@app.cell
def _(ask_button, ask_snapshot, autocast, chosen_option, holder, mo, no_model, split):
    _shown = None
    if ask_button.value:
        if holder["model"] is None:
            _shown = no_model()
        else:
            _path, _example = split["test"][ask_snapshot.value]
            with autocast(holder["model"]):
                _output = holder["model"].generate(_example.series, _example.evaluated_question(), max_new_tokens=64)
            _choice = chosen_option(_output, _example.options)
            _shown = mo.md(f"**{holder['name']}** wrote `{_output}`. Counted as **{_choice or '(no option)'}**. "
                           f"The right answer is **{_example.answer}**: {'right' if _choice == _example.answer else 'wrong'}.")
    _shown
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **An experiment, with a warning.** The model trained on one question only. Ask it anything else about
    the same Snapshot, but expect weak or odd answers. Fine-tuning on one question does not teach the
    model to explain, to count or to compare. This cell shows those limits.
    """), kind="warn")
    return


@app.cell
def _(mo):
    question = mo.ui.text_area(value="Which Channels show that this kernel uses the Tensor Cores?", label="Your question", full_width=True)
    free_button = mo.ui.run_button(label="Ask your question about the same Snapshot")
    mo.vstack([question, free_button])
    return free_button, question


@app.cell
def _(ask_snapshot, autocast, free_button, holder, mo, no_model, question, split):
    _shown = None
    if free_button.value:
        if holder["model"] is None:
            _shown = no_model()
        else:
            _path, _example = split["test"][ask_snapshot.value]
            with autocast(holder["model"]):
                _output = holder["model"].generate(_example.series, question.value, max_new_tokens=128)
            _shown = mo.md(f"**{holder['name']}** wrote:\n\n> {_output}")
    _shown
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md("## 12. Check yourself\n\nAnswer each question in your own words first. Then open it."),
        mo.accordion({
            "1. Which parts of the model change during fine-tuning, and which stay frozen?": mo.md(
                "The series encoder, the perceiver, the cross-attention layers and the word table train. "
                "Gemma's layers stay frozen: the mini-run showed that their weights did not change."),
            "2. Why do we repeat rare classes in training?": mo.md(
                "59% of the training Snapshots are matmul. Without repeats, the model can lower the loss by always "
                "answering matmul. With at least 100 questions for each class, the rare classes count too."),
            "3. Why do we use the grouped split, not the official split?": mo.md(
                "The official split puts Variants of one Problem in train and in test. Then a model can recognize a "
                "near-copy instead of the workload. MiniRocket gets 0.999 balanced accuracy there, but 0.81 on the "
                "grouped split."),
            "4. What is the validation set for, and what is the test set for?": mo.md(
                "Validation picks the best epoch (the lowest validation loss). Test gives the final score once. "
                "Nothing is chosen on test, so its score is fair."),
            "5. The training loss goes down, but the validation loss goes up. What does that mean, and what does the run do?": mo.md(
                "The model overfits: it remembers training Snapshots. The run keeps the epoch with the lowest "
                "validation loss, so the overfit epochs do not reach the results."),
        }, multiple=True),
    ])
    return


if __name__ == "__main__":
    app.run()
