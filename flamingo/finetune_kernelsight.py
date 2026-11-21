"""Fine-tune OpenTSLM-Flamingo (opentime/tslm-nov26, Gemma 3 4B) on KernelSight and score it on the
test Snapshots next to two baselines: the majority class and logistic regression.

Follows the AWS example "opentslm-finetune-arfbench" (aws-samples/generative-ai-on-amazon-sagemaker):
its utils.py is copied here unchanged and supplies the model and the training recipe. Ours: one Example
per Snapshot with one series per Channel (utils.examples_from_tasks reads one series per Source), rare
classes repeated in training, the baselines, and the evaluation, which saves every output and scores the
LLM and the baselines alike.

Input: kernelsight.npz from ~/hackathon/export_kernelsight.py, the TimeNet build of
williamhtan/kernelsight with its grouped-by-Problem split (862/291/291).
Needs one CUDA GPU with bfloat16 and about 40 GB of free disk (13.7 GB download, 13.7 GB saved model).

  pip install -r requirements.txt
  mkdir -p ../runs/smoke ../runs/flamingo
  python -u finetune_kernelsight.py --data ../kernelsight.npz --out ../runs/smoke --limit 8 --epochs 1 2>&1 | tee ../runs/smoke/train.log
  python -u finetune_kernelsight.py --data ../kernelsight.npz --out ../runs/flamingo 2>&1 | tee ../runs/flamingo/train.log

The run writes run.pid to --out, so walkthrough.py can watch a run started here.
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from utils import Example, OpenTSLM, SeriesInput, _autocast, chosen_option, train

PROMPT = (
    "These are the 24 hardware counters of one GPU kernel trace, profiled on an NVIDIA H100. "
    "Which workload class dominates the trace? Answer with one of the answer_options."
)
# NOTE: 59% of the training Snapshots are matmul; without repeats the model can learn to answer only matmul.
MIN_PER_CLASS = 100


def kernelsight_examples(data: dict[str, np.ndarray]) -> dict[str, list[tuple[str, Example]]]:
    """One (release path, Example) per Snapshot: one series per Channel, named by its counter, in Record order."""
    channels, options = data["channels"].tolist(), data["labels"].tolist()
    split = {"train": [], "validation": [], "test": []}
    for x, answer, name, path in zip(data["x"], data["y"].tolist(), data["split"].tolist(), data["paths"].tolist()):
        series = [SeriesInput(values=values.astype(float).tolist(), name=c) for values, c in zip(x, channels)]
        split[name].append((path, Example(series=series, prompt=PROMPT, options=options, answer=answer, category="")))
    return split


def oversampled(examples: list[Example]) -> list[Example]:
    """Repeat the Examples of each class until the class has at least MIN_PER_CLASS of them."""
    by_class: dict[str, list[Example]] = {}
    for example in examples:
        by_class.setdefault(example.answer, []).append(example)
    return [e for part in by_class.values() for e in (part * MIN_PER_CLASS)[: max(MIN_PER_CLASS, len(part))]]


def score(answers: list[str], predictions: list[str | None]) -> dict:
    """Accuracy, balanced accuracy, macro-F1 and per-class F1 over the classes that occur in ``answers``.
    An output that names no option (None) counts as wrong."""
    classes = sorted(set(answers))
    predicted = ["<invalid>" if p is None else p for p in predictions]
    recall = [np.mean([p == c for p, a in zip(predicted, answers) if a == c]) for c in classes]
    f1 = f1_score(answers, predicted, labels=classes, average=None, zero_division=0)
    return {
        "n": len(answers),
        "accuracy": float(np.mean([p == a for p, a in zip(predicted, answers)])),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "invalid": predictions.count(None),
        "per_class_f1": dict(zip(classes, map(float, f1))),
    }


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def interval(answers: list[str], runs: list[list[str | None]], draws: int = 1000) -> dict:
    """95% bootstrap interval over the test Snapshots of each metric, averaged over ``runs`` (one per seed)."""
    rng, truth, predicted = np.random.default_rng(0), np.array(answers), [np.array(run, dtype=object) for run in runs]
    samples = []
    for _ in range(draws):
        i = rng.integers(0, len(truth), len(truth))
        drawn = [score(truth[i].tolist(), run[i].tolist()) for run in predicted]
        samples.append({key: np.mean([d[key] for d in drawn]) for key in METRICS})
    return {key: [float(np.percentile([s[key] for s in samples], q)) for q in (2.5, 97.5)] for key in METRICS}


def baselines(data: dict[str, np.ndarray]) -> dict[str, list[str]]:
    """Fit on train, predict test: the most common training class, and logistic regression on each
    Channel's mean and std (48 features)."""
    features = np.concatenate([data["x"].mean(-1), data["x"].std(-1)], axis=1)
    y, train_rows, test_rows = data["y"], data["split"] == "train", data["split"] == "test"
    majority = Counter(y[train_rows].tolist()).most_common(1)[0][0]
    scaler = StandardScaler().fit(features[train_rows])
    model = LogisticRegression(C=1, max_iter=5000).fit(scaler.transform(features[train_rows]), y[train_rows])
    return {
        "majority": [majority] * int(test_rows.sum()),
        "logistic_regression": model.predict(scaler.transform(features[test_rows])).tolist(),
    }


def record_losses(model, path: Path) -> None:
    """Write the loss of every training question to ``path`` as it happens; utils.train prints only once per epoch.
    Validation losses (model in eval mode) are not written."""
    loss = type(model).loss.__get__(model)  # NOTE: the class method, so a second call does not wrap twice
    path.write_text("")

    def recorded(*args, **kwargs):
        value = loss(*args, **kwargs)
        if model.training:
            with path.open("a") as sink:
                sink.write(json.dumps({"loss": float(value.detach().float().mean())}) + "\n")
        return value

    model.loss = recorded


def evaluate(model, pairs: list[tuple[str, Example]], path: Path) -> dict:
    """Answer each question greedily, write every output to ``path``, and score the chosen options."""
    model.eval()
    rows = []
    for index, (trace, example) in enumerate(pairs):
        with _autocast(model):
            output = model.generate(example.series, example.evaluated_question(), max_new_tokens=64)
        rows.append({"path": trace, "answer": example.answer,
                     "prediction": chosen_option(output, example.options), "output": output})
        if index % 50 == 0:
            print(json.dumps({"evaluated": index, "of": len(pairs), **rows[-1]}), flush=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    answers, predictions = [r["answer"] for r in rows], [r["prediction"] for r in rows]
    return {**score(answers, predictions), "interval_95": interval(answers, [predictions])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="hf://opentime/tslm-nov26")
    p.add_argument("--data", default="kernelsight.npz")
    p.add_argument("--out", default="runs/flamingo")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="random N examples per split (debugging only)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.pid").write_text(str(os.getpid()))
    metrics = {"args": vars(args)}

    def report(key: str, value) -> None:
        metrics[key] = value
        print(json.dumps({key: value}), flush=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    data = dict(np.load(args.data))
    split = kernelsight_examples(data)
    if args.limit:
        rng = random.Random(0)
        split = {name: rng.sample(part, min(args.limit, len(part))) for name, part in split.items()}
    report("splits", {name: dict(Counter(e.answer for _, e in part)) for name, part in split.items()})
    train_examples = oversampled([e for _, e in split["train"]])
    report("train_oversampled", dict(Counter(e.answer for e in train_examples)))
    test_answers = data["y"][data["split"] == "test"].tolist()
    report("baselines", {name: score(test_answers, p) for name, p in baselines(data).items()})

    if not torch.cuda.is_available():  # NOTE: utils._load would fall back to a float32 CPU run that takes days
        raise SystemExit("no CUDA GPU: install a CUDA build of torch")
    model = OpenTSLM.load(args.base_model).model
    torch.manual_seed(0)
    report("before", evaluate(model, split["test"], out / "predictions_before.jsonl"))
    record_losses(model, out / "losses.jsonl")
    train(model, train_examples, [e for _, e in split["validation"]], epochs=args.epochs, accumulation=8, seed=0)
    OpenTSLM(model).save(str(out / "model"))  # NOTE: before the last evaluation, so a crash there keeps the weights
    report("after", evaluate(model, split["test"], out / "predictions_after.jsonl"))


if __name__ == "__main__":
    main()
