"""Fine-tune OpenTSLM-Flamingo (opentime/tslm-nov26, Gemma 3 4B) on kernel-xray QA v1 and compare with
its verified baselines (kernel-xray results/qa_v1, official validation).

Follows the AWS example "opentslm-finetune-arfbench": its utils.py is copied here unchanged and supplies
the model, the training recipe and generation. Ours: QA v1 records -> utils.Example, and the
evaluation in kernel-xray's metrics (categorical accuracy, balanced accuracy, macro-F1; statistics MAE
and within-tolerance), which utils.accuracy does not give for numeric answers.

Protocol as kernel-xray's baselines: train on QA v1 train only, no validation-based selection (the last
epoch is kept), evaluate once on QA v1 validation.

Inputs come from ~/hackathon/qa/regen_qa_v1.py (train.jsonl, val.jsonl, traces.npz; hashes verified).

  python finetune_qa.py --qa ../qa/out --out runs/qa
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from utils import Example, OpenTSLM, SeriesInput, _autocast, chosen_option, train

OPTIONS = {"comparison": ["Counter A", "Counter B"], "temporal": ["Yes", "No", "Same bin"], "statistics": []}
TRAIN_MEDIAN = -0.19676105341530675  # kernel-xray statistics majority baseline; fills unparseable answers
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


NUMERIC = ("statistics", "count_above")


def load(qa: Path, split: str, files: tuple[str, ...]) -> list[tuple[dict, Example]]:
    """QA v1 (`{split}.jsonl`) and/or multi-v1 (`multi_{split}.jsonl`); multi-v1 records carry their own options."""
    data = np.load(qa / "traces.npz")
    names = data["names"].tolist()
    traces = dict(zip(data["paths"].tolist(), data["x"]))
    pairs = []
    for prefix in files:
        for text in (qa / f"{prefix}{split}.jsonl").read_text().splitlines():
            row = json.loads(text)
            x = traces[row["trace_id"]]
            kind = row["type"]
            # NOTE: 6 decimals; kernel-xray's tolerance 1e-6 + 1e-5*|ref| fails most 4-decimal answers.
            answer = f"{row['answer']:.6f}" if kind == "statistics" else str(row["answer"])
            options = row.get("options", OPTIONS.get(kind, []))
            series = [SeriesInput(values=x[i].tolist(), name=names[i]) for i in range(24)]
            pairs.append((row, Example(series=series, prompt=row["question"], options=options, answer=answer, category=kind)))
    return pairs


def evaluate(model, pairs: list[tuple[dict, Example]], path: Path) -> dict:
    model.eval()
    rows = []
    for row, example in pairs:
        with _autocast(model):
            output = model.generate(example.series, example.evaluated_question(), max_new_tokens=16)
        if row["type"] in NUMERIC:
            match = NUMBER.search(output)
            pred = float(match.group()) if match else None
        else:
            pred = chosen_option(output, example.options)
        rows.append({"id": row["id"], "type": row["type"], "reference": row["answer"], "prediction": pred, "output": output})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    metrics = {}
    for kind in sorted({r["type"] for r in rows} - set(NUMERIC)):
        part = [r for r in rows if r["type"] == kind]
        classes = sorted({r["reference"] for r in part})
        recall = [np.mean([r["prediction"] == c for r in part if r["reference"] == c]) for c in classes]
        f1 = []
        for c in classes:
            tp = sum(r["prediction"] == c and r["reference"] == c for r in part)
            predicted, actual = sum(r["prediction"] == c for r in part), sum(r["reference"] == c for r in part)
            f1.append(2 * tp / (predicted + actual) if predicted + actual else 0.0)
        metrics[kind] = {"n": len(part), "accuracy": float(np.mean([r["prediction"] == r["reference"] for r in part])),
                         "balanced_accuracy": float(np.mean(recall)), "macro_f1": float(np.mean(f1)),
                         "invalid": sum(r["prediction"] is None for r in part),
                         "predictions": dict(Counter(str(r["prediction"]) for r in part))}
    part = [r for r in rows if r["type"] == "statistics"]
    if part:
        ref = np.array([r["reference"] for r in part])
        pred = np.array([TRAIN_MEDIAN if r["prediction"] is None else r["prediction"] for r in part])
        error = np.abs(pred - ref)
        metrics["statistics"] = {"n": len(part), "mae": float(error.mean()), "median_absolute_error": float(np.median(error)),
                                 "rmse": float(np.sqrt((error**2).mean())),
                                 "within_tolerance": float(np.mean(error <= 1e-6 + 1e-5 * np.abs(ref))),
                                 "invalid_filled_with_train_median": sum(r["prediction"] is None for r in part)}
    part = [r for r in rows if r["type"] == "count_above"]
    if part:
        # NOTE: an unparseable count is scored as 12, half of 24; the count is reported so it is never hidden.
        error = np.abs(np.array([12 if r["prediction"] is None else r["prediction"] for r in part]) - np.array([r["reference"] for r in part]))
        metrics["count_above"] = {"n": len(part), "exact_accuracy": float(np.mean(error == 0)), "mae": float(error.mean()),
                                  "invalid_scored_as_12": sum(r["prediction"] is None for r in part)}
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="hf://opentime/tslm-nov26")
    p.add_argument("--qa", default=str(Path.home() / "hackathon/qa/out"))
    p.add_argument("--out", default="runs/qa")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-limit", type=int, default=0, help="random N training questions (time budget)")
    p.add_argument("--eval-limit", type=int, default=0, help="random N validation questions (smoke test only)")
    p.add_argument("--sets", default="qa1,multi", help="qa1 (frozen QA v1), multi (multi-v1), or both")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)
    files = tuple({"qa1": "", "multi": "multi_"}[s] for s in args.sets.split(","))
    train_pairs, val_pairs = load(Path(args.qa), "train", files), load(Path(args.qa), "val", files)
    if args.train_limit:
        train_pairs = rng.sample(train_pairs, args.train_limit)
    if args.eval_limit:
        val_pairs = rng.sample(val_pairs, args.eval_limit)
    print(json.dumps({"train": len(train_pairs), "val": len(val_pairs),
                      "val_types": Counter(r["type"] for r, _ in val_pairs)}), flush=True)

    model = OpenTSLM.load(args.base_model).model
    torch.manual_seed(0)
    before = evaluate(model, val_pairs, out / "predictions_before.jsonl")
    print(json.dumps({"before": before}), flush=True)
    train(model, [e for _, e in train_pairs], [], epochs=args.epochs, accumulation=8, seed=0)
    after = evaluate(model, val_pairs, out / "predictions_after.jsonl")
    print(json.dumps({"after": after}), flush=True)

    OpenTSLM(model).save(str(out / "model"))
    (out / "metrics.json").write_text(json.dumps({"args": vars(args), "before": before, "after": after}, indent=1))


if __name__ == "__main__":
    main()
