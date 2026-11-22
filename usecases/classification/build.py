# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""Rows of the classification use case: one question per Snapshot, "which workload class dominates the trace?",
with the 12 options, exactly as in the class-only reference run (flamingo/finetune_kernelsight.py, runs/flamingo).

Writes runs/usecases/classification/{train,validation,test}.jsonl (the row contract of the shared brief) and
donors.json, the fixed donor of the swapped-series control for each validation and test Snapshot.

  uv run usecases/classification/build.py                     # train repeats rare classes to >= 100 rows each
  uv run usecases/classification/build.py --min-per-class 0   # ablation only: no repeats
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/usecases/classification"


def recipe(name: str):
    """A constant of finetune_kernelsight.py, read without importing it (it imports torch)."""
    tree = ast.parse((ROOT / "flamingo/finetune_kernelsight.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == name)
    return ast.literal_eval(node.value)


PROMPT, MIN_PER_CLASS = recipe("PROMPT"), recipe("MIN_PER_CLASS")


def oversampled(indices: list[int], y: list[str], minimum: int) -> list[int]:
    """finetune_kernelsight.oversampled on Snapshot indices: the same rows in the same order."""
    by_class: dict[str, list[int]] = {}
    for i in indices:
        by_class.setdefault(y[i], []).append(i)
    return [i for part in by_class.values() for i in (part * minimum)[: max(minimum, len(part))]]


def rows(data: dict, name: str, minimum: int) -> list[dict]:
    y, paths, options = data["y"].tolist(), data["paths"].tolist(), data["labels"].tolist()
    picked = np.flatnonzero(data["split"] == name).tolist()
    if name == "train" and minimum:
        picked = oversampled(picked, y, minimum)
    return [{"id": f"classification/{name}/{k:06d}", "task": "classification", "type": "workload_class",
             "path": paths[i], "channels": None, "prompt": PROMPT, "options": options, "answer": y[i],
             # NOTE: the dominant class counts Bins, so reversing or permuting the Bins keeps it
             "controls": {"reversed": y[i], "shuffled": y[i]}, "max_new_tokens": 16}
            for k, i in enumerate(picked)]


def donors(data: dict) -> dict[str, str]:
    """For each validation and test Snapshot, a Snapshot of the same split with another class (so another Problem),
    drawn uniformly with replacement. A derangement cannot work: 170 of the 291 test Snapshots are matmul and only
    121 are not."""
    rng, y, paths = np.random.default_rng(0), data["y"], data["paths"]
    chosen = {}
    for name in ("validation", "test"):
        part = np.flatnonzero(data["split"] == name)
        for i in part:
            chosen[str(paths[i])] = str(paths[rng.choice(part[y[part] != y[i]])])
    return chosen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "kernelsight.npz"))
    p.add_argument("--min-per-class", type=int, default=MIN_PER_CLASS, help="0 = no repeats in train")
    args = p.parse_args()
    data = dict(np.load(args.data))
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        part = rows(data, name, args.min_per_class)
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
        print(name, len(part), "rows,", len({r["path"] for r in part}), "Snapshots")
    (OUT / "donors.json").write_text(json.dumps(donors(data), indent=0))


if __name__ == "__main__":
    main()
