"""Segmentation of a KernelSight Snapshot as a timeline that the TSLM writes, and its scorer.

The answer lists each change of state on a grid of 16-Bin cells as '<first Bin> <state>', separated by
semicolons, e.g. '0 idle; 64 matmul; 448 idle'. A state is idle, unlabeled (a kernel without a workload
label) or an L1 workload class. A cell is busy when at least half of its Bins run a kernel; a busy cell
takes the most common class of its labelled Bins.

score(rows, outputs, control=None) decodes each text to 512 per-Bin states:
  main                     share of Bins whose state equals the answer's state (the answers score 1.0)
  main_asymmetric          main on the rows whose reversed timeline differs from the timeline (the rows where
                           the reversed control can tell an order-blind model apart)
  parsed                   share of outputs with at least one valid '<Bin> <state>' item
  bin_workload_accuracy    on the labelled Bins of the TRUE per-Bin labels (as temporal_baselines.py)
  bin_workload_macro_f1    over the classes of those Bins, pooled over rows (as temporal_baselines.py)
  bin_busy_idle_accuracy   on all 512 Bins of each row, busy = mask_any_kernel
  boundary_f1              state changes matched within 8 Bins, mean over rows
  multi_workload           the per-Bin keys on the rows whose Snapshot has more than one L1 class
The per-Bin keys use the rows whose answer is the timeline of the row's own Snapshot, forward or reversed
(control="reversed" forces reversed); n_per_bin counts them. A swapped donor's answer gets main only.

  uv run --with numpy python usecases/segmentation/score.py    # self-test on the built rows
"""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
N, W, TOLERANCE = 512, 16, 8
IDLE, UNLABELED, INVALID = -2, -1, -9

_npz = np.load(ROOT / "kernelsight.npz")
CLASSES = [str(c) for c in _npz["labels"]]
_ROW = {str(p): i for i, p in enumerate(_npz["paths"])}
_labels = np.load(ROOT / "temporal_qa/labels_cache.npz")
STATES = np.where(_labels["mask_any_kernel"].astype(bool), _labels["workload_l1"], IDLE).astype(int)
_NAME = {IDLE: "idle", UNLABELED: "unlabeled", **dict(enumerate(CLASSES))}
_CODE = {v: k for k, v in _NAME.items()}


def states(path: str) -> np.ndarray:
    """True per-Bin states of a Snapshot: IDLE, UNLABELED or a class index."""
    return STATES[_ROW[path]]


def cells(s: np.ndarray, width: int = W) -> list[int]:
    out = []
    for seg in s.reshape(-1, width):
        labelled = seg[seg >= 0]
        if 2 * np.sum(seg != IDLE) < width:
            out.append(IDLE)
        else:  # NOTE: bincount argmax breaks ties by class index, so a reversed cell keeps its state
            out.append(int(np.bincount(labelled).argmax()) if len(labelled) else UNLABELED)
    return out


def timeline(s: np.ndarray, width: int = W) -> str:
    """The answer text of per-Bin states s on cells of `width` Bins."""
    c = cells(s, width)
    return "; ".join(f"{k * width} {_NAME[v]}" for k, v in enumerate(c) if k == 0 or v != c[k - 1])


def decode(text: str) -> np.ndarray:
    """Per-Bin states of a text. A Bin takes the state of the last item that starts at or before it; Bins
    before the first valid item are INVALID. Items with an unknown state or a Bin >= 512 are skipped."""
    items = [(int(b), _CODE[w.lower()]) for b, w in re.findall(r"(\d+)\s+([A-Za-z_]+)", text)
             if int(b) < N and w.lower() in _CODE]
    pred = np.full(N, INVALID)
    for b, v in sorted(items, key=lambda item: item[0]):  # stable: a repeated Bin keeps the later item
        pred[b:] = v
    return pred


def _boundary_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    t, p = (np.flatnonzero(x[1:] != x[:-1]) + 1 for x in (truth, pred))
    if not len(t) and not len(p):
        return 1.0
    free, hits = set(range(len(t))), 0
    for b in p:
        near = [k for k in free if abs(t[k] - b) <= TOLERANCE]
        if near:
            free.remove(min(near, key=lambda k: abs(t[k] - b)))
            hits += 1
    return 0.0 if not hits else 2 * hits / (len(t) + len(p))


def _macro_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    f1 = [2 * np.sum((pred == c) & (truth == c)) / max(np.sum(pred == c) + np.sum(truth == c), 1)
          for c in np.unique(truth)]
    return float(np.mean(f1)) if f1 else 0.0


def _per_bin(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    if not pairs:
        return {"n": 0}
    truth, pred = (np.concatenate(x) for x in zip(*pairs))
    labelled = truth >= 0
    return {"n": len(pairs),
            "bin_workload_accuracy": float(np.mean(pred[labelled] == truth[labelled])),
            "bin_workload_macro_f1": _macro_f1(truth[labelled], pred[labelled]),
            "bin_busy_idle_accuracy": float(np.mean((pred != INVALID) & ((pred != IDLE) == (truth != IDLE)))),
            "boundary_f1": float(np.mean([_boundary_f1(t, p) for t, p in pairs]))}


def _truth(row: dict, control: str | None) -> np.ndarray | None:
    s = states(row["path"])
    for t in ([s[::-1]] if control == "reversed" else [s, s[::-1]]):
        if timeline(t) == row["answer"]:
            return t
    return None


def score(rows: list[dict], outputs: list[str], control: str | None = None) -> dict:
    preds = [decode(o) for o in outputs]
    hit = [float(np.mean(p == decode(r["answer"]))) for r, p in zip(rows, preds)]
    pairs = [(t, p) for r, p in zip(rows, preds) if (t := _truth(r, control)) is not None]
    multi = [(t, p) for t, p in pairs if len(np.unique(t[t >= 0])) > 1]
    by_type = {}
    for r, h in zip(rows, hit):
        by_type.setdefault(r["type"], []).append(h)
    per_bin = _per_bin(pairs)
    asymmetric = [h for r, h in zip(rows, hit) if r["controls"]["reversed"] != r["controls"]["shuffled"]]
    return {"n": len(rows), "main": float(np.mean(hit)),
            "by_type": {k: {"n": len(v), "main": float(np.mean(v))} for k, v in by_type.items()},
            "n_asymmetric": len(asymmetric), "main_asymmetric": float(np.mean(asymmetric)) if asymmetric else 0.0,
            "parsed": float(np.mean([np.any(p != INVALID) for p in preds])),
            "n_per_bin": per_bin.pop("n"), **per_bin, "multi_workload": _per_bin(multi)}


if __name__ == "__main__":
    rows = [json.loads(line) for line in open(ROOT / "runs/usecases/segmentation/test.jsonl")]
    rng = np.random.default_rng(0)
    checks = {
        "answers": score(rows, [r["answer"] for r in rows]),
        "reversed answers": score([{**r, "answer": r["controls"]["reversed"]} for r in rows],
                                  [r["controls"]["reversed"] for r in rows], control="reversed"),
        "random cells": score(rows, [timeline(np.repeat(rng.integers(-2, len(CLASSES), N // W), W))
                                     for _ in rows]),
        "another row's answer": score(rows, [rows[k]["answer"] for k in rng.permutation(len(rows))]),
        "garbage": score(rows, ["I do not know"] * len(rows)),
    }
    for name, m in checks.items():
        print(f"{name:<22} main {m['main']:.3f} parsed {m['parsed']:.2f} per-Bin n {m['n_per_bin']}"
              f" workload acc {m.get('bin_workload_accuracy', 0):.3f} F1 {m.get('bin_workload_macro_f1', 0):.3f}"
              f" busy/idle {m.get('bin_busy_idle_accuracy', 0):.3f} boundary F1 {m.get('boundary_f1', 0):.3f}")
