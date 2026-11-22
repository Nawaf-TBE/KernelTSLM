# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""baselines.json of the classification use case: every saved class model on the same 291 test rows, scored by
score.py with Problem-cluster intervals, plus the cheap "text-only" baseline and paired differences.

  saved predictions   runs/baselines/predictions_*_grouped.jsonl (MiniRocket and CNN: seed 0 only, 1 run each like the
                      LLM), runs/flamingo/predictions_after.jsonl (class-only, 3 epochs), runs/pilot*/class_test.jsonl
                      (class rows + 2,000 temporal questions, 1 epoch; pilot 3ab was the best of 4 on test)
  text_lr             logistic regression on each Channel's mean and std as the LLM's text line prints them
                      (utils._scale + utils._statistic, 4 decimals): what the LLM can read without the series
  text_lr_balanced    the same with class weights, the counterpart of the LLM's repeated rare classes

  uv run usecases/classification/baselines.py    # ~1 min on a laptop
"""

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from score import RUNS, cluster_interval, paired_difference, score  # noqa: E402

SAVED = {
    "majority": "baselines/predictions_majority_grouped.jsonl",
    "logistic_regression": "baselines/predictions_logistic_regression_grouped.jsonl",
    "nearest_neighbor": "baselines/predictions_nearest_neighbor_grouped.jsonl",
    "minirocket_seed0": "baselines/predictions_minirocket_grouped.jsonl",
    "minirocket_shuffled_seed0": "baselines/predictions_minirocket_shuffled_grouped.jsonl",
    "cnn_seed0": "baselines/predictions_cnn_grouped.jsonl",
    "flamingo_class_only": "flamingo/predictions_after.jsonl",
    "pilot1_mix": "pilot/class_test.jsonl",
    "pilot2_mix": "pilot2/class_test.jsonl",
    "pilot3ab_mix": "pilot3ab/class_test.jsonl",
    "pilot3c_mix": "pilot3c/class_test.jsonl",
}
PILOTS = ["pilot1_mix", "pilot2_mix", "pilot3ab_mix", "pilot3c_mix"]
PAIRS = [  # (a, b): a - b
    ("flamingo_class_only", "text_lr"),
    ("flamingo_class_only", "logistic_regression"),
    ("minirocket_seed0", "flamingo_class_only"),
    ("logistic_regression", "text_lr"),
    ("text_lr_balanced", "text_lr"),
    ("pilot3ab_mix", "flamingo_class_only"),
]


def from_utils(*names: str):
    """Functions of flamingo/utils.py, run from its source without importing it (it imports torch)."""
    tree = ast.parse((ROOT / "flamingo/utils.py").read_text())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"np": np}
    exec(compile(ast.Module(defs, []), "flamingo/utils.py", "exec"), namespace)  # nosec B102: own repository file
    return [namespace[n] for n in names]


_scale, _statistic = from_utils("_scale", "_statistic")


def printed(x: np.ndarray) -> np.ndarray:
    """[Snapshots, 48]: each Channel's mean, then each Channel's std, as numbers parsed back from the text line."""
    stats = [[_scale(c.astype(float).tolist())[2:] for c in snapshot] for snapshot in x]
    return np.array([[float(_statistic(v)) for v in (*(m for m, _ in s), *(d for _, d in s))] for s in stats])


def fitted(features: np.ndarray, data: dict, **options) -> dict[str, list[str]]:
    train = data["split"] == "train"
    scaler = StandardScaler().fit(features[train])
    model = LogisticRegression(C=1, max_iter=5000, **options).fit(scaler.transform(features[train]), data["y"][train])
    return {name: model.predict(scaler.transform(features[data["split"] == name])).tolist()
            for name in ("validation", "test")}


def saved(rows: list[dict], file: str) -> list[str]:
    by_path = {r["path"]: r for r in map(json.loads, open(ROOT / "runs" / file))}
    assert set(by_path) == {r["path"] for r in rows} and all(by_path[r["path"]]["answer"] == r["answer"] for r in rows)
    return [by_path[r["path"]]["prediction"] or "" for r in rows]


def report(rows: list[dict], outputs: list[str]) -> dict:
    s = score(rows, outputs)
    return {**{k: s[k] for k in ("n", "accuracy", "balanced_accuracy", "macro_f1", "problem_accuracy", "invalid")},
            "interval_95_problem_cluster": cluster_interval(rows, [outputs]), "per_class_f1": s["per_class_f1"]}


def main():
    data = dict(np.load(ROOT / "kernelsight.npz"))
    rows = {name: [json.loads(line) for line in open(RUNS / f"{name}.jsonl")] for name in ("validation", "test")}
    for name, part in rows.items():  # NOTE: the fits below predict in npz order; the rows must be in that order too
        assert [r["path"] for r in part] == data["paths"][data["split"] == name].tolist()
    features = np.concatenate([data["x"].mean(-1), data["x"].std(-1)], axis=1)  # finetune_kernelsight.baselines
    majority = Counter(data["y"][data["split"] == "train"].tolist()).most_common(1)[0][0]
    text = printed(data["x"])
    fits = {"logistic_regression_refit": fitted(features, data), "text_lr": fitted(text, data),
            "text_lr_balanced": fitted(text, data, class_weight="balanced"),
            "majority_refit": {name: [majority] * len(part) for name, part in rows.items()}}
    test = {name: saved(rows["test"], file) for name, file in SAVED.items()}
    assert fits["logistic_regression_refit"]["test"] == test["logistic_regression"], "the LR refit must match runs/baselines"
    test |= {name: fit["test"] for name, fit in fits.items() if not name.endswith("_refit")}
    out = {
        "notes": "One run per model (seed 0 of MiniRocket and CNN; the LLM has one run per recipe). Intervals: 95%, "
                 "2,000 draws of the 84 test Problems (82 validation Problems) with replacement. problem_accuracy "
                 "weights each Problem once. Validation has no softmax (10 classes); test has 11.",
        "text_features": {"max_abs_difference_to_raw": float(np.abs(text - features).max()),
                          "nonzero_printed_as_zero": int(((text == 0) & (features != 0)).sum()), "of": int(text.size)},
        "test": {name: report(rows["test"], outputs) for name, outputs in test.items()},
        "validation": {"majority": report(rows["validation"], fits["majority_refit"]["validation"]),
                       "logistic_regression": report(rows["validation"], fits["logistic_regression_refit"]["validation"]),
                       "text_lr": report(rows["validation"], fits["text_lr"]["validation"]),
                       "text_lr_balanced": report(rows["validation"], fits["text_lr_balanced"]["validation"])},
        "test_paired_differences": {f"{a} - {b}": paired_difference(rows["test"], [test[a]], [test[b]]) for a, b in PAIRS},
    }
    out["test_paired_differences"]["mean of 4 mixed pilots - flamingo_class_only"] = paired_difference(
        rows["test"], [test[p] for p in PILOTS], [test["flamingo_class_only"]])
    (RUNS / "baselines.json").write_text(json.dumps(out, indent=1))
    for name, r in out["test"].items():
        lo, hi = r["interval_95_problem_cluster"]["balanced_accuracy"]
        print(f"test {name:26s} acc {r['accuracy']:.3f} bal {r['balanced_accuracy']:.3f} [{lo:.3f}, {hi:.3f}] "
              f"F1 {r['macro_f1']:.3f} problem-acc {r['problem_accuracy']:.3f}")
    for name, r in out["validation"].items():
        print(f"validation {name:20s} acc {r['accuracy']:.3f} bal {r['balanced_accuracy']:.3f} F1 {r['macro_f1']:.3f}")
    for name, d in out["test_paired_differences"].items():
        b = d["balanced_accuracy"]
        print(f"{name:48s} bal {b['difference']:+.3f} [{b['interval_95'][0]:+.3f}, {b['interval_95'][1]:+.3f}] "
              f"P(<=0) {b['p_at_most_0']:.3f}")


if __name__ == "__main__":
    main()
