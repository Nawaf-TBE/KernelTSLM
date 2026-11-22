"""Scorer of the classification use case (pure Python + numpy), with Problem-cluster bootstrap intervals.

  score(rows, outputs)                          accuracy, balanced accuracy (= main), macro-F1, per-class F1, invalid
  cluster_interval(rows, runs)                  95% interval of each metric, resampling Problems
  paired_difference(rows, runs_a, runs_b)       a - b with the same Problem draws: interval and P(a - b <= 0)
  swap_read(rows, outputs, donor_answers)       the swapped-series control: own class vs donor class, with chance levels

``outputs`` are raw model outputs; an output that names no option counts as wrong. ``runs`` is a list of output
lists (one per seed); a draw scores each run and averages. The 291 test Snapshots come from 84 Problems (Variants of
one kernel differ only in dtype, seed or launch options), so Snapshot-level intervals are too narrow.

  uv run --with numpy usecases/classification/score.py    # self-test on runs/usecases/classification/test.jsonl
"""

import json
import re
from pathlib import Path

import numpy as np

METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
RUNS = Path(__file__).resolve().parents[2] / "runs/usecases/classification"


def chosen(output: str, options: list[str]) -> str | None:
    """utils.chosen_option: the option that the output names after the checkpoint's normalization, else None."""
    def normalized(text: str) -> str:
        text = text.strip()
        if "answer:" in text.lower():
            text = text[text.lower().rfind("answer:") + len("answer:"):]
        return " ".join(text.strip().strip(".\n\t ").lower().split())
    wanted = normalized(output)
    return next((option for option in options if normalized(option) == wanted), None)


def metrics(answers: np.ndarray, predicted: np.ndarray) -> dict:
    """Over the classes that occur in ``answers``; ``predicted`` holds None for an invalid output."""
    classes = sorted(set(answers.tolist()))
    right = predicted == answers
    recall = [right[answers == c].mean() for c in classes]
    f1 = [2 * (right & (answers == c)).sum() / ((answers == c).sum() + (predicted == c).sum()) for c in classes]
    return {"accuracy": float(right.mean()), "balanced_accuracy": float(np.mean(recall)),
            "macro_f1": float(np.mean(f1)), "per_class_f1": dict(zip(classes, map(float, f1)))}


def predictions(rows: list[dict], outputs: list[str]) -> np.ndarray:
    return np.array([chosen(o or "", r["options"] or [r["answer"]]) for r, o in zip(rows, outputs, strict=True)],
                    dtype=object)


def score(rows: list[dict], outputs: list[str]) -> dict:
    answers, predicted = np.array([r["answer"] for r in rows], dtype=object), predictions(rows, outputs)
    types = np.array([r["type"] for r in rows])
    by_type = {t: metrics(answers[types == t], predicted[types == t]) for t in sorted(set(types.tolist()))}
    problems = np.array([problem(r["path"]) for r in rows])
    whole = metrics(answers, predicted)
    return {"n": len(rows), "main": whole["balanced_accuracy"], **whole,
            # NOTE: 4 classes have 7 test Snapshots from 1 Problem each; this weights each Problem once
            "problem_accuracy": float(np.mean([(predicted == answers)[problems == g].mean() for g in set(problems)])),
            "invalid": int(sum(p is None for p in predicted)), "by_type": by_type}


# *** Problem key, copied from the referee's checks/problems.py (the connector's grouping rule) ***

INPUT_SUFFIX = "/input/tensor_input.npz"
LAUNCH_OPTIONS = re.compile(r"_(?:r[A-Z]|s\d+|sw\d+|b\d+)(?=_|$)|_T\d+")
SAME_KERNEL = {"cutlass_ws_overlap": "cutlass_gemm"}
ALIASES = {
    "cutlass_gemm/baseline": "m4096",
    "cutlass_gemm/kheavy": "m1024_n1024_k8192",
    "cutlass_gemm/m1024": "m1024_n1024_k1024",
    "cutlass_gemm/m2048": "m2048_n2048_k2048",
    "cutlass_ws_overlap/baseline": "k2048",
    "cutlass_ws_overlap/k4096": "m512_n512_k4096",
    "cutlass_ws_overlap/m1024": "m1024_n1024_k2048",
    "cutlass_ws_overlap/m1024_k2048": "m1024_n1024_k2048",
    "cutlass_ws_overlap/m2048": "m2048_n2048_k2048",
    "cutlass_ws_overlap/m2048_k2048": "m2048_n2048_k2048",
    "cutlass_ws_overlap/m4096": "m4096_n4096_k2048",
    "cutlass_ws_overlap/m4096_k2048": "m4096_n4096_k2048",
}


def problem(path: str) -> str:
    """The Problem of a Snapshot release path: motif and shape, without dtype, seed and launch options."""
    motif = path.split("/")[1]
    variant = path[len(f"kernels/{motif}/_out/"):-len(INPUT_SUFFIX)] or "baseline"
    if motif == "kernelbench":
        return f"{motif}/{variant.split('/')[0]}"
    shape = LAUNCH_OPTIONS.sub("", variant)
    return f"{SAME_KERNEL.get(motif, motif)}/{ALIASES.get(f'{motif}/{shape}', shape)}"


# *** Problem-cluster bootstrap ***

def _drawn(rows: list[dict], runs: list[list[str]], draws: int, seed: int) -> np.ndarray:
    """[draws, len(METRICS)]: each draw takes as many Problems as there are, with replacement."""
    answers = np.array([r["answer"] for r in rows], dtype=object)
    predicted = [predictions(rows, run) for run in runs]
    groups = np.array([problem(r["path"]) for r in rows])
    members = [np.flatnonzero(groups == g) for g in sorted(set(groups.tolist()))]
    rng, out = np.random.default_rng(seed), np.empty((draws, len(METRICS)))
    for d in range(draws):
        i = np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])
        out[d] = np.mean([[metrics(answers[i], p[i])[m] for m in METRICS] for p in predicted], axis=0)
    return out


def cluster_interval(rows: list[dict], runs: list[list[str]], draws: int = 2000, seed: int = 0) -> dict:
    s = _drawn(rows, runs, draws, seed)
    return {m: [float(np.percentile(s[:, k], 2.5)), float(np.percentile(s[:, k], 97.5))] for k, m in enumerate(METRICS)}


def paired_difference(rows: list[dict], runs_a: list[list[str]], runs_b: list[list[str]],
                      draws: int = 2000, seed: int = 0) -> dict:
    """Both models on the same rows and the same Problem draws (same ``seed``), so the noise of the test set cancels."""
    point = {m: np.mean([score(rows, r)[m] for r in runs_a]) - np.mean([score(rows, r)[m] for r in runs_b])
             for m in METRICS}
    d = _drawn(rows, runs_a, draws, seed) - _drawn(rows, runs_b, draws, seed)
    return {m: {"difference": float(point[m]), "interval_95": [float(np.percentile(d[:, k], 2.5)),
                                                               float(np.percentile(d[:, k], 97.5))],
                "p_at_most_0": float(np.mean(d[:, k] <= 0))} for k, m in enumerate(METRICS)}


# *** swapped-series control ***

def swap_read(rows: list[dict], outputs: list[str], donor_answers: list[str], draws: int = 2000) -> dict:
    """``outputs`` on the swapped input: donor series, own text line. "own" = the output follows the text (mean/std),
    "donor" = it follows the series. ``chance_*`` is the share expected if the outputs were shuffled across rows
    (same answer mix, no link to the input): a share above it is reliance, not the answer prior."""
    predicted = predictions(rows, outputs)
    own, donor = np.array([r["answer"] for r in rows], dtype=object), np.array(donor_answers, dtype=object)
    donor_rows = [{**r, "answer": a} for r, a in zip(rows, donor_answers, strict=True)]

    def chance(target: np.ndarray) -> float:
        return float(sum(np.mean(predicted == c) * np.mean(target == c) for c in set(target.tolist())))

    return {"n": len(rows), "own": float(np.mean(predicted == own)), "donor": float(np.mean(predicted == donor)),
            "invalid": int(sum(p is None for p in predicted)),
            "chance_own": chance(own), "chance_donor": chance(donor),
            "own_interval_95": cluster_interval(rows, [outputs], draws)["accuracy"],
            "donor_interval_95": cluster_interval(donor_rows, [outputs], draws)["accuracy"],
            "own_balanced": score(rows, outputs)["balanced_accuracy"],
            "donor_balanced": score(donor_rows, outputs)["balanced_accuracy"]}


def _self_test():
    rows = [json.loads(line) for line in open(RUNS / "test.jsonl")]
    answers = [r["answer"] for r in rows]
    perfect = score(rows, answers)
    assert perfect["main"] == 1.0 and perfect["invalid"] == 0, perfect
    rng = np.random.default_rng(0)
    random = [str(rng.choice(r["options"])) for r in rows]
    print("answers", {m: perfect[m] for m in METRICS}, "| random", {m: round(score(rows, random)[m], 3) for m in METRICS},
          "| chance 1/12 =", round(1 / 12, 3))
    assert score(rows, ["Answer: " + a.upper() + "." for a in answers])["main"] == 1.0
    assert score(rows, ["matmul or conv"] * len(rows))["invalid"] == len(rows)
    print("cluster interval of random", cluster_interval(rows, [random], draws=200))
    print("paired answers - random", paired_difference(rows, [answers], [random], draws=200)["balanced_accuracy"])
    donors = json.loads((RUNS / "donors.json").read_text())
    answer_of = {r["path"]: r["answer"] for r in rows}
    donor_answers = [answer_of[donors[r["path"]]] for r in rows]
    for name, outputs in [("text only", answers), ("series only", donor_answers), ("random", random)]:
        read = swap_read(rows, outputs, donor_answers, draws=200)
        print("swap read,", name, {k: round(v, 3) if isinstance(v, float) else v for k, v in read.items()})


if __name__ == "__main__":
    _self_test()
