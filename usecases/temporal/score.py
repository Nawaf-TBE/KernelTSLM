"""Score the temporal use case (pure Python + numpy).

  score(rows, outputs)                      accuracy per type, pooled, and main (= mean of the type accuracies)
  pairs(rows, real_outputs, reversed_outputs)
                                            reversal pair score: right on the real Snapshot AND on the same Snapshot
                                            reversed in time (scored against controls.reversed). Only rows whose
                                            reversed answer differs. An order-blind model scores exactly 0.
  paired(rows, outputs_a, outputs_b, answers_b=None)
                                            paired real-vs-control test: discordant counts, McNemar z, z clustered
                                            by Problem. Default answers_b = the real answers (as in the pilots).

An output that names no option counts as wrong. Self-test: python score.py [rows.jsonl]
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---- Problem key (same rule as the TimeNet connector and the referee's checks/problems.py) ----
INPUT_SUFFIX = "/input/tensor_input.npz"
_LAUNCH_OPTIONS = re.compile(r"_(?:r[A-Z]|s\d+|sw\d+|b\d+)(?=_|$)|_T\d+")
_SAME_KERNEL = {"cutlass_ws_overlap": "cutlass_gemm"}
_ALIASES = {
    "cutlass_gemm/baseline": "m4096", "cutlass_gemm/kheavy": "m1024_n1024_k8192",
    "cutlass_gemm/m1024": "m1024_n1024_k1024", "cutlass_gemm/m2048": "m2048_n2048_k2048",
    "cutlass_ws_overlap/baseline": "k2048", "cutlass_ws_overlap/k4096": "m512_n512_k4096",
    "cutlass_ws_overlap/m1024": "m1024_n1024_k2048", "cutlass_ws_overlap/m1024_k2048": "m1024_n1024_k2048",
    "cutlass_ws_overlap/m2048": "m2048_n2048_k2048", "cutlass_ws_overlap/m2048_k2048": "m2048_n2048_k2048",
    "cutlass_ws_overlap/m4096": "m4096_n4096_k2048", "cutlass_ws_overlap/m4096_k2048": "m4096_n4096_k2048",
}


def problem(path: str) -> str:
    """The Problem of a Snapshot release path: motif + shape, without dtype, seed and launch options."""
    motif = path.split("/")[1]
    variant = path[len(f"kernels/{motif}/_out/"):-len(INPUT_SUFFIX)] or "baseline"
    if motif == "kernelbench":
        return f"{motif}/{variant.split('/')[0]}"
    v = _LAUNCH_OPTIONS.sub("", variant)
    return f"{_SAME_KERNEL.get(motif, motif)}/{_ALIASES.get(f'{motif}/{v}', v)}"


# ---- parsing (same normalization as utils.chosen_option) ----
def _normalized(text: str) -> str:
    text = text.strip()
    if "answer:" in text.lower():
        text = text[text.lower().rfind("answer:") + len("answer:"):]
    return " ".join(text.strip().strip(".\n\t ").lower().split())


def chosen(output: str | None, options: list[str], answer: str) -> str | None:
    """The option the output names, or None. Free-text rows (no options) match the answer only."""
    if output is None:
        return None
    wanted = _normalized(output)
    return next((o for o in (options or [answer]) if _normalized(o) == wanted), None)


def _right(rows: list[dict], outputs: list[str], answers: list[str] | None = None) -> np.ndarray:
    answers = [r["answer"] for r in rows] if answers is None else answers
    return np.array([a is not None and chosen(o, r["options"], a) == a for r, o, a in zip(rows, outputs, answers)])


def score(rows: list[dict], outputs: list[str]) -> dict:
    assert len(rows) == len(outputs)
    right = _right(rows, outputs)
    invalid = np.array([chosen(o, r["options"], r["answer"]) is None for r, o in zip(rows, outputs)])
    by_type = {}
    for t in sorted({r["type"] for r in rows}):
        k = np.array([r["type"] == t for r in rows])
        by_type[t] = {"n": int(k.sum()), "accuracy": float(right[k].mean()), "invalid": int(invalid[k].sum()),
                      "chance": float(np.mean([1 / max(len(r["options"]), 1) for r, m in zip(rows, k) if m]))}
    return {"n": len(rows), "main": float(np.mean([v["accuracy"] for v in by_type.values()])),
            "pooled": float(right.mean()), "invalid": int(invalid.sum()), "by_type": by_type}


def _cluster_interval(values: np.ndarray, groups: list[str], draws: int = 1000) -> list[float]:
    """95% bootstrap interval of the mean, resampling Problems (not rows)."""
    if not len(values):
        return [float("nan")] * 2
    rng, keys = np.random.default_rng(0), sorted(set(groups))
    index = {g: np.flatnonzero(np.array(groups) == g) for g in keys}
    means = [values[np.concatenate([index[keys[j]] for j in rng.integers(0, len(keys), len(keys))])].mean()
             for _ in range(draws)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def pairs(rows: list[dict], real_outputs: list[str], reversed_outputs: list[str], hard_ids=None) -> dict:
    """Reversal pair score on the rows whose controls.reversed exists and differs from the answer.
    hard_ids (e.g. baselines.json "trend_hard_ids"): rows whose forward answer an order-blind model gets wrong.
    On them "detect the time arrow + answer from order-free facts" scores 0, so a score there needs trend reading."""
    keep = [i for i, r in enumerate(rows) if r["controls"]["reversed"] not in (None, r["answer"])]
    sub = [rows[i] for i in keep]
    real = _right(sub, [real_outputs[i] for i in keep])
    rev = _right(sub, [reversed_outputs[i] for i in keep], [r["controls"]["reversed"] for r in sub])
    both, groups = (real & rev).astype(float), [problem(r["path"]) for r in sub]

    def part(mask: np.ndarray) -> dict:
        g = [x for x, m in zip(groups, mask) if m]
        return {"n": int(mask.sum()), "problems": len(set(g)), "both_right": float(both[mask].mean()) if mask.any() else None,
                "real_right": float(real[mask].mean()) if mask.any() else None,
                "reversed_right": float(rev[mask].mean()) if mask.any() else None,
                "interval_95_problem": _cluster_interval(both[mask], g)}

    types = np.array([r["type"] for r in sub])
    out = {**part(np.ones(len(sub), bool)), "by_type": {t: part(types == t) for t in sorted({r["type"] for r in sub})}}
    if "trend" in out["by_type"]:  # the referee's split: 86% of forward trends increase
        answers = np.array([r["answer"] for r in sub])
        out["trend_forward_decreasing"] = part((types == "trend") & (answers == "decreasing"))
        out["trend_forward_increasing"] = part((types == "trend") & (answers == "increasing"))
    if hard_ids is not None:
        out["hard"] = part(np.array([r["id"] in set(hard_ids) for r in sub]))
    return out


def _z(right_a: np.ndarray, right_b: np.ndarray, groups: list[str]) -> dict:
    a_only, b_only = int((right_a & ~right_b).sum()), int((~right_a & right_b).sum())
    diff, n = right_a.astype(float) - right_b.astype(float), len(right_a)
    by = defaultdict(list)
    for g, v in zip(groups, diff):
        by[g].append(v)
    mu, G = diff.mean() if n else 0.0, len(by)
    resid = np.array([np.sum(np.array(v) - mu) for v in by.values()])
    se = np.sqrt(G / (G - 1) * np.sum(resid ** 2)) / n if G > 1 else 0.0
    return {"n": n, "a": float(right_a.mean()) if n else None, "b": float(right_b.mean()) if n else None,
            "only_a": a_only, "only_b": b_only,
            "z": (a_only - b_only) / np.sqrt(a_only + b_only) if a_only + b_only else 0.0,
            "z_problem": float(mu / se) if se else 0.0, "problems": G}


def paired(rows: list[dict], outputs_a: list[str], outputs_b: list[str], answers_b: list[str] | None = None) -> dict:
    """a = real outputs vs the answers; b = control outputs vs answers_b (default: the same real answers, so a
    model that uses order loses accuracy on the control). Rows where answers_b is None are left out."""
    keep = [i for i in range(len(rows)) if answers_b is None or answers_b[i] is not None]
    sub = [rows[i] for i in keep]
    a = _right(sub, [outputs_a[i] for i in keep])
    b = _right(sub, [outputs_b[i] for i in keep], None if answers_b is None else [answers_b[i] for i in keep])
    groups, types = [problem(r["path"]) for r in sub], np.array([r["type"] for r in sub])
    return {**_z(a, b, groups), "by_type": {t: _z(a[types == t], b[types == t], [g for g, k in zip(groups, types == t) if k])
                                            for t in sorted({r["type"] for r in sub})}}


def _self_test(path: Path) -> None:
    rows = [json.loads(line) for line in path.open()]
    rng = np.random.default_rng(1)
    answers = [r["answer"] for r in rows]
    assert score(rows, answers)["main"] == 1.0
    random = [r["options"][rng.integers(len(r["options"]))] for r in rows]
    s = score(rows, random)
    print("answers main 1.0; random main", round(s["main"], 3), "chance",
          round(np.mean([v["chance"] for v in s["by_type"].values()]), 3))
    p = pairs(rows, answers, answers)  # order-blind: the same output on the real and the reversed Snapshot
    assert p["both_right"] == 0.0, p
    perfect = [r["controls"]["reversed"] or r["answer"] for r in rows]
    assert pairs(rows, answers, perfect)["both_right"] == 1.0
    print("pairs: order-blind 0.0, perfect 1.0, eligible rows", p["n"], "problems", p["problems"])
    q = paired(rows, answers, random)
    print("paired answers vs random: only_a", q["only_a"], "only_b", q["only_b"], "z", round(q["z"], 1),
          "z_problem", round(q["z_problem"], 1))
    assert paired(rows, answers, answers)["z"] == 0.0
    print("self-test ok")


if __name__ == "__main__":
    _self_test(Path(sys.argv[1] if len(sys.argv) > 1 else
                    Path(__file__).resolve().parents[2] / "runs/usecases/temporal/test.jsonl"))
