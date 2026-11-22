"""Score KernelSight kernel diagnoses: "<finding sentences> Limiter: <limiter>. Try: <step>".

Each finding sentence is parsed with one fixed form and checked against the facts of the Snapshot as the model saw
it (original, reversed or shuffled; runs/usecases/diagnosis/facts.json). The condition comes from row["condition"]
when RUN sets it; otherwise from which reference text row["answer"] equals.

main = limiter accuracy: the text after the last "Limiter:" (up to the next "." or "Try:") must equal the reference
limiter of that condition. An output without a valid limiter counts as wrong. Also: balanced limiter accuracy (mean
recall over the reference limiters), "Try" accuracy, claim precision of the findings (per form), claims per row,
the unparsable share, and a per-workload-class breakdown.
"""

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

FACTS = Path(__file__).resolve().parents[2] / "runs/usecases/diagnosis/facts.json"
CONDITIONS = ("original", "reversed", "shuffled")
LIMITERS = ("launch gaps", "low SM utilization", "memory bandwidth", "tensor cores", "FMA pipe", "ALU pipe",
            "XU pipe", "barrier stalls", "latency")
TRIES = {
    "launch": "Fuse kernels or use CUDA Graphs to remove launch gaps.",
    "grid": "Use more threads and blocks (a bigger grid or split-K) to fill all SMs.",
    "reuse": "Improve data reuse (tiling) to cut DRAM traffic.",
    "bytes": "Move fewer bytes: fuse with neighbouring ops or use a narrower data type.",
    "tensor": "Use FP8 or larger tiles to raise tensor-core throughput.",
    "tf32": "Use bf16 or TF32 so the tensor cores run.",
    "fma": "Cut FP32 math: fewer FLOPs or fast-math intrinsics.",
    "alu": "Cut integer and compare work: simpler index math, vector loads.",
    "xu": "Use fast-math intrinsics or fewer exp, log and divide calls.",
    "barrier": "Use fewer __syncthreads barriers, e.g. warp shuffles.",
    "coalesce": "Coalesce global loads so each access uses full cache lines.",
    "hide": "Raise occupancy or instruction-level parallelism to hide latency.",
}
STEP_KEY = {v.rstrip("."): k for k, v in TRIES.items()}
UNIT_NAMES = ("DRAM traffic", "Tensor Active", "pipe: fma", "pipe: alu", "pipe: xu", "SMs Active", "L2 hit rate")
FORMS = {
    "level": re.compile(r"(?P<name>" + "|".join(map(re.escape, UNIT_NAMES)) + r") is (?P<v>\d+)%(?: of peak)? while busy"),
    "idle": re.compile(r"The GPU is idle (?P<v>\d+)% of the active span, across (?P<n>\d+) busy periods?"),
    "rank": re.compile(r"(?P<ch>.+) is in the (?P<dir>top|bottom) (?P<k>\d+)% of training while busy"),
}
LEVEL_TOL = 10  # percentage points, for "... is N% ... while busy" and "idle N%"


@lru_cache(maxsize=1)
def _facts():
    return json.loads(FACTS.read_text())


def _tolerance(n):  # as usecases/explanation/score.py: exact up to 3, else 10%
    return 0 if n <= 3 else max(1, round(0.1 * n))


def check(kind, m, snap, cond):
    if kind == "level":
        return abs(int(m["v"]) - 100 * snap["busy"][m["name"]]) <= LEVEL_TOL
    if kind == "idle":
        t = snap[cond]
        return abs(int(m["v"]) - 100 * t["idle"]) <= LEVEL_TOL and abs(int(m["n"]) - t["n_busy"]) <= _tolerance(t["n_busy"])
    pct, k = snap["pct"].get(m["ch"]), int(m["k"]) / 100
    return pct is not None and (pct >= 1 - k if m["dir"] == "top" else pct <= k)


def parse(text):
    """(limiter or None, step key or None, [(kind, match)], unparsed sentence count)."""
    head, sep, tail = text.rpartition("Try:")
    step = STEP_KEY.get(tail.strip().split("\n")[0].strip().rstrip(".")) if sep else None
    head = head if sep else text
    findings, sep, tail = head.rpartition("Limiter:")
    limiter = re.split(r"[.\n]", tail.strip())[0].strip() if sep else None
    claims, unparsed = [], 0
    for s in re.split(r"(?<=\.)\s+", (findings if sep else head).strip()):
        s = s.strip().rstrip(".")
        if not s:
            continue
        hit = next(((k, m) for k, form in FORMS.items() if (m := form.fullmatch(s))), None)
        if hit:
            claims.append(hit)
        else:
            unparsed += 1
    return (limiter if limiter in LIMITERS else None), step, claims, unparsed


def condition(row, snap):
    if row.get("condition") in CONDITIONS:
        return row["condition"]
    return next((c for c in CONDITIONS if snap[c]["answer"] == row["answer"]), "original")


def score(rows: list[dict], outputs: list[str], facts: dict | None = None) -> dict:
    facts = facts or _facts()
    lim_ok, try_ok, by_type, by_class, recall = [], [], {}, {}, {}
    claims_ok, by_form, conds = [], {k: [] for k in FORMS}, Counter()
    no_limiter = unparsed = sentences = 0
    for row, out in zip(rows, outputs):
        snap = facts[row["path"]]
        cond = condition(row, snap)
        conds[cond] += 1
        ref = snap[cond]
        limiter, step, claims, bad = parse(out)
        right = limiter == ref["limiter"]
        lim_ok.append(right)
        try_ok.append(step == ref["try"])
        recall.setdefault(ref["limiter"], []).append(right)
        by_type.setdefault(row["type"], []).append(right)
        by_class.setdefault(snap["class"], []).append(right)
        for k, m in claims:
            t = check(k, m, snap, cond)
            claims_ok.append(t)
            by_form[k].append(t)
        no_limiter += limiter is None
        unparsed += bad
        sentences += bad + len(claims)
    n = len(lim_ok)
    mean = lambda v: float(np.mean(v)) if v else None
    return {"n": n, "main": mean(lim_ok) or 0.0,
            "by_type": {t: mean(v) for t, v in sorted(by_type.items())},
            "by_class": {c: {"n": len(v), "limiter_accuracy": mean(v)} for c, v in sorted(by_class.items())},
            "limiter_balanced_accuracy": mean([mean(v) for v in recall.values()]),
            "try_accuracy": mean(try_ok),
            "claim_precision": mean(claims_ok),
            "claim_precision_by_form": {k: (round(float(np.mean(v)), 3) if v else None) for k, v in by_form.items()},
            "claims_per_row": len(claims_ok) / n if n else 0.0,
            "unparsable": no_limiter / n if n else 0.0,
            "unparsed_sentences": unparsed / sentences if sentences else 0.0,
            "conditions": dict(conds)}
