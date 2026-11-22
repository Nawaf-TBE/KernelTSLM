"""Score KernelSight explanations. Each sentence before "Answer:" is parsed with one fixed form and checked against
the facts of the Snapshot as the model saw it: original, reversed or shuffled. The condition comes from
row["condition"] when RUN sets it; otherwise from which reference text row["answer"] equals.

main = mean over rows of the row's claim precision; a row with no valid "Answer: <class>" or no parsed claim
scores 0. Also: class accuracy, claim precision per family (text / series / order), the share of claims that are
order claims, and unparsable shares. A faithful model keeps text and series claims true on shuffled input and
drops its order claims; an order-blind one keeps claiming the original order and those claims turn false.
"""

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

FACTS = Path(__file__).resolve().parents[2] / "runs/usecases/explanation/facts.json"
CONDITIONS = ("original", "reversed", "shuffled")
CLASSES = {"matmul", "conv", "activation", "normalization", "softmax", "pooling", "reduction", "attention", "loss",
           "elementwise", "memory_movement", "other"}
FORMS = {
    "level": re.compile(r"(?P<ch>.+) mean is in the (?P<dir>top|bottom) (?P<k>\d+)% of training"),
    "busy": re.compile(r"The GPU is busy in (?P<q>under 25%|25-50%|50-75%|over 75%) of Bins"),
    "n_busy": re.compile(r"The GPU has (?P<n>\d+) busy periods?"),
    "period": re.compile(r"A busy period starts every (?P<p>\d+) Bins"),
    "trend": re.compile(r"SMs Active (?P<t>rises|falls) during busy time"),
    "order": re.compile(r"Workload order: (?P<seq>[a-z_]+(?: -> [a-z_]+)+)"),
    "follows": re.compile(r"After (?P<a>[a-z_]+) comes (?P<b>an idle gap|[a-z_]+)"),
}
FAMILY = {"level": "text", "busy": "series", "n_busy": "order", "period": "order", "trend": "order", "order": "order",
          "follows": "order"}


@lru_cache(maxsize=1)
def _facts():
    return json.loads(FACTS.read_text())


def _tolerance(n):
    return 0 if n <= 3 else max(1, round(0.1 * n))


def check(kind, m, snap, cond):
    f = snap[cond]
    if kind == "level":
        pct, k = snap["pct"].get(m["ch"]), int(m["k"]) / 100
        return pct is not None and (pct >= 1 - k if m["dir"] == "top" else pct <= k)
    if kind == "busy":
        return m["q"] == snap["busy"]
    if kind == "n_busy":
        return abs(int(m["n"]) - f["n_busy"]) <= _tolerance(f["n_busy"])
    if kind == "period":
        return f["period"] is not None and abs(int(m["p"]) - f["period"]) <= _tolerance(f["period"])
    if kind == "trend":
        return m["t"] == f["trend"]
    if kind == "order":
        return m["seq"].split(" -> ") == f["order"]
    return bool(f["follows"]) and f["follows"].get(m["a"]) == m["b"]


def parse(text):
    """(class or None, [(kind, match)], unparsed sentence count)."""
    head, sep, tail = text.rpartition("Answer:")
    label = tail.strip().rstrip(".").strip() if sep else None
    claims, unparsed = [], 0
    for s in re.split(r"(?<=\.)\s+", (head if sep else text).strip()):
        s = s.strip().rstrip(".")
        if not s:
            continue
        hit = next(((k, m) for k, form in FORMS.items() if (m := form.fullmatch(s))), None)
        if hit:
            claims.append(hit)
        else:
            unparsed += 1
    return (label if label in CLASSES else None), claims, unparsed


def condition(row, snap):
    if row.get("condition") in CONDITIONS:
        return row["condition"]
    return next((c for c in CONDITIONS if snap["refs"][c] == row["answer"]), "unmatched")


def score(rows: list[dict], outputs: list[str], facts: dict | None = None) -> dict:
    facts = facts or _facts()
    per_row, by_type, conds = [], {}, Counter()
    ok = {"text": [], "series": [], "order": []}
    right, no_answer, unparsed, sentences = [], 0, 0, 0
    for row, out in zip(rows, outputs):
        snap = facts[row["path"]]
        cond = condition(row, snap)
        conds[cond] += 1
        label, claims, bad = parse(out)
        truth = [check(k, m, snap, "original" if cond == "unmatched" else cond) for k, m in claims]
        for (k, _), t in zip(claims, truth):
            ok[FAMILY[k]].append(t)
        right.append(label is not None and label == parse(row["answer"])[0])
        no_answer += label is None
        unparsed += bad
        sentences += bad + len(claims)
        value = float(np.mean(truth)) if label is not None and truth else 0.0
        per_row.append(value)
        by_type.setdefault(row["type"], []).append(value)
    n_claims = sum(len(v) for v in ok.values())
    return {"n": len(per_row), "main": float(np.mean(per_row)) if per_row else 0.0,
            "by_type": {t: float(np.mean(v)) for t, v in sorted(by_type.items())},
            "class_accuracy": float(np.mean(right)) if right else 0.0,
            "claim_precision": float(np.mean(sum(ok.values(), []))) if n_claims else None,
            "precision_by_family": {f: (round(float(np.mean(v)), 3) if v else None) for f, v in ok.items()},
            "order_claim_share": len(ok["order"]) / n_claims if n_claims else 0.0,
            "claims_per_row": n_claims / len(per_row) if per_row else 0.0,
            "unparsable": no_answer / len(per_row) if per_row else 0.0,
            "unparsed_sentences": unparsed / sentences if sentences else 0.0,
            "conditions": dict(conds)}
