# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""Build the temporal rows of the multi-task fine-tune (row contract: SPEC.md and flamingo/multitask.py).

Four question types. Each row passes ONE Channel, because each question token reads only the latents of the
most recent series (utils.py:196), and each answer is a function of that one series alone:

  trend             a named Channel: over its active Bins, increasing or decreasing? (the reversal-pair type)
  peak_side         a named Channel: is its highest value before or after Bin N? (N balanced, so N alone gives ~0.5)
  iteration_at_bin  'sass_control' (> 0 exactly in busy Bins): which busy period holds Bin k? (4 ranges)
  longest_gap_len   'sass_control': how long is the longest idle gap? (4 ranges)

controls.reversed / controls.shuffled = the answer of the same question on x[:, ::-1] / on x[:, PERM] with
PERM = np.random.default_rng(0).permutation(512) (the permutation of finetune_kernelsight.shuffled); null when
the question has no answer among the options there.

Writes runs/usecases/temporal/{train,validation,test}.jsonl, summary.json, and train_reversed.jsonl: the
train rows again (except peak_side) with "input": "reversed" (feed x[:, ::-1]) and the reversed answer. That file is OPT-IN: it is
only correct if the harness reverses the input of those rows; otherwise each pair would teach opposite answers
for one input.

  uv run --with numpy python usecases/temporal/build.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import problem  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/usecases/temporal"
PERM = np.random.default_rng(0).permutation(512)
BUSY = "sass_control"
INTRO = "The trace has 512 Bins in time order: Bin 0 is the start and Bin 511 is the end."
ASK = " Answer with one of the answer_options."
BUSY_TEXT = ("'sass_control' is above 0 exactly in the busy Bins, when a GPU kernel runs. "
             "A busy period is a run of consecutive busy Bins.")
ITER = [(1, 3), (4, 11), (12, 22), (23, None)]  # the training quartiles of pilots 2-3c
GAP = [(1, 4), (5, 6), (7, 21), (22, None)]
CAP = {"train": 40, "validation": 10, "test": 10}  # rows per (type, Problem); test 10 = the referee's trend fix
TRAIN_TARGET = {"iteration_at_bin": 1000}  # forward train rows; the other types keep all (after caps and balance)
PEAKS = 6  # peak_side Channels per Snapshot (before the N balance and the caps)
NOT_REVERSED = {"peak_side"}  # its reversed answers are 95% "after" (peaks sit before the centre): N would leak again


def labels(ranges):
    return [f"{lo} or more" if hi is None else f"{lo}-{hi}" for lo, hi in ranges]


def in_range(ranges, n):
    return labels(ranges)[next(i for i, (lo, hi) in enumerate(ranges) if hi is None or n <= hi)]


def runs(busy):
    d = np.diff(np.r_[0, busy.astype(int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


# ---- answer functions: (series of the shown Channel, params) -> answer or None ----
def trend(v, p):
    if v.max() <= 0:
        return None
    on = np.flatnonzero(v > 0.05 * v.max())
    if on[-1] - on[0] < 16:
        return None
    span = v[on[0]:on[-1] + 1]
    first, second = span[: len(span) // 2].mean(), span[len(span) // 2:].mean()
    if max(first, second) < 1.5 * min(first, second):
        return None
    return "decreasing" if first > second else "increasing"


def peak_side(v, p):
    return "after" if int(v.argmax()) > p["n"] else "before"


def iteration_at_bin(v, p):
    busy = v > 0
    if not busy[p["k"]]:
        return None
    starts, _ = runs(busy)
    return in_range(ITER, int((starts <= p["k"]).sum()))


def longest_gap_len(v, p):
    starts, ends = runs(v > 0)
    if len(starts) < 2:
        return None
    return in_range(GAP, int((starts[1:] - ends[:-1]).max()))


ANSWER = {"trend": trend, "peak_side": peak_side, "iteration_at_bin": iteration_at_bin, "longest_gap_len": longest_gap_len}
OPTIONS = {"trend": ["increasing", "decreasing"], "peak_side": ["before", "after"],
           "iteration_at_bin": labels(ITER), "longest_gap_len": labels(GAP)}


def row(t, path, channel, v, p, prompt):
    f = ANSWER[t]
    return {"task": "temporal", "type": t, "path": path, "channels": [channel], "prompt": prompt + ASK,
            "options": OPTIONS[t], "answer": f(v, p),
            "controls": {"reversed": f(v[::-1], p), "shuffled": f(v[PERM], p)}, "max_new_tokens": 16}


def candidates(i, x, path, channels):
    """All candidate rows of one Snapshot (forward answers present)."""
    out, rng = [], np.random.default_rng([i, 7])
    for c, name in enumerate(channels):  # trend: every Channel with a clear trend whose reversal flips it
        v = x[c]
        r = row("trend", path, name, v, {}, f"{INTRO} Look at the Bins where '{name}' is active (above 5% of its "
                f"maximum). Over those Bins, is the trend of '{name}' increasing or decreasing?")
        if r["answer"] and r["controls"]["reversed"] == ({"increasing", "decreasing"} - {r["answer"]}).pop():
            out.append(r)
    found = 0
    for c in rng.permutation(len(channels)):  # peak_side: as pilot_temporal.peak_rows, up to PEAKS Channels
        v, name = x[c], channels[c]
        peak, top = int(v.argmax()), float(v.max())
        rest = np.concatenate([v[: max(peak - 4, 0)], v[peak + 5:]])
        n = peak + int(rng.choice([-1, 1])) * int(rng.integers(16, 97))
        if top <= 0 or rest.max(initial=0) >= 0.8 * top or not 1 <= n <= 510:
            continue
        out.append(row("peak_side", path, name, v, {"n": n},
                       f"{INTRO} Does '{name}' reach its highest value before or after Bin {n}?") | {"_n": n})
        found += 1
        if found == PEAKS:
            break
    v = x[channels.index(BUSY)]
    starts, _ = runs(v > 0)
    if len(starts) >= 2:
        for k in rng.choice(np.flatnonzero(v > 0), size=min(6, int((v > 0).sum())), replace=False):
            out.append(row("iteration_at_bin", path, BUSY, v, {"k": int(k)},
                           f"{INTRO} {BUSY_TEXT} Bin {k} is busy. Counting busy periods from 1 at the start of the "
                           f"trace, which busy period contains Bin {k}?"))
        out.append(row("longest_gap_len", path, BUSY, v, {},
                       f"{INTRO} {BUSY_TEXT} An idle gap is a run of idle Bins between two busy periods. "
                       "How many Bins long is the longest idle gap?"))
    return [r for r in out if r["answer"] is not None]


def capped(rows, cap, rng):
    by = defaultdict(list)
    for r in rows:
        by[problem(r["path"])].append(r)
    return [part[j] for part in by.values() for j in rng.permutation(len(part))[:cap]]


def balanced(rows, key, rng):
    """As many rows of each answer within each key, so the key alone tells nothing about the answer."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[key(r)][r["answer"]].append(r)
    kept = []
    for sides in by.values():
        m = min(len(v) for v in sides.values()) if len(sides) == 2 else 0
        kept += [v[j] for v in sides.values() for j in rng.permutation(len(v))[:m]]
    return kept


def select(split, rows):
    """Cap rows per (type, Problem); then balance peak_side per range of N (all splits) and trend per Channel
    (train: all forward-decreasing pairs and as many increasing ones of the same Channel); cut to the targets."""
    rng, out = np.random.default_rng(0), []
    for t in ANSWER:
        part = [r for r in rows if r["type"] == t]
        if t == "trend" and split == "train":
            part = capped([r for r in part if r["answer"] == "decreasing"], CAP[split], rng) + \
                capped([r for r in part if r["answer"] == "increasing"], CAP[split], rng)
            part = balanced(part, lambda r: r["channels"][0], rng)
        else:
            part = capped(part, CAP[split], rng)
        if t == "peak_side":
            part = balanced(part, lambda r: r["_n"] // 32, rng)
        if split == "train" and t in TRAIN_TARGET:
            part = [part[j] for j in sorted(rng.permutation(len(part))[:TRAIN_TARGET[t]])]
        out += part
    return out


def summary(rows):
    out = {}
    for t in ANSWER:
        part = [r for r in rows if r["type"] == t]
        if not part:
            continue
        rev = [r["controls"]["reversed"] for r in part]
        shf = [r["controls"]["shuffled"] for r in part]
        out[t] = {"rows": len(part), "snapshots": len({r["path"] for r in part}),
                  "problems": len({problem(r["path"]) for r in part}),
                  "answers": dict(Counter(r["answer"] for r in part)),
                  "reversal_pairs (reversed answer differs)": sum(a not in (None, r["answer"]) for a, r in zip(rev, part)),
                  "shuffled_defined": sum(a is not None for a in shf),
                  "shuffled_differs": sum(a not in (None, r["answer"]) for a, r in zip(shf, part))}
    return out


def main():
    data = np.load(ROOT / "kernelsight.npz")  # numeric arrays only, no pickle
    x, channels = data["x"].astype(np.float64), data["channels"].tolist()
    paths, splits = data["paths"].tolist(), data["split"].tolist()
    cache = ROOT / "temporal_qa/labels_cache.npz"
    if cache.exists():  # the busy Channel equals the release busy mask on every Bin of every Snapshot
        assert ((x[:, channels.index(BUSY)] > 0) == np.load(cache)["mask_any_kernel"].astype(bool)).all()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in ("train", "validation", "test"):
        rows = [r for i in range(len(paths)) if splits[i] == split for r in candidates(i, x[i], paths[i], channels)]
        rows = select(split, rows)
        for n, r in enumerate(rows):
            r.pop("_n", None)
            rows[n] = {"id": f"temporal/{split}/{n:06d}", **r}
        (OUT / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        report[split] = summary(rows)
        if split == "train":
            rev = [{**r, "id": r["id"] + "r", "input": "reversed", "answer": r["controls"]["reversed"],
                    "controls": {"reversed": r["answer"], "shuffled": None}}
                   for r in rows if r["controls"]["reversed"] is not None and r["type"] not in NOT_REVERSED]
            (OUT / "train_reversed.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rev))
            report["train_reversed"] = summary(rev)
    (OUT / "summary.json").write_text(json.dumps(report, indent=1))
    for split, s in report.items():
        print(split, {t: (v["rows"], v["problems"]) for t, v in s.items()}, "total", sum(v["rows"] for v in s.values()))
        for t, v in s.items():
            print(f"   {t:17s} answers {v['answers']}  pairs {v['reversal_pairs (reversed answer differs)']}  "
                  f"shuffled defined {v['shuffled_defined']} differs {v['shuffled_differs']}")


if __name__ == "__main__":
    main()
