# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""Build the KernelSight explanation rows: a short rationale of checked facts, then "Answer: <class>".

Facts (every one is computed, so every reference claim is true):
  text   : percentile of a Channel's mean among the 862 training Snapshots (the mean is in the text line)
  series : share of busy Bins (order-free)
  order  : number of busy periods, loop period, trend of SMs Active over busy time, workload order, what follows what
Each fact has one fixed sentence form (score.py parses them). The facts are stored for the input as given:
original, reversed (x[:, ::-1]) and shuffled (the Bins in np.random.default_rng(0).permutation(512), the
permutation of finetune_kernelsight.shuffled). The shuffled reference keeps only the order-free sentences: that is
what a faithful model can still say. The reversed control is null when every claim of the original reference stays
true on the reversed Snapshot (about half the Snapshots): there the reversal cannot tell order use apart.

Writes runs/usecases/explanation/{train,validation,test}.jsonl, facts.json (read by score.py) and baselines.json.

  uv run --with numpy python usecases/explanation/build.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from temporal_qa.sequence import _next_incl_idle, first_order  # noqa: E402
from temporal_qa.timing import a_period, runs  # noqa: E402
from usecases.explanation.score import check, parse, score  # noqa: E402

OUT = ROOT / "runs/usecases/explanation"
DATA = np.load(ROOT / "kernelsight.npz")
LABELS = np.load(ROOT / "temporal_qa/labels_cache.npz")
CLASSES = DATA["labels"].tolist()
CHANNELS = DATA["channels"].tolist()
SMS = "SMs Active [Throughput %]"
# NOTE: a text token reads only the latents of the most recent series (flamingo/utils.py:196). SMs Active is > 0
# in 100% of busy Bins and 0% of idle Bins, so it goes last: the rationale can then read the busy timeline.
ORDER = [c for c in CHANNELS if c != SMS] + [SMS]
PERM = np.random.default_rng(0).permutation(512)
IDLE = -2
BUCKETS = (1, 5, 10, 25)
PROMPT = ("Explain which workload class dominates this GPU kernel trace. State checked facts about the counters "
          "and the timeline, then end with 'Answer: <class>'. Classes: " + ", ".join(CLASSES) + ".")


def view(i, cond):
    x, m, l1 = DATA["x"][i], LABELS["mask_any_kernel"][i].astype(np.int8), LABELS["workload_l1"][i].astype(int)
    if cond == "reversed":
        return x[:, ::-1], m[::-1], l1[::-1]
    if cond == "shuffled":
        return x[:, PERM], m[PERM], l1[PERM]
    return x, m, l1


def trend(v):
    """As temporal_qa/dynamics.ans_trend: last vs first quarter of busy time, relative to the maximum."""
    if len(v) < 8 or v.max() <= 0:
        return None
    q = len(v) // 4
    r = (np.median(v[-q:]) - np.median(v[:q])) / v.max()
    return "rises" if r > 0.1 else "falls" if r < -0.1 else None


def order_facts(i, cond):
    x, m, l1 = view(i, cond)
    w = l1.copy()
    w[m == 0] = IDLE
    ops = first_order(w)
    return {"n_busy": len(runs(m)), "period": None if a_period(m, l1) is None else int(a_period(m, l1)),
            "trend": trend(x[CHANNELS.index(SMS)][m == 1]),
            "order": [CLASSES[c] for c in ops] if len(ops) >= 2 else None,
            "follows": {CLASSES[c]: _next_incl_idle(w, c) for c in ops} if len(ops) >= 2 else None}


def busy_bucket(m):
    p = m.mean()
    return "under 25%" if p < .25 else "25-50%" if p < .5 else "50-75%" if p < .75 else "over 75%"


def percentiles(means, train):
    """Mid-rank percentile of each Channel mean among the training Snapshots (ties count half)."""
    ref = means[train]
    lower = (ref[None] < means[:, None]).mean(1)
    equal = (ref[None] == means[:, None]).mean(1)
    return lower + equal / 2


def level_sentence(ch, pct):
    top = pct >= 0.5
    k = next(b for b in BUCKETS if (pct >= 1 - b / 100 if top else pct <= b / 100))
    return f"{ch} mean is in the {'top' if top else 'bottom'} {k}% of training."


def pick_channels(pct, typical):
    """Up to 2 Channels where the Snapshot is in the top or bottom 25% in the direction typical of its class,
    most class-typical first; else the most extreme Channels."""
    agree = [c for c in range(24) if abs(pct[c] - .5) >= .25 and (pct[c] - .5) * (typical[c] - .5) > 0]
    agree.sort(key=lambda c: -abs(typical[c] - .5))
    rest = sorted((c for c in range(24) if abs(pct[c] - .5) >= .25 and c not in agree), key=lambda c: -abs(pct[c] - .5))
    return (agree + rest)[:2]


def rationale(levels, busy, f, label):
    s = list(levels) + [f"The GPU is busy in {busy} of Bins."]
    if f is not None:
        s.append(f"The GPU has {f['n_busy']} busy period{'s' if f['n_busy'] != 1 else ''}.")
        if f["period"] is not None:
            s.append(f"A busy period starts every {f['period']} Bins.")
        if f["trend"] is not None:
            s.append(f"SMs Active {f['trend']} during busy time.")
        if f["order"] is not None:
            s.append("Workload order: " + " -> ".join(f["order"]) + ".")
            nxt = f["follows"][f["order"][0]]
            if nxt is not None:
                s.append(f"After {f['order'][0]} comes {nxt}.")
    return " ".join(s + [f"Answer: {label}"])


def build():
    split, y = DATA["split"], DATA["y"].tolist()
    train = split == "train"
    pct = percentiles(DATA["x"].mean(-1), train)  # Snapshot x Channel
    typical = {c: np.median(pct[train & (DATA["y"] == c)], 0) for c in set(y) if (train & (DATA["y"] == c)).any()}
    facts, rows, counters = {}, [], Counter()
    for i, path in enumerate(DATA["paths"].tolist()):
        f = {c: order_facts(i, c) for c in ("original", "reversed", "shuffled")}
        levels = [level_sentence(CHANNELS[c], pct[i, c]) for c in pick_channels(pct[i], typical[y[i]])]
        busy = busy_bucket(LABELS["mask_any_kernel"][i])
        refs = {"original": rationale(levels, busy, f["original"], y[i]),
                "reversed": rationale(levels, busy, f["reversed"], y[i]),
                "shuffled": rationale(levels, busy, None, y[i])}
        kind = ("multi" if f["original"]["order"] else "loop" if f["original"]["n_busy"] >= 4 else "plain")
        facts[path] = {"class": y[i], "type": kind, "busy": busy, "refs": refs,
                       "pct": {ch: round(float(pct[i, c]), 4) for c, ch in enumerate(CHANNELS)}, **f}
        flips = not all(check(k, m, facts[path], "reversed") for k, m in parse(refs["original"])[1])
        name = str(split[i])
        rows.append({"id": f"explanation/{name}/{counters[name]:06d}", "task": "explanation", "type": kind,
                     "path": path, "channels": ORDER, "prompt": PROMPT, "options": [], "answer": refs["original"],
                     "controls": {"reversed": refs["reversed"] if flips else None,
                                  "shuffled": refs["shuffled"]},
                     "max_new_tokens": 112})  # longest reference: 107 tokens with the end token
        counters[name] += 1
    return rows, facts, pct, typical


# *** baselines: order-blind explainers, their text is the same under every condition ***

def class_template(label, rows_train, facts, pct, typical):
    """The class's typical rationale: its 2 most class-typical Channels at the class median, the most common busy
    share, busy-period count, period, trend and workload order of its training Snapshots."""
    t = typical[label]
    levels = [level_sentence(CHANNELS[c], t[c]) for c in sorted(range(24), key=lambda c: -abs(t[c] - .5))[:2]
              if abs(t[c] - .5) >= .25]
    mine = [facts[r["path"]] for r in rows_train if facts[r["path"]]["class"] == label]
    common = lambda key: Counter(json.dumps(f["original"][key]) for f in mine).most_common(1)[0][0]
    f = {"n_busy": json.loads(common("n_busy")), "period": json.loads(common("period")),
         "trend": json.loads(common("trend")), "order": json.loads(common("order")), "follows": None}
    if f["order"]:
        f["follows"] = {f["order"][0]: Counter(g["original"]["follows"][f["order"][0]] for g in mine
                                               if g["original"]["order"] == f["order"]).most_common(1)[0][0]}
    busy = Counter(g["busy"] for g in mine).most_common(1)[0][0]
    return rationale(levels, busy, f, label)


def baselines(rows, facts, pct, typical):
    train = [r for r in rows if r["id"].startswith("explanation/train/")]
    test = [r for r in rows if r["id"].startswith("explanation/test/")]
    index = {p: i for i, p in enumerate(DATA["paths"].tolist())}
    feats = np.concatenate([DATA["x"].mean(-1), DATA["x"].std(-1)], 1)
    feats = (feats - feats[DATA["split"] == "train"].mean(0)) / (feats[DATA["split"] == "train"].std(0) + 1e-9)
    tr = np.array([index[r["path"]] for r in train])
    minirocket = {json.loads(l)["path"]: json.loads(l)["prediction"]
                  for l in (ROOT / "runs/baselines/predictions_minirocket_grouped.jsonl").read_text().splitlines()}
    outputs = {
        "reference (self-test)": None,
        "original_reference_order_blind": [r["answer"] for r in test],  # perfect facts, but always the real order
        "class_template_minirocket": [class_template(minirocket[r["path"]], train, facts, pct, typical) for r in test],
        "nearest_train_rationale": [facts[train[int(np.argmin(((feats[tr] - feats[index[r['path']]]) ** 2).sum(1)))]
                                          ["path"]]["refs"]["original"] for r in test],
        "other_snapshot_rationale": [test[(k + 97) % len(test)]["answer"] for k in range(len(test))],
        "garbage": ["The trace looks normal." for _ in test],
    }
    result = {}
    for name, outs in outputs.items():
        result[name] = {}
        for cond in ("original", "reversed", "shuffled"):
            part = [(r, k) for k, r in enumerate(test) if cond == "original" or r["controls"][cond] is not None]
            rows_c = [{**r, "answer": r["answer"] if cond == "original" else r["controls"][cond], "condition": cond}
                      for r, _ in part]
            outs_c = [row["answer"] for row in rows_c] if outs is None else [outs[k] for _, k in part]
            result[name][cond] = score(rows_c, outs_c, facts)
    return result


def main():
    rows, facts, pct, typical = build()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        part = [r for r in rows if r["id"].startswith(f"explanation/{name}/")]
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
        print(name, len(part), dict(Counter(r["type"] for r in part)),
              "reversed control:", sum(r["controls"]["reversed"] is not None for r in part))
    (OUT / "facts.json").write_text(json.dumps(facts))
    result = baselines(rows, facts, pct, typical)
    (OUT / "baselines.json").write_text(json.dumps(result, indent=1))
    keys = ("n", "main", "class_accuracy", "claim_precision", "order_claim_share")
    for name, by in result.items():
        for cond, s in by.items():
            print(f"{name:28s} {cond:9s}", {k: s[k] for k in keys}, "precision", s["precision_by_family"])


if __name__ == "__main__":
    main()
