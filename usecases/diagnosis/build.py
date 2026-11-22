"""Build the KernelSight diagnosis rows ("Talk to Your GPU"): what limits this kernel, and what should I try?

Every number comes from the BUSY Bins (mask_any_kernel = 1) of the input as given. Percentiles are against the 862
training Snapshots only. Rules, first match wins:
  1 launch gaps         idle >= 50% of the active span (first to last busy Bin) AND >= 10 busy periods
  2 low SM utilization  SMs Active < 50% while busy
  3 a unit              the busiest of DRAM traffic (read + write, capped at 100%), Tensor Active, pipe: fma,
                        pipe: alu, pipe: xu, if it is >= 25% of peak while busy -> memory bandwidth, tensor cores,
                        FMA pipe, ALU pipe, XU pipe
  4 barrier stalls      no unit >= 25% AND stall: barrier (busy mean) in the top 10% of training
  5 latency             otherwise
Try (fixed table, one condition per limiter): memory bandwidth -> reuse if Tensor Active or pipe: fma >= 10% (a math
kernel, so data reuse exists) else bytes (a streaming kernel); FMA pipe -> tf32 if Tensor Active < 5% else fma; latency -> coalesce if gmem_coalesce_ratio is in the bottom 25% else hide.
Findings (<= 3): the limiter's own evidence first, then the first that hold of: idle share (if >= 25% and not
already said), Tensor Active (FMA limiter), L2 hit rate (memory limiter), coalescing in the bottom 25% (memory,
latency, low SM), the runner-up unit if >= 25%.

Controls: reversed = the same rules on x[:, ::-1] (identical: every fact is order-free or mirror-symmetric);
shuffled = the rules on the Bins in np.random.default_rng(0).permutation(512) (the idle share and the busy periods
change; the busy means do not).

Writes runs/usecases/diagnosis/{train,validation,test}.jsonl, facts.json (read by score.py), sanity.json,
baselines.json.

  uv run --with numpy --with scikit-learn --with tokenizers python usecases/diagnosis/build.py
"""

import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from temporal_qa.paths import LABEL_ROOT  # noqa: E402
from temporal_qa.timing import runs  # noqa: E402
from usecases.diagnosis.score import LIMITERS, TRIES, parse, score  # noqa: E402

OUT = ROOT / "runs/usecases/diagnosis"
DATA = np.load(ROOT / "kernelsight.npz")
MASK = np.load(ROOT / "temporal_qa/labels_cache.npz")["mask_any_kernel"].astype(np.int8)
CHANNELS = DATA["channels"].tolist()
CLASSES = DATA["labels"].tolist()
C = {n: k for k, n in enumerate(CHANNELS)}
SMS = "SMs Active [Throughput %]"
# NOTE: a text token reads only the latents of the most recent series (flamingo/utils.py:196). SMs Active is > 0 in
# every busy Bin and 0 in every idle Bin, so with it last the answer can read the busy timeline (gaps, busy periods)
# and divide each text-line mean by the busy share. Same choice as usecases/explanation.
ORDER = [c for c in CHANNELS if c != SMS] + [SMS]
PERM = np.random.default_rng(0).permutation(512)  # flamingo/multitask.py ORDER
GAP_SHARE, MIN_PERIODS, SM_LOW, UNIT_BUSY, TENSOR_IDLE, MATH, LOW_PCT, HIGH_PCT = .5, 10, .5, .25, .05, .1, .25, .9
UNITS = {"DRAM traffic": "memory bandwidth", "Tensor Active": "tensor cores", "pipe: fma": "FMA pipe",
         "pipe: alu": "ALU pipe", "pipe: xu": "XU pipe"}
BUCKETS = (1, 5, 10, 25)
RANKED = ("gmem_coalesce_ratio", "stall: barrier")
PROMPT = ("You are profiling this GPU kernel. A Bin is busy when a kernel runs in it. What limits its performance, "
          "and what should the programmer try next? State checked facts, then end with 'Limiter: <limiter>.' and "
          "'Try: <step>'. Limiters: " + ", ".join(LIMITERS) + ".")


def view(x, m, cond):
    if cond == "reversed":
        return x[:, ::-1], m[::-1]
    if cond == "shuffled":
        return x[:, PERM], m[PERM]
    return x, m


def busy_values(x, m):
    """Busy-Bin means as fractions of peak (L2 hit rate as a fraction of requests)."""
    b = x[:, m == 1].astype(np.float64).mean(1)
    return {"DRAM traffic": min(1.0, b[C["DRAM Read Bandwidth [Throughput %]"]] + b[C["DRAM Write Bandwidth [Throughput %]"]]),
            "Tensor Active": b[C["Tensor Active [Throughput %]"]], "pipe: fma": b[C["pipe: fma"]],
            "pipe: alu": b[C["pipe: alu"]], "pipe: xu": b[C["pipe: xu"]], "SMs Active": b[C[SMS]],
            "L2 hit rate": min(1.0, b[C["hit: l2"]])}


def timeline(m):
    r = runs(m)
    span = r[-1][1] - r[0][0]
    return {"idle": float(1 - m.sum() / span), "n_busy": len(r)}


def pct5(v):
    return int(5 * round(20 * v))


def level(name, v):
    return f"{name} is {pct5(v)}%{'' if name == 'L2 hit rate' else ' of peak'} while busy."


def idle_sentence(t):
    return (f"The GPU is idle {min(95, pct5(t['idle']))}% of the active span, across {t['n_busy']} busy "
            f"period{'s' if t['n_busy'] != 1 else ''}.")


def rank_sentence(ch, p):
    top = p >= 0.5
    k = next(b for b in BUCKETS if (p >= 1 - b / 100 if top else p <= b / 100))
    return f"{ch} is in the {'top' if top else 'bottom'} {k}% of training while busy."


def diagnose(v, pct, t):
    """(limiter, try key, finding sentences) from busy values v, busy-mean percentiles pct, timeline t."""
    units = sorted(UNITS, key=lambda u: -v[u])
    top, second = units[0], units[1]
    if t["idle"] >= GAP_SHARE and t["n_busy"] >= MIN_PERIODS:
        limiter, step, first = "launch gaps", "launch", idle_sentence(t)
    elif v["SMs Active"] < SM_LOW:
        limiter, step, first = "low SM utilization", "grid", level("SMs Active", v["SMs Active"])
    elif v[top] >= UNIT_BUSY:
        limiter, first = UNITS[top], level(top, v[top])
        step = {"memory bandwidth": "reuse" if max(v["Tensor Active"], v["pipe: fma"]) >= MATH else "bytes",
                "tensor cores": "tensor", "FMA pipe": "tf32" if v["Tensor Active"] < TENSOR_IDLE else "fma",
                "ALU pipe": "alu", "XU pipe": "xu"}[limiter]
    elif pct["stall: barrier"] >= HIGH_PCT:
        limiter, step, first = "barrier stalls", "barrier", rank_sentence("stall: barrier", pct["stall: barrier"])
    else:
        limiter, first = "latency", level(top, v[top])
        step = "coalesce" if pct["gmem_coalesce_ratio"] <= LOW_PCT else "hide"
    more = [idle_sentence(t) if t["idle"] >= LOW_PCT and limiter != "launch gaps" else None,
            level("Tensor Active", v["Tensor Active"]) if limiter == "FMA pipe" else None,
            level("L2 hit rate", v["L2 hit rate"]) if limiter == "memory bandwidth" else None,
            rank_sentence("gmem_coalesce_ratio", pct["gmem_coalesce_ratio"])
            if pct["gmem_coalesce_ratio"] <= LOW_PCT and limiter in ("memory bandwidth", "latency", "low SM utilization")
            else None,
            level(second, v[second]) if v[second] >= UNIT_BUSY and limiter != UNITS[second] else None]
    findings = [first] + [s for s in more if s and s != first][:2]
    return limiter, step, findings


def answer(limiter, step, findings):
    return " ".join(findings + [f"Limiter: {limiter}.", f"Try: {TRIES[step]}"])


def percentiles(values, train):
    """Mid-rank percentile among the training Snapshots (ties count half), as usecases/explanation/build.py."""
    ref = values[train]
    return (ref[None] < values[:, None]).mean(1) + (ref[None] == values[:, None]).mean(1) / 2


def build():
    X, split, y = DATA["x"], DATA["split"], DATA["y"].tolist()
    train = split == "train"
    busy_means = np.stack([X[i][:, MASK[i] == 1].mean(1) for i in range(len(X))])
    pct = {ch: percentiles(busy_means[:, C[ch]], train) for ch in RANKED}
    facts, rows, counters = {}, [], Counter()
    for i, path in enumerate(DATA["paths"].tolist()):
        p = {ch: float(pct[ch][i]) for ch in RANKED}
        snap = {"class": y[i], "pct": p}
        for cond in ("original", "reversed", "shuffled"):
            x, m = view(X[i], MASK[i], cond)
            v = {k: float(a) for k, a in busy_values(x, m).items()}
            t = timeline(m)
            limiter, step, findings = diagnose(v, p, t)
            snap.setdefault("busy", v)
            assert all(abs(snap["busy"][k] - v[k]) < 1e-6 for k in v)  # busy means are order-free
            snap[cond] = {**t, "limiter": limiter, "try": step, "answer": answer(limiter, step, findings)}
        facts[path] = snap
        name = str(split[i])
        rows.append({"id": f"diagnosis/{name}/{counters[name]:06d}", "task": "diagnosis",
                     "type": snap["original"]["limiter"], "path": path, "channels": ORDER, "prompt": PROMPT,
                     "options": [], "answer": snap["original"]["answer"],
                     "controls": {"reversed": snap["reversed"]["answer"], "shuffled": snap["shuffled"]["answer"]},
                     "max_new_tokens": 0})
        counters[name] += 1
    return rows, facts


# *** sanity: limiter per workload class and per source; dtype Variant pairs ***

def source(path):
    return path.split("/")[1]


def sanity(facts):
    out = {"limiter_by_class": {}, "limiter_by_source": {}, "try_overall": {}, "shuffled_changes": {}}
    paths = DATA["paths"].tolist()
    for key, fn in (("limiter_by_class", lambda p: facts[p]["class"]), ("limiter_by_source", source)):
        groups = {}
        for p in paths:
            groups.setdefault(fn(p), Counter())[facts[p]["original"]["limiter"]] += 1
        out[key] = {g: dict(c.most_common()) for g, c in sorted(groups.items())}
    out["limiter_overall"] = dict(Counter(facts[p]["original"]["limiter"] for p in paths).most_common())
    out["try_overall"] = dict(Counter(facts[p]["original"]["try"] for p in paths).most_common())
    out["shuffled_changes"] = {
        "limiter": sum(facts[p]["shuffled"]["limiter"] != facts[p]["original"]["limiter"] for p in paths),
        "answer": sum(facts[p]["shuffled"]["answer"] != facts[p]["original"]["answer"] for p in paths),
        "reversed_answer": sum(facts[p]["reversed"]["answer"] != facts[p]["original"]["answer"] for p in paths),
        "of": len(paths)}
    # KernelBench Variants of one Problem and seed that differ only in dtype
    groups = {}
    for i, p in enumerate(paths):
        m = re.search(r"kernelbench/_out/([^/]+)/dt-(fp32|bf16|fp16)_(s\d+)/", p)
        if m:
            groups.setdefault((m.group(1), m.group(3)), {})[m.group(2)] = i
    pairs = []
    for (problem, seed), g in sorted(groups.items()):
        if "fp32" not in g or "bf16" not in g:
            continue
        rec = {"problem": problem, "seed": seed}
        for dt in ("fp32", "bf16"):
            i = g[dt]
            edges = np.load(LABEL_ROOT + paths[i].replace("input/tensor_input.npz", "labels/labels.npz"))["time_edges_ns"]
            width = np.diff(edges)
            f = facts[paths[i]]
            busy_ns = float(width[MASK[i] == 1].sum())
            rec[dt] = {"class": f["class"], "limiter": f["original"]["limiter"], "busy_ms": busy_ns / 1e6,
                       "n_busy": f["original"]["n_busy"], "bin_us": float(width.mean() / 1e3),
                       "DRAM": round(f["busy"]["DRAM traffic"], 3), "Tensor": round(f["busy"]["Tensor Active"], 3)}
        pairs.append(rec)
    ratio = lambda r: r["fp32"]["busy_ms"] / r["bf16"]["busy_ms"]  # NOTE: busy periods merge when gaps < 1 Bin
    by_limiter = {}
    for r in pairs:
        by_limiter.setdefault(r["fp32"]["limiter"], []).append(ratio(r))
    out["dtype_pairs"] = {
        "n_pairs": len(pairs),
        "classes": dict(Counter(r["fp32"]["class"] for r in pairs)),
        "flip_fp32_to_bf16": dict(Counter(f"{r['fp32']['limiter']} -> {r['bf16']['limiter']}" for r in pairs).most_common()),
        "flips_to_tensor_cores": sum(r["bf16"]["limiter"] == "tensor cores" != r["fp32"]["limiter"] for r in pairs),
        "speedup_busy_time_fp32_over_bf16_by_fp32_limiter": {
            k: {"n": len(v), "median": round(float(np.median(v)), 2), "p25": round(float(np.percentile(v, 25)), 2),
                "p75": round(float(np.percentile(v, 75)), 2)} for k, v in sorted(by_limiter.items())},
        "speedup_all_median": round(float(np.median([ratio(r) for r in pairs])), 2),
        "bin_width_us_median": round(float(np.median([r[d]["bin_us"] for r in pairs for d in ("fp32", "bf16")])), 1),
        "pairs": pairs}
    return out


# *** baselines on the same test rows ***

def baselines(rows, facts):
    from sklearn.linear_model import LogisticRegression

    part = {s: [r for r in rows if r["id"].startswith(f"diagnosis/{s}/")] for s in ("train", "validation", "test")}
    index = {p: i for i, p in enumerate(DATA["paths"].tolist())}
    X = DATA["x"].astype(np.float64)
    feats = np.concatenate([X.mean(-1), X.std(-1)], 1)  # the text line's mean and std of each Channel
    tr = np.array([index[r["path"]] for r in part["train"]])
    mu, sd = feats[tr].mean(0), feats[tr].std(0) + 1e-9
    f = lambda rs: (feats[[index[r["path"]] for r in rs]] - mu) / sd
    lim = lambda rs: [facts[r["path"]]["original"]["limiter"] for r in rs]
    step = lambda rs: [facts[r["path"]]["original"]["try"] for r in rs]
    text = lambda limiters, steps: [f"Limiter: {a}. Try: {TRIES[b]}" for a, b in zip(limiters, steps)]

    majority_l = Counter(lim(part["train"])).most_common(1)[0][0]
    majority_t = Counter(step(part["train"])).most_common(1)[0][0]
    per_class = {}
    for r in part["train"]:
        per_class.setdefault(facts[r["path"]]["class"], Counter())[(facts[r["path"]]["original"]["limiter"],
                                                                    facts[r["path"]]["original"]["try"])] += 1
    cls_pick = lambda c: per_class[c].most_common(1)[0][0] if c in per_class else (majority_l, majority_t)
    minirocket = {json.loads(l)["path"]: json.loads(l)["prediction"]
                  for l in (ROOT / "runs/baselines/predictions_minirocket_grouped.jsonl").read_text().splitlines()}

    lr, chosen = {}, {}
    for target, fn in (("limiter", lim), ("try", step)):
        best = None
        for c in (0.1, 1.0, 10.0):
            model = LogisticRegression(C=c, max_iter=5000).fit(f(part["train"]), fn(part["train"]))
            acc = float(np.mean(model.predict(f(part["validation"])) == np.array(fn(part["validation"]))))
            if best is None or acc > best[0]:
                best = (acc, c, model)
        chosen[target] = {"C": best[1], "validation_accuracy": round(best[0], 3)}
        lr[target] = best[2].predict(f(part["test"])).tolist()

    test = part["test"]
    rng = np.random.default_rng(0)
    outputs = {
        "reference (self-test)": [r["answer"] for r in test],
        "random limiter and step": text(rng.choice(LIMITERS, len(test)).tolist(),
                                        rng.choice(list(TRIES), len(test)).tolist()),
        "majority limiter and step (train)": text([majority_l] * len(test), [majority_t] * len(test)),
        "workload class -> train majority (true class)": text(*zip(*[cls_pick(facts[r["path"]]["class"]) for r in test])),
        "workload class -> train majority (MiniRocket class)": text(*zip(*[cls_pick(minirocket[r["path"]]) for r in test])),
        "LR on text-line mean/std (48 features)": text(lr["limiter"], lr["try"]),
        "garbage": ["The kernel looks fine." for _ in test],
    }
    result = {"lr_choice_on_validation": chosen}
    for name, outs in outputs.items():
        result[name] = {"original": score([{**r, "condition": "original"} for r in test], outs, facts)}
    # the reference of the original input, scored on the shuffled input (order-blind answer)
    shuf = [{**r, "answer": r["controls"]["shuffled"], "condition": "shuffled"} for r in test]
    result["reference (self-test)"]["shuffled"] = score(shuf, [r["answer"] for r in shuf], facts)
    result["original reference on shuffled input (order-blind)"] = {"shuffled": score(shuf, [r["answer"] for r in test], facts)}
    result["LR on text-line mean/std (48 features)"]["shuffled"] = score(shuf, outputs["LR on text-line mean/std (48 features)"], facts)
    rev = [{**r, "answer": r["controls"]["reversed"], "condition": "reversed"} for r in test]
    result["reference (self-test)"]["reversed"] = score(rev, [r["answer"] for r in rev], facts)
    return result


def tokens(rows):
    files = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--opentime--tslm-nov26/snapshots/*/tokenizer.json"))
    if not files:
        return None
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(files[0])
    count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    index = {p: i for i, p in enumerate(DATA["paths"].tolist())}
    X = DATA["x"].astype(np.float64)
    stat = lambda v: f"{v:.4e}" if abs(v) >= 1e6 else f"{v:.4f}"
    prompt, ans = [], []
    for r in rows:
        x = X[index[r["path"]]]
        lines = " ".join(f"This is time series channel {ch}, with mean {stat(x[C[ch]].mean())} and std {stat(x[C[ch]].std())}."
                         for ch in r["channels"])
        prompt.append(count(lines + " " + r["prompt"]))
        ans.append(max(count(r["answer"]), count(r["controls"]["shuffled"])))
    return {"prompt_mean": round(float(np.mean(prompt)), 1), "prompt_max": int(max(prompt)),
            "answer_mean": round(float(np.mean(ans)), 1), "answer_max": int(max(ans))}


def main():
    rows, facts = build()
    tok = tokens(rows)
    max_new = (tok["answer_max"] + 8) if tok else 96  # + end token and slack
    for r in rows:
        r["max_new_tokens"] = max_new
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("train", "validation", "test"):
        part = [r for r in rows if r["id"].startswith(f"diagnosis/{name}/")]
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
        print(name, len(part), dict(Counter(r["type"] for r in part).most_common()),
              "shuffled answer differs:", sum(r["controls"]["shuffled"] != r["answer"] for r in part))
    (OUT / "facts.json").write_text(json.dumps(facts))
    s = sanity(facts)
    s["tokens_gemma3"] = tok
    (OUT / "sanity.json").write_text(json.dumps(s, indent=1))
    print("tokens", tok, "max_new_tokens", max_new)
    for k in ("limiter_overall", "try_overall", "shuffled_changes", "limiter_by_class", "limiter_by_source"):
        print(k, json.dumps(s[k], indent=None))
    d = {k: v for k, v in s["dtype_pairs"].items() if k != "pairs"}
    print("dtype_pairs", json.dumps(d))
    result = baselines(rows, facts)
    (OUT / "baselines.json").write_text(json.dumps(result, indent=1))
    keys = ("n", "main", "limiter_balanced_accuracy", "try_accuracy", "claim_precision", "claims_per_row", "unparsable")
    for name, by in result.items():
        if name == "lr_choice_on_validation":
            print(name, by)
            continue
        for cond, sc in by.items():
            print(f"{name:52s} {cond:9s}", {k: (round(sc[k], 3) if isinstance(sc[k], float) else sc[k]) for k in keys})


if __name__ == "__main__":
    main()
