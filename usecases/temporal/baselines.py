# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""CPU baselines for the temporal rows, all on the SAME test rows (runs/usecases/temporal/test.jsonl).

  majority          most common train answer per type
  text_only         reads only the prompt: trend -> the Channel's train majority; peak_side -> majority per range
                    of 32 values of N, and the pilot rule "after when N < 205"; iteration_at_bin -> majority per
                    range of 32 values of k; longest_gap_len has a constant prompt (= majority)
  order_blind       gradient boosting on order-free features of the shown series (33 quantiles, mean, std, busy
                    share) + the number in the prompt + the Channel. Same output for real, reversed and shuffled.
  order_aware       the same learner + 64 means of 8 Bins (time order kept), trained on train + train_reversed
  trend_arrow_blind a perfect time-arrow detector + the order_blind forward answer, no trend reading: the strongest
                    shortcut we can compute (the referee's "arrow + the Channel's usual answer" is void here, because
                    train trend rows are balanced per Channel). trend_hard_ids = the trend rows it gets wrong.

  uv run usecases/temporal/baselines.py      (~1 min)
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build import PERM  # noqa: E402
from score import paired, pairs, score  # noqa: E402

ROOT = HERE.parents[1]
OUT = ROOT / "runs/usecases/temporal"
DATA = np.load(ROOT / "kernelsight.npz")  # numeric arrays only, no pickle
CH = DATA["channels"].tolist()
X = dict(zip(DATA["paths"].tolist(), DATA["x"].astype(np.float64)))


def read(name: str) -> list[dict]:
    return [json.loads(line) for line in (OUT / f"{name}.jsonl").open()]


def series(row: dict, view: str) -> np.ndarray:
    v = X[row["path"]][CH.index(row["channels"][0])]
    if row.get("input") == "reversed":
        v = v[::-1]
    return {"real": v, "reversed": v[::-1], "shuffled": v[PERM]}[view]


def number(row: dict) -> int:
    m = re.search(r"Bin (\d+)\?", row["prompt"])
    return int(m.group(1)) if m else -1


def blind(row: dict, view: str) -> list[float]:
    v = np.sort(series(row, view))  # sorted first: exactly the same features for any order of the Bins
    z = (v - v.mean()) / (v.std() or 1.0)
    return [*np.quantile(z, np.linspace(0, 1, 33)), np.log1p(abs(v.mean())), np.log1p(v.std()), (v > 0).mean(),
            number(row), CH.index(row["channels"][0])]


def aware(row: dict, view: str) -> list[float]:
    v = series(row, view)
    z = (v - v.mean()) / (v.std() or 1.0)
    return [*z.reshape(64, 8).mean(1), *blind(row, view)]


def learned(features, train: list[dict], test: list[dict]) -> dict[str, list[str]]:
    """Fit one model per type; predict the test rows on the real, reversed and shuffled Snapshot."""
    out = {view: [None] * len(test) for view in ("real", "reversed", "shuffled")}
    for t in sorted({r["type"] for r in test}):
        tr = [r for r in train if r["type"] == t]
        model = HistGradientBoostingClassifier(random_state=0, max_iter=200).fit(
            np.array([features(r, "real") for r in tr]), [r["answer"] for r in tr])
        idx = [i for i, r in enumerate(test) if r["type"] == t]
        for view in out:
            for i, p in zip(idx, model.predict(np.array([features(test[i], view) for i in idx]))):
                out[view][i] = str(p)
    return out


def table(name: str, test: list[dict], outputs: dict[str, list[str]], hard=None) -> dict:
    """score on the real Snapshot; reversal pair score; paired real vs shuffled (against the real answers)."""
    shuffled_rows = [i for i, r in enumerate(test) if r["controls"]["shuffled"] is not None]
    sub = [{**test[i], "answer": test[i]["controls"]["shuffled"]} for i in shuffled_rows]
    return {"what": name, "real": score(test, outputs["real"]),
            "reversal_pairs": pairs(test, outputs["real"], outputs["reversed"], hard),
            "real_vs_shuffled": paired(test, outputs["real"], outputs["shuffled"]),
            "shuffled_vs_control_answer": score(sub, [outputs["shuffled"][i] for i in shuffled_rows])}


def same(outputs: list[str]) -> dict[str, list[str]]:
    return {"real": outputs, "reversed": outputs, "shuffled": outputs}


def main():
    train, reversed_, test = read("train"), read("train_reversed"), read("test")
    report = {"test_rows": len(test), "note": "same rows and scorer as the LLM; one run each, CPU"}

    majority = {t: Counter(r["answer"] for r in train if r["type"] == t).most_common(1)[0][0] for t in {r["type"] for r in train}}
    report["majority"] = table("most common train answer per type", test, same([majority[r["type"]] for r in test]))

    def key(r):
        return (r["type"], r["channels"][0] if r["type"] == "trend" else number(r) // 32)
    counts = defaultdict(Counter)
    for r in train:
        counts[key(r)][r["answer"]] += 1
    text = [counts[key(r)].most_common(1)[0][0] if counts[key(r)] else majority[r["type"]] for r in test]
    report["text_only"] = table("prompt only: trend Channel, N or k in ranges of 32", test, same(text))
    pilot_rule = [("after" if number(r) < 205 else "before") if r["type"] == "peak_side" else o for r, o in zip(test, text)]
    report["text_only"]["peak_side_pilot_rule_N_lt_205"] = score(
        [r for r in test if r["type"] == "peak_side"], [o for r, o in zip(test, pilot_rule) if r["type"] == "peak_side"])

    blind_out = learned(blind, train, test)
    hard = [r["id"] for r, o in zip(test, blind_out["real"]) if r["type"] == "trend" and o != r["answer"]]
    report["trend_hard_ids"] = hard
    report["order_blind"] = table("gradient boosting, order-free features + prompt number", test, blind_out, hard)
    report["order_aware"] = table("gradient boosting, 64 Bin-chunk means + the same, train + train_reversed", test,
                                  learned(aware, train + reversed_, test), hard)

    trend = [r for r in test if r["type"] == "trend"]
    fwd = [o for r, o in zip(test, blind_out["real"]) if r["type"] == "trend"]
    flip = {"increasing": "decreasing", "decreasing": "increasing"}
    report["trend_arrow_blind"] = {"what": "perfect time-arrow detector + the order_blind forward answer (no trend reading); "
                                           "it scores 0 on the hard pairs (trend_hard_ids) by construction",
                                   "reversal_pairs": pairs(trend, fwd, [flip[o] for o in fwd], report["trend_hard_ids"])}
    (OUT / "baselines.json").write_text(json.dumps(report, indent=1, default=float))

    print(f"{'baseline':18s} {'main':>6s} {'pooled':>7s}  per type (real)  |  pairs all / trend dec  |  real-shuf z_problem")
    for name in ("majority", "text_only", "order_blind", "order_aware"):
        b = report[name]
        p = b["reversal_pairs"]
        print(f"{name:18s} {b['real']['main']:6.3f} {b['real']['pooled']:7.3f}  "
              f"{ {t: round(v['accuracy'], 3) for t, v in b['real']['by_type'].items()} }  |  "
              f"{p['both_right']:.3f} / {p.get('trend_forward_decreasing', {}).get('both_right')}  |  "
              f"{b['real_vs_shuffled']['z_problem']:.1f} (only real {b['real_vs_shuffled']['only_a']}, "
              f"only shuf {b['real_vs_shuffled']['only_b']})")
    print("peak_side pilot rule N<205:", round(report["text_only"]["peak_side_pilot_rule_N_lt_205"]["main"], 3))
    print("trend hard pairs (order_blind forward answer wrong):", len(report["trend_hard_ids"]))
    for name in ("trend_arrow_blind",):
        a = report[name]["reversal_pairs"]
        print(name, "pairs: all", round(a["both_right"], 3), "fwd-decreasing", round(a["trend_forward_decreasing"]["both_right"], 3),
              "fwd-increasing", round(a["trend_forward_increasing"]["both_right"], 3))
    for name in ("order_blind", "order_aware"):
        print(name, "hard pairs", report[name]["reversal_pairs"]["hard"]["both_right"], "pairs by type", {t: round(v["both_right"], 3) for t, v in report[name]["reversal_pairs"]["by_type"].items()},
              "| real-vs-shuffled by type", {t: (v["only_a"], v["only_b"]) for t, v in report[name]["real_vs_shuffled"]["by_type"].items()})


if __name__ == "__main__":
    main()
