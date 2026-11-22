"""Build the temporal question set: questions about a Snapshot whose answer depends on time order.

The templates come from four families in temporal_qa/ (sequence, timing, dynamics, localization). Each was
scored for how often its answer changes when the trace is reversed or shuffled, and for how well a model does
without time order; only the strong templates are used here. Each template is drawn DRAWS times per Snapshot
with a different random seed, and identical questions are kept once.

Every row: path, split, template, family, question, options ([] = free answer, a number), answer, and the
answer the same question would have with the trace reversed / its Bins shuffled (a control, never shown to the
model). Writes runs/temporal_qa/{train,validation,test}.jsonl and summary.json.

  uv run --no-project --with numpy python experiments/temporal_order/temporal_questions.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import sys

import numpy as np

HERE = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(HERE))
from temporal_qa import dynamics, localization, sequence, timing  # noqa: E402

DATA = np.load(HERE / "kernelsight.npz")
HARD_ONLY = {"which_range", "bin_state"}  # their full sets have a strong position shortcut; see the hard subset

# (family, template name, function, draws per Snapshot)
TEMPLATES = [
    ("dynamics", "peak_order", dynamics.make_peak_order, 4),
    ("dynamics", "onset", dynamics.make_onset, 3),
    ("localization", "kernel_edge", localization.kernel_edge, 3),
    ("localization", "which_range", localization.which_range, 3),
    ("localization", "bin_state", localization.bin_state, 3),
    ("localization", "class_switch", localization.class_switch, 3),
    ("timing", "iteration_at_bin", timing.iteration_at_bin, 3),
    ("timing", "longest_gap_len", timing.longest_gap_len, 1),
    ("timing", "loop_period", timing.loop_period, 1),
    ("sequence", "loop_order", sequence.make_loop_order, 1),
    ("sequence", "next_incl_idle", sequence.make_next_incl_idle, 1),
    ("sequence", "next_op", sequence.make_next_op, 1),
    ("sequence", "prev_op", sequence.make_prev_op, 1),
]


def build():
    rows, seen = [], set()
    for i, (path, split) in enumerate(zip(DATA["paths"].tolist(), DATA["split"].tolist())):
        for family, name, make, draws in TEMPLATES:
            for draw in range(draws):
                rng = np.random.default_rng([i, draw, sum(map(ord, name))])
                for q in make(i, rng):
                    if name in HARD_ONLY and not q.get("meta", {}).get("hard"):
                        continue
                    options = [str(o) for o in (q["options"] or [])]
                    key = (i, q["question"], tuple(options))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append({"path": path, "split": split, "family": family, "template": name,
                                 "question": q["question"], "options": options, "answer": str(q["answer"]),
                                 "answer_reversed": None if q["answer_reversed"] is None else str(q["answer_reversed"]),
                                 "answer_shuffled": None if q["answer_shuffled"] is None else str(q["answer_shuffled"])})
    return rows


def summarize(rows):
    out = defaultdict(dict)
    for name in dict.fromkeys(r["template"] for r in rows):
        part = [r for r in rows if r["template"] == name]
        answers = Counter(r["answer"] for r in part)
        out[name] = {"family": part[0]["family"],
                     "questions": dict(Counter(r["split"] for r in part)),
                     "snapshots": len({r["path"] for r in part}),
                     "top_answer_share": round(answers.most_common(1)[0][1] / len(part), 3),
                     "changed_if_reversed": round(np.mean([r["answer_reversed"] != r["answer"] for r in part]), 3),
                     "changed_if_shuffled": round(np.mean([r["answer_shuffled"] != r["answer"] for r in part]), 3)}
    return dict(out)


def main():
    out = HERE / "runs/temporal_qa"
    out.mkdir(parents=True, exist_ok=True)
    rows = build()
    for split in ("train", "validation", "test"):
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["split"] == split))
    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{'template':<18}{'train':>7}{'val':>6}{'test':>6}{'snaps':>7}{'top ans':>9}{'rev chg':>9}{'shuf chg':>10}")
    for name, s in summary.items():
        q = s["questions"]
        print(f"{name:<18}{q.get('train', 0):>7}{q.get('validation', 0):>6}{q.get('test', 0):>6}{s['snapshots']:>7}"
              f"{s['top_answer_share']:>9.2f}{s['changed_if_reversed']:>9.2f}{s['changed_if_shuffled']:>10.2f}")
    total = Counter(r["split"] for r in rows)
    print(f"{'TOTAL':<18}{total['train']:>7}{total['validation']:>6}{total['test']:>6}")


if __name__ == "__main__":
    main()
