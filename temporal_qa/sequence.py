"""Sequence / order-of-operations question templates for KernelSight traces.

Each template: make(i, rng) -> list[dict] with keys
template, question, options, answer, answer_reversed, answer_shuffled.
The reversed / shuffled answers apply the SAME question text to the trace with its Bins
reversed / randomly permuted (permutation drawn from rng, so deterministic given rng).
"""
from collections import Counter
from itertools import permutations

import numpy as np

from .paths import LABEL_ROOT, NPZ

D = np.load(NPZ)
CLASSES = [str(c) for c in D["labels"]]
N_BINS = 512
INTRO = ("The trace has 512 time Bins of GPU counters; Bin 0 is the start of the trace and Bin 511 is the end. ")
IDLE = -2  # bin state: GPU idle (no kernel)
UNL = -1   # bin state: kernel runs but has no label

_cache = {}


def bins(i):
    """Per-Bin state: class index, UNL, or IDLE."""
    if i not in _cache:
        p = LABEL_ROOT + str(D["paths"][i]).replace("input/tensor_input.npz", "labels/labels.npz")
        L = np.load(p)
        w = L["workload_l1"].astype(int).copy()
        w[L["mask_any_kernel"] == 0] = IDLE
        _cache[i] = w
    return _cache[i]


def variants(i, rng):
    w = bins(i)
    return w, w[::-1], w[rng.permutation(N_BINS)]


def op_runs(w):
    """Sequence of labeled operations in time order, idle/unlabeled Bins dropped, repeats collapsed."""
    s = [int(c) for c in w if c >= 0]
    return [c for k, c in enumerate(s) if k == 0 or s[k - 1] != c]


def first_order(w):
    seen = []
    for c in op_runs(w):
        if c not in seen:
            seen.append(c)
    return seen


def neighbor(w, x, step, min_share=0.6):
    """Most common operation right after (step=+1) or before (step=-1) operation x. None if unclear."""
    r = op_runs(w)
    nb = [r[k + step] for k in range(len(r)) if r[k] == x and 0 <= k + step < len(r)]
    if not nb:
        return None
    c, n = Counter(nb).most_common(1)[0]
    return CLASSES[c] if n / len(nb) >= min_share else None


def item(t, q, opts, a, ar, ash):
    return dict(template=t, question=INTRO + q, options=opts, answer=a,
                answer_reversed=ar, answer_shuffled=ash)


def present(w):
    return first_order(w)


# ---------------- templates on multi-workload Snapshots ----------------

def make_next_op(i, rng):
    w, wr, ws = variants(i, rng)
    ops = present(w)
    out = []
    if len(ops) < 3:  # with 2 operations there is only one option
        return out
    for x in ops:
        a = neighbor(w, x, +1)
        if a is None:
            continue
        opts = [CLASSES[c] for c in ops if c != x]
        q = (f"Ignore idle gaps. Which operation usually runs right after the {CLASSES[x]} operation? "
             f"Options: {', '.join(opts)}.")
        out.append(item("next_op", q, opts, a, neighbor(wr, x, +1), neighbor(ws, x, +1)))
    return out


def make_prev_op(i, rng):
    w, wr, ws = variants(i, rng)
    ops = present(w)
    out = []
    if len(ops) < 3:  # with 2 operations there is only one option
        return out
    for x in ops:
        a = neighbor(w, x, -1)
        if a is None:
            continue
        opts = [CLASSES[c] for c in ops if c != x]
        q = (f"Ignore idle gaps. Which operation usually runs right before the {CLASSES[x]} operation? "
             f"Options: {', '.join(opts)}.")
        out.append(item("prev_op", q, opts, a, neighbor(wr, x, -1), neighbor(ws, x, -1)))
    return out


def _first(w):
    o = first_order(w)
    return CLASSES[o[0]] if o else None


def _last(w):
    o = first_order(w[::-1])
    return CLASSES[o[0]] if o else None


def make_first_op(i, rng):
    w, wr, ws = variants(i, rng)
    ops = present(w)
    if len(ops) < 2:
        return []
    opts = [CLASSES[c] for c in sorted(ops)]
    q = f"Which operation runs first in the trace? Options: {', '.join(opts)}."
    return [item("first_op", q, opts, _first(w), _first(wr), _first(ws))]


def make_last_op(i, rng):
    w, wr, ws = variants(i, rng)
    ops = present(w)
    if len(ops) < 2:
        return []
    opts = [CLASSES[c] for c in sorted(ops)]
    q = f"Which operation runs last in the trace? Options: {', '.join(opts)}."
    return [item("last_op", q, opts, _last(w), _last(wr), _last(ws))]


def _which_first(w, x, y):
    o = first_order(w)
    return CLASSES[x] if o.index(x) < o.index(y) else CLASSES[y]


def make_x_before_y(i, rng):
    w, wr, ws = variants(i, rng)
    ops = present(w)
    out = []
    for a_ in range(len(ops)):
        for b_ in range(a_ + 1, len(ops)):
            x, y = (ops[a_], ops[b_]) if rng.random() < 0.5 else (ops[b_], ops[a_])
            opts = [CLASSES[x], CLASSES[y]]
            q = f"Which starts first in the trace: {CLASSES[x]} or {CLASSES[y]}?"
            out.append(item("x_before_y", q, opts, _which_first(w, x, y), _which_first(wr, x, y),
                            _which_first(ws, x, y)))
    return out


def _order_str(o):
    return " -> ".join(CLASSES[c] for c in o)


def make_loop_order(i, rng):
    w, wr, ws = variants(i, rng)
    ops = present(w)
    if len(ops) < 2:
        return []
    true, rev, shf = (_order_str(first_order(v)) for v in (w, wr, ws))
    perms = [p for p in permutations(ops)]
    others = [_order_str(p) for p in perms if _order_str(p) not in (true, rev)]
    pick = [others[k] for k in rng.permutation(len(others))[:2]]
    opts = sorted(set([true, rev] + pick))
    opts = [opts[k] for k in rng.permutation(len(opts))]
    q = ("In what order do the operations first appear (the order of operations in one loop iteration)? "
         "Options: " + " | ".join(opts) + ".")
    return [item("loop_order", q, opts, true, rev if rev in opts else None, shf if shf in opts else None)]


# ---------------- templates on all Snapshots ----------------

NEXT_OPTS = ["the GPU goes idle", "a different operation starts", "the trace ends"]


def _after_bin(w, b):
    """What follows the operation running at Bin b. None if Bin b has no labeled kernel."""
    c = w[b]
    if c < 0:
        return None
    j = b
    while j < N_BINS and w[j] == c:
        j += 1
    while j < N_BINS and w[j] == UNL:  # unlabeled tail of the same kernel
        j += 1
    if j == N_BINS:
        return NEXT_OPTS[2]
    return NEXT_OPTS[0] if w[j] == IDLE else NEXT_OPTS[1]


def _pick_bin(w, fn, rng):
    """Pick the answer uniformly among those this Snapshot offers, then a Bin with that answer
    (keeps the rare answers from vanishing)."""
    by = {}
    for b in np.where(w >= 0)[0]:
        by.setdefault(fn(w, int(b)), []).append(int(b))
    if not by:
        return None
    keys = sorted(by)
    bs = by[keys[rng.integers(len(keys))]]
    return bs[rng.integers(len(bs))]


def make_next_after_bin(i, rng):
    w, wr, ws = variants(i, rng)
    b = _pick_bin(w, _after_bin, rng)
    if b is None:
        return []
    q = (f"A kernel runs at Bin {b}. What happens right after this operation ends? "
         f"Options: {' | '.join(NEXT_OPTS)}.")
    return [item("next_after_bin", q, NEXT_OPTS, _after_bin(w, b), _after_bin(wr, b), _after_bin(ws, b))]


PREV_OPTS = ["the GPU was idle", "a different operation ran", "it is the start of the trace"]


def _before_bin(w, b):
    a = _after_bin(w[::-1], N_BINS - 1 - b)
    return None if a is None else PREV_OPTS[NEXT_OPTS.index(a)]


def make_prev_before_bin(i, rng):
    w, wr, ws = variants(i, rng)
    b = _pick_bin(w, _before_bin, rng)
    if b is None:
        return []
    q = (f"A kernel runs at Bin {b}. What happened right before this operation started? "
         f"Options: {' | '.join(PREV_OPTS)}.")
    return [item("prev_before_bin", q, PREV_OPTS, _before_bin(w, b), _before_bin(wr, b), _before_bin(ws, b))]


PHASE_OPTS = ["busy the whole time", "idle, then busy", "busy, then idle", "idle, then busy, then idle",
              "busy and idle alternate several times"]


def _phase(w, min_len=3):
    """Phase pattern of busy (kernel) vs idle Bins. Idle/busy runs shorter than min_len Bins are merged."""
    busy = (w != IDLE).astype(int)
    runs = []
    for v in busy:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    # merge short runs into neighbors
    merged = []
    for v, n in runs:
        if n < min_len and merged:
            merged[-1][1] += n
        elif merged and merged[-1][0] == v:
            merged[-1][1] += n
        else:
            merged.append([v, n])
    pat = [v for v, _ in merged]
    pat = [v for k, v in enumerate(pat) if k == 0 or pat[k - 1] != v]
    table = {(1,): 0, (0, 1): 1, (1, 0): 2, (0, 1, 0): 3}
    if tuple(pat) in table:
        return PHASE_OPTS[table[tuple(pat)]]
    if 1 not in pat:
        return None
    return PHASE_OPTS[4]


def make_phase_order(i, rng):
    w, wr, ws = variants(i, rng)
    a = _phase(w)
    if a is None:
        return []
    q = ("Ignore very short blips (under 3 Bins). In what order do busy phases (a kernel runs) and idle phases "
         f"(no kernel) happen? Options: {' | '.join(PHASE_OPTS)}.")
    return [item("phase_order", q, PHASE_OPTS, a, _phase(wr), _phase(ws))]


def _runs_with_idle(w, min_idle=1):
    """Time-ordered states: class index or IDLE, unlabeled kernel Bins dropped, repeats collapsed."""
    s = [int(c) for c in w if c != UNL]
    return [c for k, c in enumerate(s) if k == 0 or s[k - 1] != c]


def _next_incl_idle(w, x, min_share=0.6):
    r = _runs_with_idle(w)
    nb = [r[k + 1] for k in range(len(r) - 1) if r[k] == x]
    if not nb:
        return None
    c, n = Counter(nb).most_common(1)[0]
    if n / len(nb) < min_share:
        return None
    return "an idle gap" if c == IDLE else CLASSES[c]


def make_next_incl_idle(i, rng):
    w, wr, ws = variants(i, rng)
    ops = present(w)
    out = []
    if len(ops) < 2:
        return out
    for x in ops:
        a = _next_incl_idle(w, x)
        if a is None:
            continue
        opts = [CLASSES[c] for c in ops if c != x] + ["an idle gap"]
        q = (f"What usually comes right after the {CLASSES[x]} operation ends? "
             f"Options: {', '.join(opts)}.")
        out.append(item("next_incl_idle", q, opts, a, _next_incl_idle(wr, x), _next_incl_idle(ws, x)))
    return out


TEMPLATES = {
    "next_op": make_next_op,
    "prev_op": make_prev_op,
    "first_op": make_first_op,
    "last_op": make_last_op,
    "x_before_y": make_x_before_y,
    "loop_order": make_loop_order,
    "next_after_bin": make_next_after_bin,
    "prev_before_bin": make_prev_before_bin,
    "phase_order": make_phase_order,
    "next_incl_idle": make_next_incl_idle,
}
