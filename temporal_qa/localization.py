"""Localization question templates: what happens where / when in a GPU counter trace.

Each template has make(i, rng) -> list[dict]. The answer is recomputed with the same rule
on the reversed trace (answer_reversed) and on a Bin-shuffled trace (answer_shuffled).
Per-Bin state: -2 = idle (no kernel), -1 = kernel without label, 0..11 = L1 class index.
"""
import numpy as np

from .paths import LABEL_ROOT, NPZ as DATA

N = 512
IDLE, UNLAB = -2, -1
INTRO = "The trace has 512 Bins in time order: Bin 0 is the start of the trace and Bin 511 is the end."

_d = np.load(DATA)
LABELS = [str(s) for s in _d['labels']]
PATHS = [str(p) for p in _d['paths']]
SPLIT = [str(s) for s in _d['split']]
Y = [str(s) for s in _d['y']]
_cache = {}


def states(i):
    """Per-Bin state array for Snapshot i (numeric keys only, no pickle)."""
    if i not in _cache:
        l = np.load(LABEL_ROOT + PATHS[i].replace('input/tensor_input.npz', 'labels/labels.npz'))
        m = l['mask_any_kernel'].astype(bool)
        _cache[i] = np.where(m, l['workload_l1'], IDLE).astype(int)
    return _cache[i]


def name(s):
    return 'idle' if s == IDLE else LABELS[s]


def views(s, rng):
    """Original, reversed and shuffled state sequences (shuffled trace Bin j holds original Bin perm[j])."""
    return s, s[::-1], s[rng.permutation(N)]


def rng_text(a, w):
    return f"Bins {a}-{a + w - 1}"


def pure_starts(s, v, w):
    ok = (s == v).astype(int)
    c = np.convolve(ok, np.ones(w, int), 'valid')
    return np.flatnonzero(c == w)


def busy_span(s):
    f = np.flatnonzero(s != IDLE)
    return f[0], f[-1]


def choose(starts, w, rng, near=None):
    """Random start. With near=c, draw among the (up to) 3 starts whose centre Bin is closest to c.
    Matching the busy range to the position of the idle range removes the centred-block position prior
    (idle near the edges, busy in the middle)."""
    starts = np.asarray(starts)
    if near is not None and len(starts) > 1:
        dist = np.abs(starts + (w - 1) / 2 - near)
        starts = starts[np.argsort(dist, kind='stable')[:3]]
    return int(rng.choice(starts))


def centre(a, w):
    return a + (w - 1) / 2


def pick_window(s, v, rng, widths=(16, 8, 4), interior=False, near=None):
    """Random pure window of state v; prefer the largest width; for idle prefer windows inside the busy span."""
    lo, hi = busy_span(s)
    if interior:
        for w in widths:
            a = pure_starts(s, v, w)
            a_in = a[(a > lo) & (a + w - 1 < hi)]
            if len(a_in):
                return choose(a_in, w, rng, near), w, True
    for w in widths:
        a = pure_starts(s, v, w)
        if len(a):
            return choose(a, w, rng, near), w, False
    return None


def has_gap(s, w):
    """True if an idle run of >= w Bins lies strictly inside the busy span (a gap the centred-block prior cannot see)."""
    lo, hi = busy_span(s)
    a = pure_starts(s, IDLE, w)
    return bool(np.any((a > lo) & (a + w - 1 < hi)))


def present(s):
    return [int(v) for v in np.unique(s) if v >= 0]


def distractors(exclude, k, rng):
    pool = [c for c in range(len(LABELS)) if c not in exclude]
    return [int(c) for c in rng.choice(pool, size=k, replace=False)]


def majority(seq):
    vals, cnt = np.unique(seq, return_counts=True)
    top = np.flatnonzero(cnt == cnt.max())
    return int(vals[top[0]]) if len(top) == 1 else None


def unique_true(flags):
    idx = [k for k, f in enumerate(flags) if f]
    return idx[0] if len(idx) == 1 else None


def unique_argmax(vals):
    vals = np.asarray(vals, float)
    top = np.flatnonzero(vals == vals.max())
    return int(top[0]) if len(top) == 1 else None


LETTERS = 'ABCD'


# ---------------------------------------------------------------- T1 window_state
def window_state(i, rng):
    """What is the GPU doing in Bins a-b? One question per state present (idle + each class)."""
    s = states(i)
    out = []
    # one width for all questions of a Snapshot, so the width does not leak the answer
    gap = has_gap(s, 4)
    lo, hi = busy_span(s)
    def idle_ok(w):
        a = pure_starts(s, IDLE, w)
        return np.any((a > lo) & (a + w - 1 < hi)) if gap else len(a) > 0
    w_ok = [w for w in (16, 8, 4) if idle_ok(w) and any(len(pure_starts(s, c, w)) for c in present(s))]
    if not w_ok:
        return []
    d_idle = None
    for v in [IDLE] + present(s):
        # busy ranges are matched to the idle range on distance from the trace centre
        win = pick_window(s, v, rng, (w_ok[0],), interior=(v == IDLE), near=d_idle)
        if win is None:
            continue
        a, w, inner = win
        if v == IDLE:
            d_idle = centre(a, w)
        cls = present(s)[:3]
        opts = [IDLE] + cls + distractors(cls, 3 - len(cls), rng)
        rng.shuffle(opts)
        texts = [name(o) for o in opts]

        def ans(seq):
            m = majority(seq[a:a + w])
            return name(m) if m in opts else None
        o, r, sh = views(s, rng)
        out.append(dict(template='window_state',
                        question=f"{INTRO} What is the GPU doing in {rng_text(a, w)}? "
                                 f"Answer with one of: {', '.join(texts)}.",
                        options=texts, answer=ans(o), answer_reversed=ans(r), answer_shuffled=ans(sh),
                        meta=dict(a=a, w=w, interior=inner, state=v, hard=has_gap(s, 4))))
    return out


# ---------------------------------------------------------------- T2 bin_state
def bin_state(i, rng):
    """At Bin b, is the GPU running X, Y, or idle? One idle Bin (inside the busy span if possible) and one busy Bin."""
    s = states(i)
    lo, hi = busy_span(s)
    cls = present(s)
    out = []
    d_idle = None
    for v in [IDLE, 'busy']:
        if v == IDLE:
            cand = np.flatnonzero(s == IDLE)
            inner = cand[(cand > lo) & (cand < hi)]
            cand = inner if len(inner) else cand
        else:
            cand = np.flatnonzero(s >= 0)
        if not len(cand):
            continue
        b = choose(cand, 1, rng, d_idle)
        if v == IDLE:
            d_idle = centre(b, 1)
        x = int(s[b]) if s[b] >= 0 else int(rng.choice(cls))
        other = [c for c in cls if c != x]
        y = int(rng.choice(other)) if other else distractors([x], 1, rng)[0]
        opts = [x, y, IDLE]
        rng.shuffle(opts)
        texts = [name(o) for o in opts]

        def ans(seq):
            return name(int(seq[b])) if int(seq[b]) in opts else None
        o, r, sh = views(s, rng)
        out.append(dict(template='bin_state',
                        question=f"{INTRO} At Bin {b}, is the GPU running {name(x)}, running {name(y)}, or idle? "
                                 f"Answer with one of: {', '.join(texts)}.",
                        options=texts, answer=ans(o), answer_reversed=ans(r), answer_shuffled=ans(sh),
                        meta=dict(b=b, interior=bool(lo < b < hi), state=int(s[b]), hard=has_gap(s, 1))))
    return out


# ---------------------------------------------------------------- T3 which_range_idle / busy
def which_range(i, rng):
    """Which of these Bin ranges is idle (or busy)? One odd range out of four, same width."""
    s = states(i)
    out = []
    for target in ['idle', 'busy']:
        for w in (16, 8, 4):
            idle_a = pure_starts(s, IDLE, w)
            busy_a = pure_starts((s != IDLE).astype(int), 1, w)
            lo, hi = busy_span(s)
            idle_in = idle_a[(idle_a > lo) & (idle_a + w - 1 < hi)]
            one, three = (idle_a, busy_a) if target == 'idle' else (busy_a, idle_a)
            if target == 'idle' and len(idle_in):
                one = idle_in
            if target == 'busy' and len(idle_in) >= 3:
                three = idle_in
            picked = []
            if len(one) and len(three):
                picked = [int(rng.choice(one))]
                for a in rng.permutation(three):
                    if all(abs(int(a) - p) >= w for p in picked):
                        picked.append(int(a))
                    if len(picked) == 4:
                        break
            if len(picked) == 4:
                break
        if len(picked) < 4:
            continue
        order = rng.permutation(4)
        starts = [picked[k] for k in order]
        texts = [f"{LETTERS[k]}: {rng_text(a, w)}" for k, a in enumerate(starts)]

        def ans(seq):
            frac = [np.mean((seq[a:a + w] == IDLE) if target == 'idle' else (seq[a:a + w] != IDLE)) for a in starts]
            k = unique_argmax(frac)
            return None if k is None else LETTERS[k]
        o, r, sh = views(s, rng)
        verb = 'idle (no kernel running in any of its Bins)' if target == 'idle' else 'busy (a kernel runs in every Bin)'
        out.append(dict(template='which_range',
                        question=f"{INTRO} Which of these Bin ranges is {verb}? {' '.join(texts)}. Answer A, B, C or D.",
                        options=list(LETTERS), answer=ans(o), answer_reversed=ans(r), answer_shuffled=ans(sh),
                        meta=dict(starts=starts, w=w, target=target, interior=bool(target == 'idle' and len(idle_in)), hard=bool(len(idle_in)))))
    return out


# ---------------------------------------------------------------- T4 first_start
def first_start(i, rng):
    """In which eighth of the trace does the first kernel (or class X) start?"""
    s = states(i)
    eighths = [f"Bins {k * 64}-{k * 64 + 63}" for k in range(8)]
    targets = [('kernel', None)] + ([(name(c), c) for c in present(s)] if len(present(s)) > 1 else [])
    out = []
    for txt, c in targets:
        def ans(seq):
            hit = np.flatnonzero(seq != IDLE) if c is None else np.flatnonzero(seq == c)
            return eighths[hit[0] // 64] if len(hit) else None
        o, r, sh = views(s, rng)
        what = 'the first kernel' if c is None else f'the first {txt} kernel'
        out.append(dict(template='first_start',
                        question=f"{INTRO} Each option is one eighth of the trace. In which eighth does {what} start? "
                                 f"Answer with one of: {', '.join(eighths)}.",
                        options=eighths, answer=ans(o), answer_reversed=ans(r), answer_shuffled=ans(sh),
                        meta=dict(cls=c, hard=c is not None)))
    return out


# ---------------------------------------------------------------- T5 kernel_edge (idle -> busy / busy -> idle)
def _edges(seq):
    busy = (seq != IDLE).astype(int)
    d = np.diff(busy)
    return np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1  # Bin where the new state begins


def _edge_windows(seq, w):
    """Per window start a: (#starts, #ends) whose new-state Bin t satisfies a < t <= a+w-1 (both sides inside)."""
    st, en = _edges(seq)
    def count(ev, a):
        return int(np.sum((ev > a) & (ev <= a + w - 1)))
    return count, st, en


def kernel_edge(i, rng):
    """Which Bin range contains a kernel start (idle -> busy)? Or a kernel end (busy -> idle)?
    Distractors: a range with the opposite edge, one pure idle and one pure busy range."""
    s = states(i)
    out = []
    w = 8
    count, st, en = _edge_windows(s, w)
    allA = np.arange(N - w + 1)
    c_st = np.array([count(st, a) for a in allA])
    c_en = np.array([count(en, a) for a in allA])
    only_st = allA[(c_st > 0) & (c_en == 0)]
    only_en = allA[(c_en > 0) & (c_st == 0)]
    pure_idle = pure_starts(s, IDLE, w)
    pure_busy = pure_starts((s != IDLE).astype(int), 1, w)
    for target in ['start', 'end']:
        one, opp = (only_st, only_en) if target == 'start' else (only_en, only_st)
        if not len(one) or not len(opp):
            continue
        # prefer the order-adversarial pair: the asked edge lies to the RIGHT (start) / LEFT (end) of the opposite
        # edge, so "start = left mixed range" fails; needs an idle gap or loop inside the busy span
        sign = 1 if target == 'start' else -1
        diff = one[:, None] - opp[None, :]
        ok = np.abs(diff) >= w
        if not ok.any():
            continue
        adv = bool((ok & (sign * diff > 0)).any())
        r1, r2 = np.nonzero(ok & (sign * diff > 0)) if adv else np.nonzero(ok)
        k = int(rng.integers(len(r1)))
        picked = [int(one[r1[k]]), int(opp[r2[k]])]
        for pool in (pure_idle, pure_busy):
            for a in rng.permutation(pool) if len(pool) else []:
                if all(abs(int(a) - p) >= w for p in picked):
                    picked.append(int(a))
                    break
        if len(picked) < 4:
            continue
        order = rng.permutation(4)
        starts = [picked[k] for k in order]
        texts = [f"{LETTERS[k]}: {rng_text(a, w)}" for k, a in enumerate(starts)]

        def ans(seq):
            cnt, st2, en2 = _edge_windows(seq, w)
            a_ev, b_ev = (st2, en2) if target == 'start' else (en2, st2)
            k = unique_true([cnt(a_ev, a) > 0 and cnt(b_ev, a) == 0 for a in starts])
            return None if k is None else LETTERS[k]
        o, r, sh = views(s, rng)
        what = ('a kernel start (the GPU goes from idle to busy)' if target == 'start'
                else 'a kernel end (the GPU goes from busy to idle)')
        out.append(dict(template='kernel_edge',
                        question=f"{INTRO} Which of these Bin ranges contains {what}? {' '.join(texts)}. Answer A, B, C or D.",
                        options=list(LETTERS), answer=ans(o), answer_reversed=ans(r), answer_shuffled=ans(sh),
                        meta=dict(starts=starts, w=w, target=target, hard=adv)))
    return out


# ---------------------------------------------------------------- T6 class_switch (multi-workload only)
def class_switch(i, rng):
    """Where does the workload switch from X to Y (Bin t-1 is X, Bin t is Y)?
    Distractors: a range with the reverse switch Y -> X if any, else ranges with X and Y but no X -> Y."""
    s = states(i)
    cls = present(s)
    if len(cls) < 2:
        return []
    w = 8
    pairs = {}
    for t in range(1, N):
        if s[t - 1] >= 0 and s[t] >= 0 and s[t - 1] != s[t]:
            pairs.setdefault((int(s[t - 1]), int(s[t])), []).append(t)
    if not pairs:
        return []
    keys = sorted(pairs)
    out = []
    for kk in rng.permutation(len(keys))[:2]:
        x, y = keys[kk]

        def has(seq, a, p, q):
            seg = seq[a:a + w]
            return bool(np.any((seg[:-1] == p) & (seg[1:] == q)))
        allA = np.arange(N - w + 1)
        good = [a for a in allA if has(s, a, x, y)]
        rev = [a for a in allA if has(s, a, y, x) and not has(s, a, x, y)]
        both = [a for a in allA if x in s[a:a + w] and y in s[a:a + w] and not has(s, a, x, y) and not has(s, a, y, x)]
        other = [a for a in allA if not has(s, a, x, y) and not has(s, a, y, x)]
        picked = [int(rng.choice(good))]
        # one reverse-switch range (same multiset, opposite order), then ranges that hold X and Y without the switch
        for pool in (rev, both, both, other, other, other):
            for a in rng.permutation(pool) if len(pool) else []:
                if all(abs(int(a) - p) >= w for p in picked):
                    picked.append(int(a))
                    break
            if len(picked) == 4:
                break
        if len(picked) < 4:
            continue
        order = rng.permutation(4)
        starts = [picked[k] for k in order]
        texts = [f"{LETTERS[k]}: {rng_text(a, w)}" for k, a in enumerate(starts)]

        def ans(seq):
            k = unique_true([has(seq, a, x, y) for a in starts])
            return None if k is None else LETTERS[k]
        o, r, sh = views(s, rng)
        out.append(dict(template='class_switch',
                        question=f"{INTRO} Which of these Bin ranges contains a switch from {name(x)} directly to "
                                 f"{name(y)} (one Bin {name(x)}, the next Bin {name(y)})? {' '.join(texts)}. "
                                 f"Answer A, B, C or D.",
                        options=list(LETTERS), answer=ans(o), answer_reversed=ans(r), answer_shuffled=ans(sh),
                        meta=dict(starts=starts, w=w, x=x, y=y, hard=True)))
    return out


TEMPLATES = [window_state, bin_state, which_range, first_start, kernel_edge, class_switch]
