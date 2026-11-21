"""Timing family: idle gaps, launches, loops. Each template: make(i, rng) -> list[dict].

All answers come from mask_any_kernel (and workload_l1 for multi-workload Snapshots).
Every trace has symmetric idle padding (idle before the first kernel == idle after the last kernel),
so positions are given relative to the ACTIVE SPAN (first busy Bin to last busy Bin), not the whole trace.
"""
from functools import lru_cache
import numpy as np

ROOT = '/Users/m7m2od/TimeNet-kernelsight/kernelsight/'
LABELS = ['matmul', 'conv', 'activation', 'normalization', 'softmax', 'pooling', 'reduction',
          'attention', 'loss', 'elementwise', 'memory_movement', 'other']
INTRO = ("The trace has 512 Bins in time order; Bin 0 is the start of the trace and Bin 511 the end. "
         "A Bin is busy if a GPU kernel runs in it and idle if not. A busy period is a run of consecutive busy Bins; "
         "an idle gap is a run of idle Bins between two busy periods. ")


@lru_cache(maxsize=None)
def _paths():
    return np.load('/Users/m7m2od/hackathon/kernelsight.npz')['paths']


@lru_cache(maxsize=4096)
def load(i):
    L = np.load(ROOT + str(_paths()[i]).replace('input/tensor_input.npz', 'labels/labels.npz'))
    return L['mask_any_kernel'].astype(np.int8), L['workload_l1'].astype(int)


def runs(m, v=1):
    """(start, end_exclusive) of runs where m == v."""
    d = np.diff(np.r_[0, (m == v).astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def gaps(m):
    """Idle gaps strictly between busy periods."""
    b = runs(m)
    return [(b[k][1], b[k + 1][0]) for k in range(len(b) - 1)]


# ---------- answer functions: (mask, l1) -> answer str or None ----------

def a_n_busy(m, l1):
    return str(len(runs(m)))


def a_longest_gap_pos(m, l1):
    g = gaps(m)
    if not g:
        return None
    L = np.array([e - s for s, e in g])
    if (L == L.max()).sum() > 1:
        return None
    s, e = g[int(L.argmax())]
    b = runs(m)
    lo, hi = b[0][0], b[-1][1]
    c = ((s + e) / 2 - lo) / (hi - lo)
    return 'early (first third)' if c < 1 / 3 else ('late (last third)' if c > 2 / 3 else 'middle (second third)')


def a_first_vs_last(m, l1):
    b = runs(m)
    if len(b) < 2:
        return None
    f, l = b[0][1] - b[0][0], b[-1][1] - b[-1][0]
    if abs(f - l) <= max(1, 0.1 * max(f, l)):
        return 'about the same'
    return 'first is longer' if f > l else 'last is longer'


def a_busier_half(m, l1):
    b = runs(m)
    if len(b) < 2:
        return None
    lo, hi = b[0][0], b[-1][1]
    mid = (lo + hi) // 2
    h1, h2 = m[lo:mid].sum(), m[mid:hi].sum()
    if h1 == h2:
        return None
    return 'first half' if h1 > h2 else 'second half'


def a_period(m, l1):
    b = runs(m)
    if len(b) < 4:
        return None
    return str(int(round(np.median(np.diff([s for s, _ in b])))))


def a_longest_gap_len(m, l1):
    g = gaps(m)
    if not g:
        return None
    return str(max(e - s for s, e in g))


def a_first_after_idle(m, l1):
    """Most common L1 class of the first labeled busy Bin of each busy period."""
    present = sorted(set(l1[l1 >= 0]))
    if len(present) < 2:
        return None
    firsts = []
    for s, e in runs(m):
        lab = l1[s:e][l1[s:e] >= 0]
        if len(lab):
            firsts.append(lab[0])
    if not firsts:
        return None
    v, c = np.unique(firsts, return_counts=True)
    if (c == c.max()).sum() > 1:
        return None
    return LABELS[v[c.argmax()]]


def a_first_vs_last_gap(m, l1):
    g = gaps(m)
    if len(g) < 2:
        return None
    f, l = g[0][1] - g[0][0], g[-1][1] - g[-1][0]
    if f == l:
        return 'same length'
    return 'first gap is longer' if f > l else 'last gap is longer'


def a_iteration_at(m, l1, k):
    """Number (from 1) of the busy period that contains Bin k, or 'idle'."""
    if m[k] == 0:
        return 'idle'
    return str(sum(1 for s, _ in runs(m) if s <= k))


def a_busy_share(m, l1):  # order-free control
    p = m.mean()
    return '<25%' if p < .25 else ('25-50%' if p < .5 else ('50-75%' if p < .75 else '>75%'))


# ---------- templates ----------

T = {
    'n_busy_periods': (a_n_busy, None,
        "How many separate busy periods (kernel launches or loop iterations) are in the trace? Answer with a number."),
    'longest_gap_position': (a_longest_gap_pos, ['early (first third)', 'middle (second third)', 'late (last third)'],
        "The active span runs from the first busy Bin to the last busy Bin. Where in the active span is the longest idle gap between kernels?"),
    'first_vs_last_period': (a_first_vs_last, ['first is longer', 'last is longer', 'about the same'],
        "Compare the length in Bins of the first busy period and the last busy period (about the same = within 10% or 1 Bin). Which is longer?"),
    'busier_half': (a_busier_half, ['first half', 'second half'],
        "Split the active span (first busy Bin to last busy Bin) into two halves. In which half is the GPU busy in more Bins?"),
    'loop_period': (a_period, None,
        "The kernel runs in a loop. How many Bins pass from the start of one busy period to the start of the next (median over the trace)? Answer with a number."),
    'longest_gap_len': (a_longest_gap_len, None,
        "How many Bins long is the longest idle gap between two busy periods? Answer with a number."),
    'first_after_idle': (a_first_after_idle, None,
        "Several workloads run in a loop with idle gaps. Which workload usually runs first in a busy period, right after the GPU was idle?"),
    'first_vs_last_gap': (a_first_vs_last_gap, ['first gap is longer', 'last gap is longer', 'same length'],
        "Compare the first idle gap between kernels with the last idle gap between kernels. Which is longer?"),
    'iteration_at_bin': (a_iteration_at, None,
        "Bin {k} is busy. Counting busy periods from 1 at the start of the trace, which busy period contains Bin {k}? Answer with a number."),
    'busy_share_control': (a_busy_share, ['<25%', '25-50%', '50-75%', '>75%'],
        "What share of all 512 Bins is busy?"),
}


def _make(name, i, rng):
    f, opts, q = T[name]
    m, l1 = load(i)
    perm = rng.permutation(512)
    if name == 'iteration_at_bin':
        if len(runs(m)) < 2:
            return []
        k = int(rng.choice(np.where(m == 1)[0]))
        g = lambda mm, ll: a_iteration_at(mm, ll, k)
        return [dict(template=name, question=INTRO + q.format(k=k), options=None, answer=g(m, l1),
                     answer_reversed=g(m[::-1].copy(), l1[::-1].copy()), answer_shuffled=g(m[perm], l1[perm]), bin=k)]
    ans = f(m, l1)
    if ans is None:
        return []
    if opts is None and name == 'first_after_idle':
        opts = [LABELS[k] for k in sorted(set(l1[l1 >= 0]))]
    return [dict(template=name, question=INTRO + q, options=opts, answer=ans,
                 answer_reversed=f(m[::-1].copy(), l1[::-1].copy()),
                 answer_shuffled=f(m[perm], l1[perm]))]


def _bind(name):
    def make(i, rng):
        return _make(name, i, rng)
    make.__name__ = name
    return make


for _n in T:
    globals()[_n] = _bind(_n)
