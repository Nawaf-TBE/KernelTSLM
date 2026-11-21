"""Counter-dynamics question templates. Answers come from the counters x [24, 512] only.

Each template has an answer function ans_<name>(x, params) so that the same question can be
answered on the true, reversed and shuffled trace. make_<name>(i, rng) builds the questions.
"Busy" Bin = any of the 20 per-kernel counters (channels 0-19) is > 0. This equals the label
mask_any_kernel on all 1444 Snapshots.
"""
import numpy as np
from .dynamics_load import X, CH, Y

C = {n: k for k, n in enumerate(CH)}
MEMORY = ['DRAM Read Bandwidth [Throughput %]', 'DRAM Write Bandwidth [Throughput %]', 'hit: l2', 'pipe: lsu']
COMPUTE = ['pipe: alu', 'pipe: fma', 'pipe: tensor_op_hmma', 'pipe: xu', 'SMs Active [Throughput %]',
           'Tensor Active [Throughput %]']
# Counters that can change inside a run (SASS modality and atom are near-constant per Snapshot).
VARYING = [c for c in CH if not c.startswith('sass_') and not c.startswith('atom')]
INTRO = ("The trace has 512 time Bins; Bin 0 is the start of the trace and Bin 511 is the end. "
         "A Bin is busy when a kernel runs in it. ")


def busy(x):
    return (x[:20] > 0).any(0)


def bursts(x):
    """List of (start, end_exclusive) for runs of busy Bins."""
    d = np.diff(np.r_[0, busy(x).astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def reverse(x):
    return x[:, ::-1]


def shuffle(x, rng):
    return x[:, rng.permutation(x.shape[1])]


_PRIOR = {}
# Per template: drop counters (pairs) whose train answer share for (name, class) is above this.
# Measured: the filter helps the pair templates and hurts or empties the single-counter ones.
MAX_PRIOR = {'peak_order': 0.75, 'onset': 0.75}


def _prior(name, cand_fn):
    """P(answer | key, Snapshot class) over TRAIN Snapshots, for every candidate counter (pair).
    The class is in the key because the model can read the class from the trace easily."""
    if name not in _PRIOR:
        from collections import Counter, defaultdict
        from .dynamics_load import SPLIT, Y
        t = defaultdict(Counter)
        for i in np.where(SPLIT == 'train')[0]:
            for key, _, a in cand_fn(X[i]):
                if a is not None:
                    t[(key, Y[i])][a] += 1
        _PRIOR[name] = {k: {a: n / sum(c.values()) for a, n in c.items()} for k, c in t.items()}
    return _PRIOR[name]


def _pick(name, cand_fn, x, rng, i):
    """Pick one candidate (key, params, answer) with weight 1 / P_train(answer | key).
    This flattens the answer distribution inside each counter (pair), so a model that only
    reads WHICH counters are named cannot beat chance by much."""
    prior = _prior(name, cand_fn)
    # Drop counters (pairs) whose answer is predictable from name + class alone.
    cands = [c for c in cand_fn(x) if c[2] is not None
             and max(prior.get((c[0], Y[i]), {'': 0}).values()) <= MAX_PRIOR.get(name, 1.0)]
    if not cands:
        return None
    w = np.array([1.0 / prior.get((k, Y[i]), {}).get(a, 1.0) for k, _, a in cands])
    return cands[rng.choice(len(cands), p=w / w.sum())]


def _q(template, question, options, answer, rev, shf, **meta):
    return dict(template=template, question=question, options=options, answer=answer,
                answer_reversed=rev, answer_shuffled=shf, **meta)


# 1. trend (3-way) and 1b. direction (2-way) ------------------------------------------------
TREND_OPTS = ['rises', 'falls', 'stays about flat']


def ans_trend(x, p):
    v = x[C[p['counter']]][busy(x)]
    if len(v) < 8 or v.max() <= 0:
        return None
    q = len(v) // 4
    r = (np.median(v[-q:]) - np.median(v[:q])) / v.max()
    return 'rises' if r > 0.1 else 'falls' if r < -0.1 else 'stays about flat'


def cands_trend(x):
    return [(c, {'counter': c}, ans_trend(x, {'counter': c})) for c in VARYING]


def make_trend(i, rng):
    x = X[i]
    pick = _pick('trend', cands_trend, x, rng, i)
    if pick is None:
        return []
    key, p, a = pick
    q = (INTRO + "Look only at the busy Bins. From the first quarter of busy time to the last quarter, "
         f"does the counter '{key}' rise, fall, or stay about flat?")
    return [_q('trend', q, TREND_OPTS, a, ans_trend(reverse(x), p), ans_trend(shuffle(x, rng), p), key=key)]


def ans_direction(x, p):
    a = ans_trend(x, p)
    return a if a in ('rises', 'falls') else None


def cands_direction(x):
    return [(c, {'counter': c}, ans_direction(x, {'counter': c})) for c in VARYING]


def make_direction(i, rng):
    x = X[i]
    pick = _pick('direction', cands_direction, x, rng, i)
    if pick is None:
        return []
    key, p, a = pick
    q = (INTRO + f"Look only at the busy Bins. The counter '{key}' changes during the run. "
         "Is it higher in the last quarter of busy time than in the first quarter (rises), or lower (falls)?")
    return [_q('direction', q, ['rises', 'falls'], a, ans_direction(reverse(x), p),
               ans_direction(shuffle(x, rng), p), key=key)]


# 2. peak order and 5. onset order: one memory counter vs one compute counter -------------------
SAME = 'at about the same time'
PAIRS = [(m, c) for m in MEMORY for c in COMPUTE]


def _first_bin(v, mode):
    if v.max() <= 0:
        return None
    return int(np.argmax(v)) if mode == 'peak' else int(np.argmax(v > 0.5 * v.max()))


def _ans_order(x, p, mode, tol):
    ta, tb = _first_bin(x[C[p['a']]], mode), _first_bin(x[C[p['b']]], mode)
    if ta is None or tb is None:
        return None
    if abs(ta - tb) <= tol:
        return SAME
    return p['a'] if ta < tb else p['b']


def ans_peak_order(x, p):
    return _ans_order(x, p, 'peak', 3)


def ans_onset(x, p):
    return _ans_order(x, p, 'onset', 1)


def _cands_order(ans):
    def f(x):
        out = []
        for m, c in PAIRS:
            a = ans(x, {'a': m, 'b': c})
            # canonical answer per unordered pair: which GROUP wins, so the prior is order-free
            canon = None if a is None else 'same' if a == SAME else 'memory' if a == m else 'compute'
            out.append((m + '|' + c, {'a': m, 'b': c}, canon))
        return out
    return f


cands_peak_order = _cands_order(ans_peak_order)
cands_onset = _cands_order(ans_onset)


def _make_order(name, ans, cands, text, i, rng):
    x = X[i]
    pick = _pick(name, cands, x, rng, i)
    if pick is None:
        return []
    _, p, _ = pick
    if rng.random() < 0.5:
        p = {'a': p['b'], 'b': p['a']}
    q = INTRO + text.format(a=p['a'], b=p['b'])
    return [_q(name, q, [p['a'], p['b'], SAME], ans(x, p), ans(reverse(x), p), ans(shuffle(x, rng), p),
               key=p['a'] + '|' + p['b'])]


def make_peak_order(i, rng):
    return _make_order('peak_order', ans_peak_order, cands_peak_order,
                       "Which counter reaches its highest value first: '{a}' or '{b}'? "
                       f"If the two peaks are within 3 Bins, answer '{SAME}'.", i, rng)


def make_onset(i, rng):
    return _make_order('onset', ans_onset, cands_onset,
                       "Which counter first rises above half of its own maximum: '{a}' or '{b}'? "
                       f"If both do so within 1 Bin, answer '{SAME}'.", i, rng)


# 3. memory-bound vs compute-bound moment -------------------------------------------------
MB_OPTS = ['the most memory-bound moment', 'the most compute-bound moment']


def boundness(x):
    mem = (x[C[MEMORY[0]]] + x[C[MEMORY[1]]]) / 2
    comp = np.max([x[C[c]] for c in ['pipe: alu', 'pipe: fma', 'pipe: tensor_op_hmma', 'pipe: xu']], 0)
    return mem - comp


def ans_mem_compute(x, p=None):
    b = busy(x)
    if b.sum() < 8:
        return None
    s = np.where(b, boundness(x), np.nan)
    hi, lo = int(np.nanargmax(s)), int(np.nanargmin(s))
    if s[hi] - s[lo] < 0.1 or abs(hi - lo) <= 3:
        return None
    return MB_OPTS[0] if hi < lo else MB_OPTS[1]


def make_mem_compute(i, rng):
    x = X[i]
    a = ans_mem_compute(x)
    if a is None:
        return []
    q = (INTRO + "In each busy Bin, compare memory traffic (mean of DRAM Read and DRAM Write bandwidth) "
         "with the busiest compute pipe (max of pipe: alu, fma, tensor_op_hmma, xu). "
         "Which comes first in time: the most memory-bound moment or the most compute-bound moment?")
    return [_q('mem_compute', q, MB_OPTS, a, ans_mem_compute(reverse(x)), ans_mem_compute(shuffle(x, rng)), key='-')]


# 4. first burst vs last burst ------------------------------------------------------------
FL_OPTS = ['higher in the first burst', 'higher in the last burst', 'about equal']


def ans_first_last(x, p):
    bs = [(s, e) for s, e in bursts(x) if e - s >= 3]
    v = x[C[p['counter']]]
    if len(bs) < 2 or v.max() <= 0:
        return None
    f, l = np.median(v[bs[0][0]:bs[0][1]]), np.median(v[bs[-1][0]:bs[-1][1]])
    r = (l - f) / v.max()
    return FL_OPTS[1] if r > 0.1 else FL_OPTS[0] if r < -0.1 else FL_OPTS[2]


def cands_first_last(x):
    return [(c, {'counter': c}, ans_first_last(x, {'counter': c})) for c in VARYING]


def make_first_last(i, rng):
    x = X[i]
    pick = _pick('first_last', cands_first_last, x, rng, i)
    if pick is None:
        return []
    key, p, a = pick
    q = (INTRO + "A burst is a run of consecutive busy Bins (at least 3 Bins long). Compare the first "
         f"burst and the last burst. Is the counter '{key}' higher in the first burst, "
         "higher in the last burst, or about equal?")
    return [_q('first_last', q, FL_OPTS, a, ans_first_last(reverse(x), p), ans_first_last(shuffle(x, rng), p), key=key)]


# 6. number of cycles (bursts) ------------------------------------------------------------
def ans_cycles(x, p=None):
    n = len([1 for s, e in bursts(x) if e - s >= 3])
    return str(n) if n >= 1 else None


def make_cycles(i, rng):
    x = X[i]
    a = ans_cycles(x)
    if a is None:
        return []
    q = (INTRO + "A burst is a run of consecutive busy Bins at least 3 Bins long. Is the trace periodic? "
         "Answer with the number of bursts (1 means one continuous run).")
    return [_q('cycles', q, None, a, ans_cycles(reverse(x)), ans_cycles(shuffle(x, rng)), key='-')]


TEMPLATES = {'trend': make_trend, 'direction': make_direction, 'peak_order': make_peak_order,
             'onset': make_onset, 'first_last': make_first_last, 'mem_compute': make_mem_compute,
             'cycles': make_cycles}
