"""Independent frozen-baseline arithmetic and functional inference; no training."""
import numpy as np
import torch
from torch.nn import functional as F


def count_confusion(truth, predicted):
    """Count explicit (true,predicted) pairs independently of the original bincount code."""
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.shape != predicted.shape or not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(predicted.dtype, np.integer):
        raise ValueError('Integer, aligned predictions required')
    if np.any((truth < 0) | (truth >= 12)) or np.any((predicted < 0) | (predicted >= 12)):
        raise ValueError('Unknown or out-of-range class cannot be evaluated')
    result = np.zeros((12, 12), dtype=np.int64)
    np.add.at(result, (truth, predicted), 1)
    return result


def metrics_from_counts(counts, supported, names):
    """Independent TP/FP/FN precision/recall and harmonic-mean F1."""
    cm = np.asarray(counts, dtype=np.int64)
    rows = []
    total, correct = int(cm.sum()), int(np.trace(cm))
    for c, name in enumerate(names):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum()) - tp; fn = int(cm[c].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(dict(class_id=c, name=name, tp=tp, fp=fp, fn=fn,
                         support=tp+fn, predicted=tp+fp, precision=precision, recall=recall, f1=f1))
    micro_p = correct / total if total else None
    micro_r = correct / total if total else None
    return {'confusion_matrix': cm.tolist(), 'valid_bins': total,
            'accuracy': correct / total if total else None,
            'micro_f1': 2*micro_p*micro_r/(micro_p+micro_r) if correct else (0.0 if total else None),
            'macro_f1_supported': float(np.mean([rows[c]['f1'] for c in supported])),
            'macro_f1_all12': float(np.mean([row['f1'] for row in rows])), 'per_class': rows}


def independent_temporal_masks(labels, radius=4):
    """Interior uses a nine-bin equality window; boundary uses adjacent valid changes."""
    labels = np.asarray(labels)
    n = len(labels)
    interior, boundary = np.zeros(n, bool), np.zeros(n, bool)
    for t in range(radius, n-radius):
        interior[t] = labels[t] >= 0 and np.all(labels[t-radius:t+radius+1] == labels[t])
    for t in range(1, n):
        if labels[t-1] >= 0 and labels[t] >= 0 and labels[t-1] != labels[t]:
            for position in range(max(0, t-radius), min(n, t+radius)):
                if labels[position] >= 0:
                    boundary[position] = True
    return interior, boundary


def functional_frozen_forward(x, weights):
    """Explicit eval-mode operations from the actual state keys, independent of CNN.forward."""
    def linear_norm(h, prefix):
        return F.layer_norm(h.transpose(1,2), (64,), weights[prefix+'.weight'], weights[prefix+'.bias'], 1e-5).transpose(1,2)
    h = F.conv1d(x, weights['projection.weight'], weights['projection.bias'])
    for block, dilation in enumerate([1,2,4]):
        residual = h
        for norm_index, conv_index in [(0,2),(4,6)]:
            h = F.gelu(linear_norm(h, f'blocks.{block}.temporal.{norm_index}.norm'))
            h = F.conv1d(h, weights[f'blocks.{block}.temporal.{conv_index}.weight'],
                         weights[f'blocks.{block}.temporal.{conv_index}.bias'], padding=2*dilation, dilation=dilation)
        h = residual + h
    return F.conv1d(F.gelu(linear_norm(h, 'final_norm.norm')), weights['head.weight'], weights['head.bias'])


def bootstrap_whole_traces(per_trace_counts, supported, replicates=10000, seed=20261119):
    """Percentile intervals on pooled counts from resampled whole trace clusters."""
    n = len(next(iter(per_trace_counts.values())))
    generator = np.random.default_rng(seed)
    multiplicity = generator.multinomial(n, np.full(n, 1/n), size=replicates)
    distributions, report = {}, {}
    for name, trace_cm in per_trace_counts.items():
        pooled = np.tensordot(multiplicity, trace_cm, axes=(1,0))
        tp = np.diagonal(pooled, axis1=1, axis2=2)
        support, predicted = pooled.sum(2), pooled.sum(1)
        denom = 2*tp+(predicted-tp)+(support-tp)
        f1 = np.divide(2*tp, denom, out=np.zeros_like(tp,dtype=float), where=denom>0)
        macro = f1[:,supported].mean(1)
        accuracy = tp.sum(1)/pooled.sum((1,2))
        missing = support[:,supported] == 0
        distributions[name] = {'macro_f1':macro, 'accuracy':accuracy}
        report[name] = {'macro_f1_supported_percentile95':np.quantile(macro,[.025,.975]).tolist(),
                        'accuracy_percentile95':np.quantile(accuracy,[.025,.975]).tolist(),
                        'replicates_missing_any_supported_class':int(missing.any(1).sum()),
                        'missing_by_supported_class':{str(c):int(missing[:,i].sum()) for i,c in enumerate(supported)},
                        'observed_class_trace_support':{str(c):int((trace_cm.sum(2)[:,c]>0).sum()) for c in supported}}
    difference = distributions['weighted']['macro_f1']-distributions['unweighted']['macro_f1']
    report['paired_weighted_minus_unweighted'] = {
        'macro_f1_percentile95':np.quantile(difference,[.025,.975]).tolist(),
        'fraction_greater_than_zero':float((difference>0).mean()),
        'same_trace_resamples_for_both_conditions':True}
    return report
