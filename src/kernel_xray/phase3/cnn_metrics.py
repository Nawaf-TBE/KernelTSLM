"""Fixed-vocabulary classification and predeclared temporal diagnostics."""
import numpy as np


DIAGNOSTIC_PROTOCOL = {
    'boundary_radius_bins': 4,
    'boundary': 'four bins on each side of adjacent valid differing primary labels; intersect valid mask',
    'interior': 'valid constant-class runs length >=9; discard four bins at each edge; never bridge unknowns',
    'multiclass_trace': 'at least two distinct valid primary L1 classes; score all its valid bins',
    'single_class_collapse_fraction': 0.95,
    'per_trace_macro': 'fixed 11 training-supported classes, zero_division=0; empty traces excluded',
    'per_trace_present_macro': 'classes present in the trace valid truth, zero_division=0',
    'selection': 'validation macro-F1 over fixed training-supported classes; strict improvement, earliest epoch on tie',
    'experiment_tie': 'unweighted',
}


def confusion_matrix(truth, prediction, classes=12):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or np.any((truth < 0) | (truth >= classes)) or np.any((prediction < 0) | (prediction >= classes)):
        raise ValueError('Confusion inputs must contain aligned valid vocabulary IDs')
    return np.bincount((classes * truth + prediction).astype(np.int64), minlength=classes**2).reshape(classes, classes)


def classification_metrics(cm, supported, names):
    cm = np.asarray(cm, dtype=np.int64)
    tp, support, predicted = np.diag(cm), cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(len(names)), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(names)), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(names)), where=support + predicted > 0)
    total = int(cm.sum())
    accuracy = float(tp.sum() / total) if total else None
    return {
        'valid_bins': total, 'accuracy': accuracy, 'micro_f1': accuracy,
        'macro_f1_supported': float(f1[supported].mean()), 'macro_f1_all12': float(f1.mean()),
        'supported_class_ids': list(map(int, supported)),
        'unsupported_convention': 'all 12 IDs retained; zero_division=0; no truth/prediction gives F1=0',
        'per_class': [dict(class_id=i, name=name, precision=float(precision[i]), recall=float(recall[i]),
                           f1=float(f1[i]), support=int(support[i]), predicted=int(predicted[i]))
                      for i, name in enumerate(names)],
        'prediction_distribution': [int(x) for x in predicted], 'confusion_matrix': cm.tolist(),
        'single_class_collapse': bool(total and predicted.max() / total >= 0.95),
    }


def temporal_masks(y, radius=4):
    """Return interior/boundary masks without treating unknowns as operation changes."""
    y = np.asarray(y)
    valid = y >= 0
    boundary, interior = np.zeros(len(y), bool), np.zeros(len(y), bool)
    changes = np.flatnonzero(valid[:-1] & valid[1:] & (y[:-1] != y[1:])) + 1
    for change in changes:
        boundary[max(0, change - radius):min(len(y), change + radius)] = True
    start = 0
    while start < len(y):
        if y[start] < 0:
            start += 1
            continue
        stop = start + 1
        while stop < len(y) and y[stop] == y[start]:
            stop += 1
        if stop - start >= 2 * radius + 1:
            interior[start + radius:stop - radius] = True
        start = stop
    return interior & valid, boundary & valid


def common_confusions(cm, names, count=10):
    cm = np.asarray(cm)
    pairs = [(int(cm[i, j]), i, j) for i in range(len(names)) for j in range(len(names)) if i != j and cm[i, j]]
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [dict(true=names[i], predicted=names[j], bins=n, true_class_error_fraction=n / int(cm[i].sum()))
            for n, i, j in pairs[:count]]
