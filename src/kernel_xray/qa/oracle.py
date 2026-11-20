"""Float64 arithmetic on unchanged normalized float32 inputs."""
import numpy as np

TYPES = ('statistics', 'comparison', 'temporal')


class Rejected(ValueError):
    pass


def validate(x, names, kind, channels, start, end):
    if not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.shape != (24, 512) or not np.isfinite(x).all():
        raise ValueError('Expected finite float32 [24,512] normalized counters')
    if len(names) != 24 or len(set(names)) != 24 or not all(isinstance(n,str) for n in names):
        raise ValueError('Expected 24 unique ordered channel names')
    if kind not in TYPES or len(channels) != (1 if kind == 'statistics' else 2):
        raise ValueError('Question type/channel arity')
    if any(type(c) is not int or not 0 <= c < 24 for c in channels) or len(set(channels)) != len(channels):
        raise ValueError('Distinct valid channel indices required')
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= 512:
        raise ValueError('Expected nonempty half-open interval within [0,512]')


def question(kind, names, channels, start, end):
    a = names[channels[0]]
    if kind == 'statistics':
        return f'What is the average value of {a} between time bins {start} and {end}? Use [{start}, {end}) in normalized units.'
    b = names[channels[1]]
    if kind == 'comparison':
        return f'Which counter has the higher maximum value between bins [{start}, {end}): {a} or {b}? Values are normalized.'
    return f'Does {a} reach its maximum before {b} within bins [{start}, {end})? Values are normalized.'


def oracle(x, names, kind, channels, start, end, config):
    validate(x, names, kind, channels, start, end)
    values = x[channels, start:end].astype(np.float64)
    if kind == 'statistics':
        value = float(values[0].mean(dtype=np.float64))
        return value, {'sum':float(values[0].sum(dtype=np.float64)), 'count':end-start}
    maxima = values.max(axis=1)
    if kind == 'comparison':
        gap = float(abs(maxima[0]-maxima[1]))
        threshold = config['comparison_atol'] + config['comparison_rtol'] * float(np.abs(maxima).max())
        if gap == 0:
            answer = 'Tie'
        elif gap <= threshold:
            raise Rejected('comparison_near_tie')
        else:
            answer = 'Counter A' if maxima[0] > maxima[1] else 'Counter B'
        return answer, {'maxima':maxima.tolist(), 'gap':gap, 'separation_threshold':threshold}
    if end-start < 2:
        raise Rejected('insufficient_peak_samples')
    ordered = np.sort(values, axis=1)
    gaps = ordered[:,-1]-ordered[:,-2]
    thresholds = config['peak_atol'] + config['peak_rtol'] * np.abs(maxima)
    if np.any(gaps <= thresholds):
        raise Rejected('nonunique_or_ambiguous_peak')
    bins = (np.argmax(values,axis=1)+start).tolist()
    answer = 'Yes' if bins[0]<bins[1] else ('No' if bins[0]>bins[1] else 'Same bin')
    return answer, {'maxima':maxima.tolist(), 'peak_bins':bins, 'runner_up':ordered[:,-2].tolist(),
                    'peak_gaps':gaps.tolist(), 'peak_thresholds':thresholds.tolist()}
