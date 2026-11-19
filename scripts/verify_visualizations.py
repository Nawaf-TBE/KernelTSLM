#!/usr/bin/env python3
"""Check a gallery against training data; does not load validation/test tensors."""
import argparse
import json
from pathlib import Path

import numpy as np
from visualize_traces import ROOT, load_trace, sha256, spans


def verify(root, gallery):
    manifest = json.loads((gallery / 'manifest.json').read_text())
    assert manifest['split'] == 'train', 'Gallery must use training only'
    assert manifest['split_sha256'] == sha256(root / 'splits/train.json'), 'Split hash changed'
    splits = {name: json.loads((root / f'splits/{name}.json').read_text())
              for name in ('train', 'val', 'test')}
    paths = {name: {r['path'] for r in split['traces']} for name, split in splits.items()}
    for name, split in splits.items():
        assert split['n'] == len(split['traces']) == len(paths[name]), f'{name} count mismatch'
    assert not (paths['train'] & paths['val'] or paths['train'] & paths['test'] or paths['val'] & paths['test'])
    selected = set()
    primary_classes = set()
    overlap_examples = 0
    for item in manifest['traces']:
        assert item['path'] in paths['train'] and item['path'] not in selected
        selected.add(item['path'])
        trace = load_trace(root, item)
        assert sha256(trace['path']) == item['input_sha256']
        assert sha256(trace['label_path']) == item['labels_sha256']
        plot = (gallery / item['plot']).resolve()
        assert plot.is_relative_to(gallery.resolve()), 'Plot outside gallery'
        assert plot.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert sha256(plot) == item['plot_sha256']
        assert int(trace['mask'].sum()) == item['labeled_bins']
        assert int(trace['overlap'].sum()) == item['overlap_bins']
        edges = (trace['edges'] - trace['edges'][0]) / 1e6
        intervals = spans(trace['overlap'], edges)
        assert [list(s) for s in intervals] == item['overlap_intervals_ms']
        assert np.isclose(sum(b-a for a, b in intervals), np.diff(edges)[trace['overlap']].sum())
        expected_scale = np.ones(24)
        if manifest['normalization'] == 'p99':
            expected_scale = np.percentile(np.abs(trace['x']), 99, axis=1)
            expected_scale = np.where(expected_scale > 0, expected_scale, 1.)
        if manifest['versions']['numpy'] == np.__version__:
            np.testing.assert_array_equal(expected_scale, item['display_scale_per_channel'])
        else:
            # Percentile interpolation differs slightly across NumPy versions
            # on float32 tensors. Source-file and plot hashes remain exact.
            np.testing.assert_allclose(expected_scale, item['display_scale_per_channel'],
                                       rtol=8 * np.finfo(np.float32).eps, atol=0)
        assert item['active_l1_classes'] == [trace['vocab'][i] for i in np.flatnonzero(trace['multi'].any(axis=0))]
        category = item['category']
        if category == 'overlap':
            assert trace['overlap'].any()
            overlap_examples += 1
        else:
            index = trace['vocab'].index(category)
            assert np.any(trace['mask'] & (trace['y'] == index))
            primary_classes.add(category)
    assert primary_classes == {label for r in splits['train']['traces'] for label in r['l1_labels']}
    # Edge cases for contiguous intervals, including runs touching either end.
    assert spans(np.array([True, False, True, True]), np.arange(5)) == [(0., 1.), (2., 4.)]
    assert spans(np.zeros(4, dtype=bool), np.arange(5)) == []
    return dict(status='passed', split_counts={name: len(p) for name, p in paths.items()},
                shared_trace_paths=0, verified_plots=len(selected),
                observed_primary_l1_classes=sorted(primary_classes), overlap_examples=overlap_examples,
                checks=['training-only selection', 'split counts and path disjointness',
                        'source and plot SHA-256', 'tensor and label alignment/consistency',
                        'category presence', 'overlap bins and exact interval coverage',
                        'display scaling', 'PNG signatures', 'interval edge cases'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--gallery', type=Path, default=ROOT / 'outputs/phase1/traces')
    args = parser.parse_args()
    print(json.dumps(verify(args.data_root.resolve(), args.gallery.resolve()), indent=2))


if __name__ == '__main__':
    main()
