#!/usr/bin/env python3
"""Render a deterministic training-only gallery without changing dataset files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'kernel-xray-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace(root, record):
    path = (root / record['path']).resolve()
    if not path.is_relative_to((root / 'kernels').resolve()):
        raise ValueError(f'Trace path outside kernels/: {path}')
    label_path = path.parent.parent / 'labels/labels.npz'
    # Upstream string metadata uses object arrays; only load trusted local data.
    with np.load(path, allow_pickle=True) as data:
        x = data['data'].copy()
        names = data['counter_names'].astype(str).tolist()
        edges = data['time_edges_ns'].copy()
    with np.load(label_path, allow_pickle=True) as labels:
        y = labels['workload_l1'].copy()
        mask = labels['mask_labeled'].astype(bool)
        multi = labels['workload_l1_multihot'].copy()
        vocab = labels['vocab_l1'].astype(str).tolist()
        label_edges = labels['time_edges_ns'].copy()
        flagged = bool(labels['multihot_has_overlap'].item())
    if x.shape != (24, 512) or len(names) != 24 or len(vocab) != 12:
        raise ValueError(f'Unexpected tensor/vocabulary shape: {path}')
    if y.shape != (512,) or mask.shape != (512,) or multi.shape != (512, 12):
        raise ValueError(f'Unexpected label shapes: {label_path}')
    if edges.shape != (513,) or not np.all(np.diff(edges) > 0):
        raise ValueError(f'Invalid time grid: {path}')
    if not np.array_equal(edges, label_edges) or not np.isfinite(x).all():
        raise ValueError(f'Time alignment or finite-value check failed: {path}')
    if not np.array_equal(mask, y >= 0) or np.any(y < -1) or np.any(y >= len(vocab)):
        raise ValueError(f'Invalid primary labels or mask: {label_path}')
    if not np.isin(multi, [0, 1]).all() or not np.all(multi[np.where(mask)[0], y[mask]] == 1):
        raise ValueError(f'Invalid multi-hot labels: {label_path}')
    overlap = multi.sum(axis=1) >= 2
    if flagged != bool(overlap.any()):
        raise ValueError(f'Overlap flag disagrees with multi-hot labels: {label_path}')
    return dict(x=x, names=names, edges=edges, y=y, mask=mask, multi=multi,
                vocab=vocab, overlap=overlap, path=path, label_path=label_path, source_path=record['path'])


def spans(mask, edges):
    """Convert runs of true bins to exact [start, end] edge intervals."""
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return [(float(edges[a]), float(edges[b]))
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def select_traces(root, records, overlap_examples):
    # Sorting removes dependence on JSON record order; no random sampling.
    records = sorted(records, key=lambda record: record['path'])
    selected, used = [], set()
    classes = sorted({label for record in records for label in record['l1_labels']})
    for label in classes:
        for record in records:
            if record['path'] in used or label not in record['l1_labels']:
                continue
            trace = load_trace(root, record)
            idx = trace['vocab'].index(label)
            if np.any(trace['mask'] & (trace['y'] == idx)):
                selected.append((label, record, trace))
                used.add(record['path'])
                break
        else:
            raise ValueError(f'No distinct primary-label example for {label}')
    count = 0
    if overlap_examples:
        for record in records:
            if record['path'] in used:
                continue
            trace = load_trace(root, record)
            if trace['overlap'].any():
                selected.append(('overlap', record, trace))
                used.add(record['path'])
                count += 1
                if count == overlap_examples:
                    break
    if count != overlap_examples:
        raise ValueError(f'Requested {overlap_examples} distinct overlap examples; found {count}')
    return selected


def render(trace, category, destination, normalization, dpi):
    x, vocab = trace['x'], trace['vocab']
    scale = np.ones(24)
    if normalization == 'p99':
        scale = np.percentile(np.abs(x), 99, axis=1)
        scale = np.where(scale > 0, scale, 1.0)
        display = np.clip(x / scale[:, None], 0, 1)
        color_label = 'Channel activity / channel p99 (clipped to [0, 1])'
    else:
        display = x
        color_label = 'Stored counter value (dataset physical scaling)'
    time_ms = (trace['edges'] - trace['edges'][0]) / 1e6
    colors = list(plt.colormaps['tab20'].colors[:len(vocab)])
    cmap = ListedColormap(['#c9cdd2'] + colors)
    norm = BoundaryNorm(np.arange(-1.5, len(vocab) + .5), cmap.N)
    fig = plt.figure(figsize=(16, 12), layout='constrained')
    grid = fig.add_gridspec(4, 2, width_ratios=[1, .018], height_ratios=[5, .65, 2.7, 1])
    axes = [fig.add_subplot(grid[0, 0])]
    for row in range(1, 4):
        axes.append(fig.add_subplot(grid[row, 0], sharex=axes[0]))
    image = axes[0].pcolormesh(time_ms, np.arange(25), display,
                             cmap='viridis', shading='flat',
                             **({'vmin': 0, 'vmax': 1} if normalization == 'p99' else {}))
    axes[0].set_ylim(24, 0)
    axes[0].set_yticks(np.arange(24) + .5, trace['names'], fontsize=8)
    axes[0].set_title('Hardware counters — display scaling only; original tensor unchanged', loc='left')
    fig.colorbar(image, cax=fig.add_subplot(grid[0, 1]), label=color_label)
    axes[1].pcolormesh(time_ms, [0, 1], trace['y'][None, :], cmap=cmap, norm=norm, shading='flat')
    axes[1].set_yticks([])
    axes[1].set_title('Primary L1 label — gray = unlabeled (excluded from primary-label metrics)', loc='left')
    multi_colors = np.where(trace['multi'].T > 0, np.arange(len(vocab))[:, None], -1)
    axes[2].pcolormesh(time_ms, np.arange(len(vocab) + 1), multi_colors, cmap=cmap, norm=norm, shading='flat')
    axes[2].set_ylim(len(vocab), 0)
    axes[2].set_yticks(np.arange(len(vocab)) + .5, vocab, fontsize=9)
    axes[2].set_title('Multi-hot L1 ground truth — simultaneous colored rows indicate multiple classes', loc='left')
    active = trace['multi'].sum(axis=1)
    axes[3].stairs(active, time_ms, color='#243b53', linewidth=1.1)
    axes[3].set_ylim(-.1, max(2.5, float(active.max()) + .5))
    axes[3].set_yticks(range(int(active.max()) + 1))
    axes[3].set_ylabel('Active L1\nclasses')
    axes[3].set_xlabel('Time from first trace bin (ms)')
    intervals = spans(trace['overlap'], time_ms)
    for axis in (axes[0], axes[3]):
        for start, end in intervals:
            axis.axvspan(start, end, color='#e45756', alpha=.22, linewidth=0)
    axes[3].legend(handles=[Patch(facecolor='#e45756', alpha=.35, label='≥2 active L1 classes')], loc='upper right')
    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    axes[0].set_xlim(time_ms[0], time_ms[-1])
    fig.suptitle(f'Kernel X-ray | train | {category} | labeled {int(trace["mask"].sum())}/512 | '
                 f'overlap {int(trace["overlap"].sum())}/512 bins\n' +
                 trace["source_path"], fontsize=13)
    fig.savefig(destination, dpi=dpi, metadata={'Software': 'Kernel X-ray Phase 1 Task 6'})
    plt.close(fig)
    return dict(display_scale_per_channel=scale.tolist(), overlap_intervals_ms=intervals,
                labeled_bins=int(trace['mask'].sum()), overlap_bins=int(trace['overlap'].sum()),
                active_l1_classes=[vocab[i] for i in np.flatnonzero(trace['multi'].any(axis=0))])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/phase1/traces')
    parser.add_argument('--normalization', choices=['p99', 'stored'], default='p99')
    parser.add_argument('--overlap-examples', type=int, default=4)
    parser.add_argument('--dpi', type=int, default=140)
    args = parser.parse_args(argv)
    if args.overlap_examples < 0 or args.dpi < 50:
        parser.error('overlap-examples must be nonnegative and dpi must be at least 50')
    root, out = args.data_root.resolve(), args.output_dir.resolve()
    # Keep all writes out of the original data and upstream documentation.
    protected = [root / name for name in ('kernels', 'splits', 'tools', 'docs/dataset', '.cache')]
    if any(out == p or out.is_relative_to(p) for p in protected):
        parser.error('output-dir must be outside original dataset directories')
    split_path = root / 'splits/train.json'
    split = json.loads(split_path.read_text())
    records = split['traces']
    if len({r['path'] for r in records}) != len(records) or split['n'] != len(records):
        raise ValueError('Training split count or uniqueness check failed')
    selected = select_traces(root, records, args.overlap_examples)
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(split='train', split_sha256=sha256(split_path),
                    selection='lexicographic path; one distinct primary example per observed L1 class, then overlap examples',
                    normalization=args.normalization, dpi=args.dpi,
                    versions=dict(python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__), traces=[])
    for index, (category, record, trace) in enumerate(selected, 1):
        filename = f'{index:02d}_{category}.png'
        details = render(trace, category, out / filename, args.normalization, args.dpi)
        manifest['traces'].append(dict(category=category, path=record['path'], plot=filename,
                                       input_sha256=sha256(trace['path']), labels_sha256=sha256(trace['label_path']),
                                       plot_sha256=sha256(out / filename), **details))
        print(f'{filename}: {details["labeled_bins"]} labeled bins, {details["overlap_bins"]} overlap bins')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {len(selected)} training traces and manifest to {out}')


if __name__ == '__main__':
    main()
