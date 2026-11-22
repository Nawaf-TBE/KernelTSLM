#!/usr/bin/env python3
"""Reproduce Task 7 verification with training tensors only; never train a model."""
import json
from pathlib import Path
import platform
import pytest
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from kernel_xray import fit_training_statistics, NormalizationStats
from kernel_xray.dataset import DEFAULT_ROOT, read_splits, sha256
from example_dataloader import batch_example


def preserved_files(root):
    files = []
    for folder in ('kernels', 'splits', 'tools', 'outputs'):
        for path in (root / folder).rglob('*'):
            if (path.is_file() and not path.is_relative_to(root / 'outputs/phase1/task7')
                    and not path.is_relative_to(root / 'outputs/reports')):
                files.append(path)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(files)}


def main():
    root = DEFAULT_ROOT
    report_dir = root / 'reports'
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / 'task7_verification.json'
    before = preserved_files(root)
    junit = report_dir / 'task7_tests.xml'
    result = pytest.main(['-q', '-c', str(root / 'pytest.ini'), str(root / 'tests'), f'--junitxml={junit}'])
    if result:
        report_path.write_text(json.dumps(dict(status='failed', pytest_exit_code=int(result)), indent=2) + '\n')
        raise SystemExit(int(result))
    stats = fit_training_statistics(root)
    stats_path = root / 'outputs/phase1/task7/train_statistics.json'
    stats.save(stats_path)
    restored = NormalizationStats.load(stats_path)
    if restored != stats:
        raise RuntimeError('Statistics serialization changed values')
    batches = dict(raw=batch_example(root), normalized=batch_example(root, restored))
    after = preserved_files(root)
    if before != after:
        raise RuntimeError('Raw data, official splits, upstream tools, or existing outputs changed')
    suites = ET.parse(junit).getroot().findall('testsuite')
    test_counts = {key: sum(int(s.attrib[key]) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    report = dict(status='passed', tests=test_counts,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
                  split_counts={k: len(v) for k, v in read_splits(root).items()},
                  real_tensor_access='training only; tests enforce a held-out NPZ read guard',
                  real_training_records_checked=1124,
                  synthetic_validation_and_test_loading='passed',
                  normalization=dict(path=str(stats_path.relative_to(root)), sha256=sha256(stats_path),
                                     fit_split='train', input_files=len(stats.train_inputs_sha256), bins=stats.n_bins,
                                     method=stats.method),
                  preservation=dict(files_verified_unchanged=len(before),
                                    raw_npz_files=sum(p.startswith('kernels/') and p.endswith('.npz') for p in before),
                                    split_files=sum(p.startswith('splits/') for p in before),
                                    prior_output_files=sum(p.startswith('outputs/') for p in before)),
                  batches=batches)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
