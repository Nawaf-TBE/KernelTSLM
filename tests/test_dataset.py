from dataclasses import replace
import json
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from kernel_xray import KernelSightDataset, NormalizationStats, collate_records, fit_training_statistics
from kernel_xray.dataset import DEFAULT_ROOT, read_splits


def rewrite_npz(path, key, transform):
    with np.load(path, allow_pickle=True) as z:
        data = {k: z[k].copy() for k in z.files}
    data[key] = transform(data[key])
    np.savez(path, **data)


def test_all_real_training_records_preserve_schema_labels_and_order(train):
    schema = None
    overlap_traces = overlap_bins = unlabeled_bins = 0
    for index, rec in enumerate(train.records):
        item = train[index]
        assert item['features'].shape == (24, 512)
        assert item['features'].dtype == torch.float32
        assert item['features'].data_ptr() != item['raw_features'].data_ptr()
        assert item['time_edges_ns'].shape == (513,)
        assert item['time_edges_ns'].dtype == torch.int64
        path = DEFAULT_ROOT / rec['path']
        with np.load(path, allow_pickle=True) as raw:
            np.testing.assert_array_equal(item['raw_features'], raw['data'])
            np.testing.assert_array_equal(item['features'], raw['data'])
            np.testing.assert_array_equal(item['time_edges_ns'], raw['time_edges_ns'])
            assert item['metadata']['counter_names'] == tuple(raw['counter_names'])
            for key in ('kernels', 'kernel_names', 'kernel_function_index'):
                np.testing.assert_array_equal(item['metadata']['input'][key], raw[key])
        with np.load(path.parent.parent / 'labels/labels.npz', allow_pickle=True) as lab:
            assert item['metadata']['path'] == rec['path']
            assert item['metadata']['split'] == 'train'
            for level, classes in [('l1', 12), ('l2', 73)]:
                y = item['targets'][f'workload_{level}']
                mh = item['targets'][f'workload_{level}_multihot']
                valid = item['masks'][f'valid_{level}']
                assert y.dtype == torch.int64 and y.shape == (512,)
                assert mh.dtype == torch.float32 and mh.shape == (512, classes)
                assert valid.dtype == torch.bool
                np.testing.assert_array_equal(y, lab[f'workload_{level}'])
                np.testing.assert_array_equal(mh, lab[f'workload_{level}_multihot'])
                assert torch.equal(valid, y >= 0)
                assert bool(((y == -1) | ((y >= 0) & (y < classes))).all())
                assert bool(mh[torch.arange(512)[valid], y[valid]].all())
                assert torch.equal(item['masks'][f'multihot_{level}'], mh.bool().any(dim=1))
            for key in ('mask_labeled', 'mask_any_kernel'):
                np.testing.assert_array_equal(item['masks'][key], lab[key].astype(bool))
            active = item['targets']['workload_l1_multihot'].sum(dim=1).long()
            assert torch.equal(item['targets']['multihot_n_active'], active)
            assert torch.equal(item['targets']['overlap_l1'], active >= 2)
            assert item['targets']['multihot_has_overlap'].item() == bool((active >= 2).any())
            np.testing.assert_array_equal(active, lab['multihot_n_active'])
            assert item['targets']['multihot_has_overlap'].item() == bool(lab['multihot_has_overlap'])
            assert torch.equal(item['masks']['mask_labeled'], item['masks']['valid_l1'])
        current = (item['metadata']['counter_names'], item['metadata']['vocab_l1'], item['metadata']['vocab_l2'])
        if schema is None:
            schema = current
        assert current == schema
        overlap_traces += int(item['targets']['multihot_has_overlap'])
        overlap_bins += int(item['targets']['overlap_l1'].sum())
        unlabeled_bins += int((item['targets']['workload_l1'] == -1).sum())
    assert len(train) == 1124
    assert overlap_traces > 0 and overlap_bins > 0 and unlabeled_bins > 0


@pytest.mark.parametrize('split,count', [('train', 1124), ('val', 160), ('test', 160)])
def test_official_splits_lazy_paths_and_counts(split, count, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Constructor must not load any tensor')
    monkeypatch.setattr(np, 'load', forbidden)
    dataset = KernelSightDataset(split=split)
    by_path = KernelSightDataset(split=DEFAULT_ROOT / f'splits/{split}.json')
    assert len(dataset) == len(by_path) == count
    assert dataset.records == by_path.records
    paths = {name: {r['path'] for r in records} for name, records in read_splits(DEFAULT_ROOT).items()}
    assert not (paths['train'] & paths['val'] or paths['train'] & paths['test'] or paths['val'] & paths['test'])


@pytest.mark.parametrize('split', ['iid', 'param_ood', 'composed', 'length_ood', '/tmp/custom.json'])
def test_reject_analysis_tags_and_nonofficial_paths(split):
    with pytest.raises(ValueError, match='official'):
        KernelSightDataset(split=split)


@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_synthetic_heldout_loading_with_training_statistics(synthetic_root, synthetic_stats, split):
    raw = KernelSightDataset(synthetic_root, split)[0]
    normalized = KernelSightDataset(synthetic_root, split, synthetic_stats)[0]
    assert torch.equal(raw['raw_features'], normalized['raw_features'])
    expected = ((raw['raw_features'].numpy().astype(np.float64) - np.asarray(synthetic_stats.mean)[:, None]) /
                synthetic_stats.scale[:, None]).astype(np.float32)
    np.testing.assert_array_equal(normalized['features'], expected)
    assert torch.equal(raw['time_edges_ns'], normalized['time_edges_ns'])
    for key in raw['targets']:
        assert torch.equal(raw['targets'][key], normalized['targets'][key])
    assert raw['metadata']['counter_names'] == normalized['metadata']['counter_names']


def test_fit_reads_only_train_inputs_and_known_population(synthetic_root, monkeypatch):
    original, opened = np.load, []
    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', spy)
    stats = fit_training_statistics(synthetic_root)
    assert len(opened) == 2
    assert all('/train' in path and path.endswith('/input/tensor_input.npz') for path in opened)
    assert stats.n_bins == 1024
    np.testing.assert_array_equal(stats.mean, [2., 4.] + [0.] * 22)
    np.testing.assert_array_equal(stats.std, [2., 4.] + [0.] * 22)
    np.testing.assert_array_equal(stats.scale, [2., 4.] + [1.] * 22)
    # Changes to either held-out tensor cannot affect fitted statistics.
    for name in ('val', 'test'):
        path = synthetic_root / read_splits(synthetic_root)[name][0]['path']
        rewrite_npz(path, 'data', lambda x: x + 9999)
    assert fit_training_statistics(synthetic_root) == stats


def test_real_training_stats_against_independent_numpy_reference(train):
    stats = fit_training_statistics(DEFAULT_ROOT)
    blocks = []
    for rec in train.records:
        with np.load(DEFAULT_ROOT / rec['path'], allow_pickle=True) as z:
            blocks.append(z['data'].astype(np.float64))
    values = np.concatenate(blocks, axis=1)
    assert stats.n_bins == 1124 * 512
    np.testing.assert_allclose(stats.mean, values.mean(axis=1), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stats.std, values.std(axis=1), rtol=1e-12, atol=1e-12)
    normalized = KernelSightDataset(normalization=stats)[0]
    assert torch.equal(normalized['raw_features'], train[0]['raw_features'])
    np.testing.assert_allclose(normalized['features'],
                               ((blocks[0] - np.asarray(stats.mean)[:, None]) / stats.scale[:, None]).astype(np.float32))


def test_stats_roundtrip_and_determinism(synthetic_root, synthetic_stats, tmp_path):
    path = tmp_path / 'stats.json'
    synthetic_stats.save(path)
    first = path.read_bytes()
    fit_training_statistics(synthetic_root).save(path)
    assert path.read_bytes() == first
    assert NormalizationStats.load(path) == synthetic_stats
    synthetic_stats.validate_sources(synthetic_root)


@pytest.mark.parametrize('change', ['split', 'input', 'population', 'count'])
def test_reject_stale_or_nontraining_statistics(synthetic_root, synthetic_stats, change):
    stats = synthetic_stats
    if change == 'split':
        p = synthetic_root / 'splits/train.json'
        p.write_text(p.read_text() + '\n')
    elif change == 'input':
        p = synthetic_root / read_splits(synthetic_root)['train'][0]['path']
        rewrite_npz(p, 'data', lambda x: x + 1)
    elif change == 'population':
        stats = replace(stats, train_inputs_sha256={'kernels/fixture/_out/test0/input/tensor_input.npz': 'bad'})
    else:
        stats = replace(stats, n_bins=512)
    with pytest.raises(ValueError):
        KernelSightDataset(synthetic_root, normalization=stats)


@pytest.mark.parametrize('corruption,message', [('time', 'timestamps'), ('primary', 'primary IDs'),
                                               ('mask', 'mask mismatch'), ('multi', 'primary class absent'),
                                               ('overlap', 'overlap counts'), ('hierarchy', 'parents')])
def test_reject_corrupt_label_contract(synthetic_root, corruption, message):
    path = synthetic_root / read_splits(synthetic_root)['train'][0]['path']
    label = path.parent.parent / 'labels/labels.npz'
    key, transform = {
        'time': ('time_edges_ns', lambda x: x + 1),
        'primary': ('workload_l2', lambda x: np.full_like(x, 73)),
        'mask': ('mask_labeled', lambda x: 1 - x),
        'multi': ('workload_l2_multihot', lambda x: np.zeros_like(x)),
        'overlap': ('multihot_n_active', lambda x: x + 1),
        'hierarchy': ('l2_parent_l1', lambda x: np.full_like(x, 12)),
    }[corruption]
    rewrite_npz(label, key, transform)
    with pytest.raises(ValueError, match=message):
        KernelSightDataset(synthetic_root)[0]


@pytest.mark.parametrize('corruption', ['duplicate', 'cross_split', 'escape', 'count'])
def test_reject_bad_split_indexes(synthetic_root, corruption):
    path = synthetic_root / 'splits/train.json'
    data = json.loads(path.read_text())
    if corruption == 'duplicate':
        data['traces'][1] = data['traces'][0]
    elif corruption == 'cross_split':
        data['traces'][0]['path'] = read_splits(synthetic_root)['val'][0]['path']
    elif corruption == 'escape':
        data['traces'][0]['path'] = '../../outside/input/tensor_input.npz'
    else:
        data['n'] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        KernelSightDataset(synthetic_root)


def test_channel_order_rejected_for_stats_and_collation(synthetic_root, synthetic_stats):
    data = KernelSightDataset(synthetic_root)
    a, b = data[0], data[1]
    b['metadata']['counter_names'] = tuple(reversed(b['metadata']['counter_names']))
    with pytest.raises(ValueError, match='schema/order'):
        collate_records([a, b])
    path = synthetic_root / data.records[0]['path']
    rewrite_npz(path, 'counter_names', lambda x: x[::-1])
    with pytest.raises(ValueError, match='channel order'):
        fit_training_statistics(synthetic_root)


def test_batches_reproducibility_and_metadata(train):
    def loader():
        return DataLoader(train, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(7),
                          num_workers=0, collate_fn=collate_records)
    a, b = next(iter(loader())), next(iter(loader()))
    assert a['features'].shape == a['raw_features'].shape == (4, 24, 512)
    assert a['time_edges_ns'].shape == (4, 513)
    assert a['targets']['workload_l1'].shape == a['targets']['workload_l2'].shape == (4, 512)
    assert a['targets']['workload_l1_multihot'].shape == (4, 512, 12)
    assert a['targets']['workload_l2_multihot'].shape == (4, 512, 73)
    assert a['masks']['valid_l1'].shape == (4, 512)
    assert len(a['metadata']) == 4
    assert torch.equal(a['features'], b['features'])
    assert [m['path'] for m in a['metadata']] == [m['path'] for m in b['metadata']]
    assert torch.equal(pickle.loads(pickle.dumps(train))[0]['features'], train[0]['features'])


def test_record_ownership(train):
    item = train[0]
    before = item['raw_features'].clone()
    item['features'].zero_()
    assert torch.equal(item['raw_features'], before)
    item['raw_features'].zero_()
    item['targets']['workload_l1'].fill_(11)
    assert torch.equal(train[0]['raw_features'], before)
    records = train.records
    records[0]['path'] = 'bad'
    assert train.records[0]['path'] != 'bad'
