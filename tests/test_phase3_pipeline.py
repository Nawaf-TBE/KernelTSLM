"""TimeF baseline contracts; no real test Records or test NPZs may be decoded."""
import copy
from dataclasses import replace
import pickle

import numpy as np
import pytest
import torch

pytest.importorskip('timenet')
pytest.importorskip('timenet_connectors')

from timenet.dataset import Record, Signal, Source
from timenet.dataset.axis import OrdinalAxis
from timenet.format.control_reader import DuckDBControlReader
from timenet.format.duckdb import connect_control
from timenet.types import TimeSeriesSpec, ureg
from timenet.reader import LoadPlan
from kernel_xray.dataset import NormalizationStats, KernelSightDataset
from kernel_xray.phase3 import (
    TimeFTraceDataset, collate_traces, flatten_labeled_bins, make_dataloaders, valid_primary_logits,
)
from kernel_xray.phase3.timef_dataset import DEFAULT_VERSION, DEFAULT_VERSIONS, FULL_VERSION, record_to_sample


@pytest.fixture(autouse=True)
def forbid_test_record_decoding(monkeypatch):
    with connect_control(DEFAULT_VERSION / 'control.duckdb', read_only=True) as conn:
        allowed = {row[0] for row in conn.execute('''
            SELECT DISTINCT rr.record_id FROM task_record_refs rr JOIN tasks t USING (task_id)
            WHERE rr.field = 'inputs' AND t.split IN ('train', 'validation')
        ''').fetchall()}
    original = DuckDBControlReader.read_records

    def guarded(reader, ids):
        assert set(ids) <= allowed, 'Real test TimeF Record decode'
        return original(reader, ids)

    monkeypatch.setattr(DuckDBControlReader, 'read_records', guarded)
    from timenet.storage.parquet import SignalRowGroupCache
    read_values = SignalRowGroupCache.read_values

    def guard_group(cache, version, *args, **kwargs):
        from pathlib import Path
        assert Path(version.root).resolve() in {p.resolve() for p in DEFAULT_VERSIONS.values()}
        return read_values(cache, version, *args, **kwargs)

    monkeypatch.setattr(SignalRowGroupCache, 'read_values', guard_group)


@pytest.fixture
def dataset():
    value = TimeFTraceDataset()
    yield value
    value.close()


def test_batch_shapes_and_cpu(dataset):
    batch = collate_traces(dataset.__getitems__([0, 1, 2, 3]))
    assert batch['X'].shape == batch['raw_X'].shape == (4, 24, 512)
    assert batch['X'].dtype == torch.float32 and batch['X'].device.type == 'cpu'
    assert batch['time_edges_ns'].shape == (4, 513)
    assert batch['time_edges_ns'].dtype == torch.int64
    for level, classes in [('l1', 12), ('l2', 73)]:
        assert batch['targets'][f'workload_{level}'].shape == (4, 512)
        assert batch['targets'][f'workload_{level}'].dtype == torch.int64
        assert batch['targets'][f'workload_{level}_multihot'].shape == (4, 512, classes)
        assert batch['targets'][f'workload_{level}_multihot'].dtype == torch.float32
        assert batch['masks'][f'valid_{level}'].shape == (4, 512)
        assert batch['masks'][f'valid_{level}'].dtype == torch.bool


@pytest.mark.parametrize('index', [0, 301, 700, 1123])
def test_exact_phase1_source_and_order(dataset, index):
    item = dataset[index]
    source = KernelSightDataset(split='train', normalization=dataset.normalization)[index]
    assert torch.equal(item['X'], source['features'])
    assert item['raw_X'].numpy().tobytes() == source['raw_features'].numpy().tobytes()
    assert torch.equal(item['time_edges_ns'], source['time_edges_ns'])
    for key in source['targets']:
        assert torch.equal(item['targets'][key], source['targets'][key])
    for key in source['masks']:
        assert torch.equal(item['masks'][key], source['masks'][key])
    for key in ['counter_names', 'vocab_l1', 'vocab_l2']:
        assert item['metadata'][key] == source['metadata'][key]
    assert item['metadata']['l2_parent_l1'] == tuple(source['metadata']['l2_parent_l1'].tolist())


def test_official_split_isolation_and_validation_shape(dataset):
    val = TimeFTraceDataset('val')
    try:
        assert len(dataset) == 1124 and len(val) == 160
        assert not {x.path for x in dataset.identities} & {x.path for x in val.identities}
        assert [x.split_index for x in dataset.identities] == list(range(1124))
        assert val[0]['metadata']['split'] == 'val'
        assert val[0]['X'].shape == (24, 512)
        with pytest.raises(ValueError, match='reserved'):
            TimeFTraceDataset('test')
    finally:
        val.close()


def test_deterministic_dataloaders():
    a = make_dataloaders(seed=7, batch_size=4)
    b = make_dataloaders(seed=7, batch_size=4)
    c = make_dataloaders(seed=8, batch_size=4)
    try:
        paths = lambda batch: [m['trace_id'] for m in batch['metadata']]
        first = paths(next(iter(a['train'])))
        assert first == paths(next(iter(b['train'])))
        assert paths(next(iter(c['train']))) != first
        assert paths(next(iter(a['val']))) == [x.path for x in a['val'].dataset.identities[:4]]
    finally:
        for loaders in [a, b, c]:
            for loader in loaders.values():
                loader.dataset.close()


def test_masked_flatten_and_temporal_alignment(dataset):
    batch = collate_traces(dataset.__getitems__([0, 1]))
    for level, classes in [('l1', 12), ('l2', 73)]:
        flat = flatten_labeled_bins(batch, level)
        assert flat['X'].shape == (int(batch['masks'][f'valid_{level}'].sum()), 24)
        assert (flat['y'] >= 0).all()
        assert torch.equal(flat['X'], batch['X'][flat['trace_index'], :, flat['bin_index']])
        logits = torch.arange(2 * classes * 512).reshape(2, classes, 512).float()
        selected, targets = valid_primary_logits(logits, batch, level)
        assert torch.equal(targets, flat['y'])
        assert torch.equal(selected, logits[flat['trace_index'], :, flat['bin_index']])
    unknown = copy.deepcopy(batch)
    unknown['targets']['workload_l1'].fill_(-1)
    unknown['masks']['valid_l1'].zero_()
    assert flatten_labeled_bins(unknown)['X'].shape == (0, 24)
    assert flatten_labeled_bins(unknown)['y'].shape == (0,)


def test_mask_and_mixed_split_errors(dataset):
    sample = dataset[0]
    wrong = copy.deepcopy(sample)
    wrong['metadata']['split'] = 'val'
    with pytest.raises(ValueError, match='Mixed split'):
        collate_traces([sample, wrong])
    batch = collate_traces([sample])
    batch['masks']['valid_l1'] = ~batch['masks']['valid_l1']
    with pytest.raises(ValueError, match='alignment'):
        flatten_labeled_bins(batch)


def test_overlap_multihot_and_unknowns(dataset):
    index = next(i for i, item in enumerate(dataset.identities) if 'cutlass_ws_overlap' in item.path)
    item = dataset[index]
    active = item['targets']['workload_l1_multihot'].sum(-1).long()
    assert torch.equal(item['targets']['multihot_n_active'], active)
    assert torch.equal(item['targets']['overlap_l1'], active >= 2)
    assert torch.equal(item['masks']['overlap_l1_valid'], active > 0)
    assert (active >= 2).any()
    for level in ['l1', 'l2']:
        y = item['targets'][f'workload_{level}']
        assert torch.equal(item['masks'][f'valid_{level}'], y >= 0)
        assert (y == -1).any()


def test_feature_leakage_sentinels(dataset):
    identity = dataset.identities[0]
    record = dataset._reader_for_process().get(Record, identity.record_id, load=LoadPlan.ALL_EAGER)
    original = record_to_sample(record, identity, None)['X']
    record.metadata['TARGET_SENTINEL'] = 'CLASS_FILENAME_FINGERPRINT_DO_NOT_USE'
    record.metadata['kernelsight_v4']['arrays']['fingerprint']['kernel_names']['values'] = ['TARGET_SENTINEL']
    lab = record.metadata['kernelsight_v4']['arrays']['labels']
    lab['workload_l1']['values'] = [-1] * 512
    lab['workload_l2']['values'] = [-1] * 512
    lab['mask_labeled']['values'] = [0] * 512
    lab['vocab_l1']['values'] = ['TARGET_SENTINEL_' + x for x in lab['vocab_l1']['values']]
    torch.testing.assert_close(record_to_sample(record, identity, None)['X'], original, rtol=0, atol=0)
    target = Signal.from_values(np.zeros(512, dtype=np.float32),
                               spec=TimeSeriesSpec(spec_type='labels', name='TARGET_SENTINEL', unit_value=ureg.dimensionless),
                               name='TARGET_SENTINEL', time_axis=OrdinalAxis())
    target.metadata['channel_index'] = 0
    ordered = sorted(record.signals, key=lambda s: s.metadata['channel_index'])
    tainted = replace(record, sources=(Source(name='tainted', signals=(target, *ordered[1:])),))
    with pytest.raises(ValueError, match='input spec'):
        record_to_sample(tainted, identity, None)


def test_normalization_definition_and_zero_variance(dataset):
    item = dataset[0]
    stats = dataset.normalization
    expected = ((item['raw_X'].numpy().astype(np.float64) - np.asarray(stats.mean)[:, None]) /
                stats.scale[:, None]).astype(np.float32)
    assert np.array_equal(item['X'].numpy(), expected)
    assert stats.n_bins == 1124 * 512
    assert len(stats.train_inputs_sha256) == 1124
    zero = replace(stats, std=tuple([0.0] * 24))
    assert (zero.scale == 1).all()
    identity = dataset.identities[0]
    record = dataset._reader_for_process().get(Record, identity.record_id, load=LoadPlan.ALL_EAGER)
    z = record_to_sample(record, identity, zero)
    assert np.array_equal(z['X'].numpy(), (item['raw_X'].numpy().astype(np.float64) - np.asarray(stats.mean)[:, None]).astype(np.float32))
    bad = replace(stats, train_inputs_sha256={})
    with pytest.raises(ValueError, match='official training inputs'):
        TimeFTraceDataset('val', normalization=bad)


def test_pickle_drops_reader_and_preserves_order(dataset):
    expected = dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))
    try:
        assert restored._reader is None
        assert restored.identities == dataset.identities
        assert torch.equal(restored[0]['X'], expected['X'])
    finally:
        restored.close()


@pytest.mark.parametrize('device', ['cuda', 'mps'])
def test_optional_accelerator(dataset, device):
    available = torch.cuda.is_available() if device == 'cuda' else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f'{device} not available in this execution environment')
    x = collate_traces(dataset.__getitems__([0, 1]))['X'].to(device)
    assert x.shape == (2, 24, 512) and x.dtype == torch.float32


def test_validation_does_not_fit_normalization(dataset, monkeypatch):
    import kernel_xray.dataset as phase1

    def forbidden(*args, **kwargs):
        raise AssertionError('Preprocessing must never be fitted implicitly')

    monkeypatch.setattr(phase1, 'fit_training_statistics', forbidden)
    val = TimeFTraceDataset('val')
    try:
        assert val.normalization == dataset.normalization
        assert torch.isfinite(val[0]['X']).all()
    finally:
        val.close()


def test_mixed_full_release_rejected_before_array_decoding():
    with pytest.raises(ValueError, match='single-partition'):
        TimeFTraceDataset('val', version=FULL_VERSION)
