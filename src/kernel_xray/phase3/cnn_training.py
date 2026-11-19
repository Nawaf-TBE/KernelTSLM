"""Controlled CNN training on isolated TimeF views; no original/test reader access."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cnn import CNNConfig, L1TemporalCNN, masked_cross_entropy
from .cnn_metrics import classification_metrics, confusion_matrix, temporal_masks, common_confusions
from .timef_dataset import DEFAULT_ROOT, DEFAULT_VERSIONS, collate_traces, make_dataloaders


@contextmanager
def isolated_reads():
    """Fail before any TimeF Record/row-group outside the two approved views loads."""
    from timenet.reader import TimeFReader
    from timenet.storage.parquet import SignalRowGroupCache
    allowed = {path.resolve() for path in DEFAULT_VERSIONS.values()}
    original_init, original_values, original_np = TimeFReader.__init__, SignalRowGroupCache.read_values, np.load

    def initialize(reader, version, *args, **kwargs):
        if Path(version.root).resolve() not in allowed:
            raise AssertionError('Model development may only open isolated train/val TimeF views')
        return original_init(reader, version, *args, **kwargs)

    def values(cache, version, *args, **kwargs):
        if Path(version.root).resolve() not in allowed:
            raise AssertionError('Mixed/test row-group decode prohibited')
        return original_values(cache, version, *args, **kwargs)

    def np_load(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve().is_relative_to((DEFAULT_ROOT / 'kernels').resolve()):
            raise AssertionError('Training does not decode raw dataset NPZs')
        return original_np(path, *args, **kwargs)

    with patch.object(TimeFReader, '__init__', initialize), patch.object(SignalRowGroupCache, 'read_values', values), patch.object(np, 'load', np_load):
        yield


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


class CachedTraces(Dataset):
    """RAM cache of unchanged whole trace tensors from the existing TimeF pipeline."""
    def __init__(self, x, y, valid, metadata):
        self.x, self.y, self.valid, self.metadata = x, y, valid, metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return {'X': self.x[index], 'targets': {'workload_l1': self.y[index]},
                'masks': {'valid_l1': self.valid[index]}, 'metadata': self.metadata[index]}

    def subset(self, indices):
        return CachedTraces(self.x[indices], self.y[indices], self.valid[indices], [self.metadata[i] for i in indices])

    def fingerprint(self):
        digest = hashlib.sha256()
        for array in [self.x, self.y, self.valid]:
            digest.update(array.numpy().tobytes())
        digest.update(json.dumps([m['trace_id'] for m in self.metadata]).encode())
        return digest.hexdigest()


def load_caches(batch_size=16):
    loaders = make_dataloaders(batch_size=batch_size, shuffle_train=False, num_workers=0)
    result = {}
    try:
        for split, loader in loaders.items():
            x, y, valid, metadata = [], [], [], []
            for batch in loader:
                x.append(batch['X'])
                y.append(batch['targets']['workload_l1'])
                valid.append(batch['masks']['valid_l1'])
                metadata.extend(batch['metadata'])
            result[split] = CachedTraces(torch.cat(x), torch.cat(y), torch.cat(valid), metadata)
            if not torch.equal(result[split].valid, result[split].y >= 0):
                raise AssertionError('Cached mask alignment')
            print(f'Cached {split}: {len(metadata)} whole traces', flush=True)
    finally:
        for loader in loaders.values():
            loader.dataset.close()
    if {m['trace_id'] for m in result['train'].metadata} & {m['trace_id'] for m in result['val'].metadata}:
        raise AssertionError('Train/validation traces overlap')
    return result


def cache_loader(cache, batch_size, seed=7, shuffle=False):
    return DataLoader(cache, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=collate_traces, generator=torch.Generator().manual_seed(seed))


def device_batch(batch, device):
    # No metadata, masks or labels are appended to X. Only targets/masks move separately for loss.
    return {'X': batch['X'].to(device),
            'targets': {'workload_l1': batch['targets']['workload_l1'].to(device)},
            'masks': {'valid_l1': batch['masks']['valid_l1'].to(device)}}


def gradients_finite(model):
    return bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all().item())


def synchronize(device):
    if str(device) == 'mps':
        torch.mps.synchronize()


def evaluate(model, cache, device, batch_size, supported, names, predictions_path=None):
    """Full masked evaluation; diagnostic subsets never influence checkpoint selection."""
    model.eval()
    total_cm = np.zeros((12, 12), dtype=np.int64)
    diagnostic_cm = {name: total_cm.copy() for name in ['interior', 'boundary', 'multiclass_traces']}
    per_trace = []
    prediction_rows = {'trace_index': [], 'bin_index': [], 'true_class': [], 'predicted_class': [], 'probabilities': []}
    ce_sum = 0.0
    seen = 0
    with torch.no_grad():
        for batch in cache_loader(cache, batch_size):
            moved = device_batch(batch, device)
            logits = model(moved['X'])
            loss = masked_cross_entropy(logits, moved)
            n = int(batch['masks']['valid_l1'].sum())
            if loss is not None:
                ce_sum += float(loss.item()) * n
            probability = logits.softmax(1).permute(0, 2, 1).cpu().numpy()
            prediction = probability.argmax(-1)
            for i, meta in enumerate(batch['metadata']):
                truth = batch['targets']['workload_l1'][i].numpy()
                valid = batch['masks']['valid_l1'][i].numpy()
                cm = confusion_matrix(truth[valid], prediction[i, valid])
                total_cm += cm
                row = classification_metrics(cm, supported, names)
                present = np.flatnonzero(cm.sum(1))
                per_trace.append({'trace_id': meta['trace_id'], 'valid_bins': int(valid.sum()),
                                  'accuracy': row['accuracy'],
                                  'macro_f1_supported': row['macro_f1_supported'] if valid.any() else None,
                                  'macro_f1_present_truth': float(np.mean([row['per_class'][c]['f1'] for c in present])) if valid.any() else None})
                interior, boundary = temporal_masks(truth)
                for key, mask in [('interior', interior), ('boundary', boundary)]:
                    diagnostic_cm[key] += confusion_matrix(truth[mask], prediction[i, mask])
                if len(present) >= 2:
                    diagnostic_cm['multiclass_traces'] += cm
                if predictions_path is not None:
                    indices = np.flatnonzero(valid)
                    prediction_rows['trace_index'].extend([seen] * len(indices))
                    prediction_rows['bin_index'].extend(indices.tolist())
                    prediction_rows['true_class'].extend(truth[valid].tolist())
                    prediction_rows['predicted_class'].extend(prediction[i, valid].tolist())
                    prediction_rows['probabilities'].append(probability[i, valid])
                seen += 1
    result = classification_metrics(total_cm, supported, names)
    valid_total = result['valid_bins']
    result.update(masked_cross_entropy=ce_sum / valid_total if valid_total else None,
                  traces=seen, unknown_bins=len(cache) * 512 - valid_total,
                  valid_label_coverage=valid_total / (len(cache) * 512),
                  mean_per_trace_accuracy=float(np.mean([r['accuracy'] for r in per_trace if r['valid_bins']])),
                  mean_per_trace_macro_f1_supported=float(np.mean([r['macro_f1_supported'] for r in per_trace if r['valid_bins']])),
                  mean_per_trace_macro_f1_present_truth=float(np.mean([r['macro_f1_present_truth'] for r in per_trace if r['valid_bins']])),
                  traces_without_valid_labels=sum(not r['valid_bins'] for r in per_trace),
                  empty_trace_convention='excluded from per-trace means; contribute no confusion counts',
                  diagnostics={key: classification_metrics(cm, supported, names) for key, cm in diagnostic_cm.items()},
                  common_confusions=common_confusions(total_cm, names), per_trace=per_trace)
    if predictions_path is not None:
        arrays = {key: np.asarray(value, dtype=np.int32) for key, value in prediction_rows.items() if key != 'probabilities'}
        arrays['probabilities'] = np.concatenate(prediction_rows['probabilities']).astype(np.float32)
        arrays['trace_ids'] = np.asarray([r['trace_id'] for r in per_trace])
        if len(arrays['true_class']) != valid_total or (arrays['true_class'] < 0).any():
            raise AssertionError('Prediction output contains unknown/duplicate bins')
        np.savez_compressed(predictions_path, **arrays)
    return result


def train_experiment(kind, caches, config, device, weights, supported, names, output, provenance):
    """Identical seed/order/schedule for both predeclared weighting conditions."""
    output.mkdir(exist_ok=False)
    seed_everything(config['seed'], config['cpu_threads'])
    model_config = CNNConfig(**{**config['model'], 'dilations': tuple(config['model']['dilations'])})
    model = L1TemporalCNN(model_config).to(device)
    initial_sha = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loader = cache_loader(caches['train'], config['batch_size'], seed=config['seed'], shuffle=True)
    class_weights = None if weights is None else weights.to(device)
    history, best, bad_epochs = [], -1.0, 0
    start = perf_counter()
    peak_mps = 0
    for epoch in range(1, config['maximum_epochs'] + 1):
        model.train()
        cm = np.zeros((12, 12), dtype=np.int64)
        loss_sum = unweighted_sum = valid_bins = skipped = 0
        order = hashlib.sha256()
        epoch_start = perf_counter()
        for batch_index, batch in enumerate(loader):
            order.update(json.dumps([m['trace_id'] for m in batch['metadata']]).encode())
            moved = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(moved['X'])  # sole model input
            loss = masked_cross_entropy(logits, moved, class_weights)
            n = int(batch['masks']['valid_l1'].sum())
            if loss is None:
                skipped += 1
                continue
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            if batch_index == 0 and not gradients_finite(model):
                raise RuntimeError('Nonfinite gradients')
            optimizer.step()
            with torch.no_grad():
                base_loss = masked_cross_entropy(logits, moved)
                mask = batch['masks']['valid_l1'].numpy()
                prediction = logits.detach().argmax(1).cpu().numpy()
                cm += confusion_matrix(batch['targets']['workload_l1'].numpy()[mask], prediction[mask])
            loss_sum += loss.item() * n
            unweighted_sum += base_loss.item() * n
            valid_bins += n
            if str(device) == 'mps':
                peak_mps = max(peak_mps, torch.mps.driver_allocated_memory())
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError('Nonfinite model parameters')
        val = evaluate(model, caches['val'], device, config['batch_size'], supported, names)
        row = {'epoch': epoch, 'train_objective': loss_sum / valid_bins,
               'train_unweighted_ce': unweighted_sum / valid_bins,
               'train_online_macro_f1': classification_metrics(cm, supported, names)['macro_f1_supported'],
               'val_ce': val['masked_cross_entropy'], 'val_macro_f1': val['macro_f1_supported'],
               'val_accuracy': val['accuracy'], 'training_valid_bins': valid_bins,
               'zero_label_batches_skipped': skipped, 'trace_order_sha256': order.hexdigest(),
               'seconds': perf_counter() - epoch_start}
        history.append(row)
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        print(f'{kind} epoch {epoch}: val macro11={row["val_macro_f1"]:.5f} accuracy={row["val_accuracy"]:.5f} ({row["seconds"]:.1f}s)', flush=True)
        if row['val_macro_f1'] > best:
            best, bad_epochs = row['val_macro_f1'], 0
            torch.save({'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'model_config': asdict(model_config), 'epoch': epoch, 'selection_macro_f1': best,
                        'training_config': config, 'class_weights': None if weights is None else weights.tolist(),
                        'provenance': provenance}, output / 'best.pt')
        else:
            bad_epochs += 1
        if bad_epochs >= config['patience']:
            break
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    val = evaluate(model, caches['val'], device, config['batch_size'], supported, names, output / 'validation_predictions.npz')
    training = evaluate(model, caches['train'], device, config['batch_size'], supported, names)
    result = {'kind': kind, 'best_epoch': checkpoint['epoch'], 'epochs_executed': len(history),
              'parameter_count': model.parameter_count, 'receptive_field_bins': model_config.receptive_field,
              'initial_parameters_sha256': initial_sha, 'validation': val, 'best_checkpoint_training': training,
              'training_seconds_including_final_evaluations': perf_counter() - start,
              'sampled_mps_driver_peak_bytes': peak_mps, 'device': str(device),
              'early_stopped': len(history) < config['maximum_epochs']}
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
