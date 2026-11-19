"""Train, evaluate and freeze the two predeclared counter-only CNN conditions."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from kernel_xray.phase3.cnn import CNNConfig, L1TemporalCNN, bounded_class_weights, masked_cross_entropy
from kernel_xray.phase3.cnn_metrics import DIAGNOSTIC_PROTOCOL
from kernel_xray.phase3.cnn_training import (
    seed_everything, load_caches, cache_loader, device_batch, gradients_finite,
    isolated_reads, train_experiment, evaluate,
)
from kernel_xray.phase3.timef_dataset import DEFAULT_STATISTICS, DEFAULT_VERSIONS

OUT = ROOT / 'outputs/phase3/cnn'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def preserve_check():
    before = json.loads((OUT / 'preservation_before.json').read_text())
    changes = [name for name, expected in before.items() if sha(ROOT / name) != expected]
    if changes:
        raise AssertionError(f'Prior files changed: {changes}')
    result = {'status': 'passed', 'files_verified_unchanged': len(before), 'changes': []}
    (OUT / 'preservation_after.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def model_config(config):
    return CNNConfig(**{**config['model'], 'dilations': tuple(config['model']['dilations'])})


def choose_device(config, training):
    """Use MPS only after actual counter forward/backward and CPU-eval parity pass."""
    requested = config['device']
    report = {'requested': requested, 'mps_available': torch.backends.mps.is_available()}
    device = 'mps' if requested != 'cpu' and torch.backends.mps.is_available() else 'cpu'
    if requested == 'mps' and device != 'mps':
        raise RuntimeError('MPS explicitly requested but unavailable')
    batch = next(iter(cache_loader(training, config['batch_size'])))
    seed_everything(config['seed'], config['cpu_threads'])
    cpu = L1TemporalCNN(model_config(config)).eval()
    with torch.no_grad():
        expected = cpu(batch['X'])
    try:
        model = L1TemporalCNN(model_config(config)).to(device)
        model.load_state_dict(cpu.state_dict())
        model.eval()
        moved = device_batch(batch, device)
        with torch.no_grad():
            actual = model(moved['X']).cpu()
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = masked_cross_entropy(model(moved['X']), moved)
            loss.backward()
            if not torch.isfinite(loss).item() or not gradients_finite(model):
                raise RuntimeError('Nonfinite device sanity')
            optimizer.step()
        report.update(selected=device, stability_steps=10, cpu_parity='passed', max_abs_difference=float((actual-expected).abs().max()))
    except Exception as error:
        if device != 'mps' or requested == 'mps':
            raise
        report.update(selected='cpu', mps_failure=repr(error))
        device = 'cpu'
    return device, report


def overfit_sanity(config, training, device, supported, names):
    options = config['overfit_sanity']
    indices = [i for i in range(len(training)) if len(torch.unique(training.y[i][training.valid[i]])) >= 2
               and training.valid[i].sum() >= 32][:options['training_traces']]
    if len(indices) != options['training_traces']:
        raise AssertionError('Not enough training-only multiclass sanity traces')
    tiny = training.subset(indices)
    batch = device_batch(next(iter(cache_loader(tiny, len(tiny)))), device)
    seed_everything(config['seed'], config['cpu_threads'])
    model = L1TemporalCNN(model_config(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0)
    model.eval()
    with torch.no_grad():
        initial = masked_cross_entropy(model(batch['X']), batch).item()
    history = []
    for step in range(1, options['maximum_steps'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = masked_cross_entropy(model(batch['X']), batch)
        loss.backward()
        if not gradients_finite(model):
            raise RuntimeError('Nonfinite overfit gradient')
        optimizer.step()
        if step % 25 == 0 or step == options['maximum_steps']:
            metrics = evaluate(model, tiny, device, len(tiny), supported, names)
            row = {'step': step, 'loss': metrics['masked_cross_entropy'], 'accuracy': metrics['accuracy']}
            history.append(row)
            print('overfit', row, flush=True)
            if row['accuracy'] >= options['minimum_accuracy'] and row['loss'] <= initial * options['maximum_final_initial_loss_ratio']:
                break
    passed = history[-1]['accuracy'] >= options['minimum_accuracy'] and history[-1]['loss'] <= initial * options['maximum_final_initial_loss_ratio']
    result = {'status': 'passed' if passed else 'failed', 'source': 'training only',
              'trace_ids': [m['trace_id'] for m in tiny.metadata], 'initial_loss': initial,
              'history': history, 'criteria': options}
    (OUT / 'sanity' / 'overfit.json').write_text(json.dumps(result, indent=2) + '\n')
    if not passed:
        raise RuntimeError('Training-only subset overfit criterion failed; full experiments not started')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/phase3_cnn.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if any((OUT / name).exists() for name in ['unweighted', 'weighted', 'frozen', 'sanity']):
        raise RuntimeError('Use a fresh CNN output root; prior runs are preserved')
    (OUT / 'sanity').mkdir()
    protocol = {'training': config, 'diagnostics': DIAGNOSTIC_PROTOCOL,
                'weighting': 'r_c=min(4,sqrt(max_training_count/count_c)); zero for count0; rescale sum(count*w)/N=1',
                'selection_classes': 'all and only class IDs with positive training support, fixed before validation',
                'model_input': 'batch[X] only, full trace; no targets/masks/metadata/timestamps/features appended'}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    with (OUT / 'sanity/tests.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'pytest', str(ROOT / 'tests/test_cnn_baseline.py'), '-q',
                        '--junitxml=' + str(OUT / 'sanity/tests.xml')], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    preserve_check()
    seed_everything(config['seed'], config['cpu_threads'])
    start = perf_counter()
    with isolated_reads(), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        caches = load_caches(config['batch_size'])
        training = caches['train']
        counts = torch.bincount(training.y[training.valid], minlength=12)
        supported = torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()
        names = list(training.metadata[0]['vocab_l1'])
        if len(supported) != 11 or len(names) != 12 or int(counts.sum()) != 233821:
            raise AssertionError('Training support differs from verified phase 3 data')
        device, device_report = choose_device(config, training)
        (OUT / 'sanity/device.json').write_text(json.dumps(device_report, indent=2) + '\n')
        overfit_sanity(config, training, device, supported, names)
        preserve_check()
        provenance = {'source_files': {str(path.relative_to(ROOT)): sha(path) for path in [
            ROOT / 'src/kernel_xray/phase3/cnn.py', ROOT / 'src/kernel_xray/phase3/cnn_metrics.py',
            ROOT / 'src/kernel_xray/phase3/cnn_training.py', ROOT / 'scripts/run_cnn_baselines.py',
            args.config, ROOT / 'tests/test_cnn_baseline.py']},
            'normalization_sha256': sha(DEFAULT_STATISTICS),
            'view_manifest_sha256': {split: sha(path/'manifest.json') for split,path in DEFAULT_VERSIONS.items()},
            'cache_fingerprints': {split: cache.fingerprint() for split,cache in caches.items()},
            'source_revision': 'd8cd8e804e5b619aadbb082a1fb3a8791b60a680',
            'software': {name: version(name) for name in ['torch','numpy','timenet','pyarrow']},
            'python': platform.python_version(), 'platform': platform.platform(),
            'device': device_report, 'training_counts': counts.tolist(), 'supported_classes': supported,
            'names': names, 'config': config, 'parameter_count': L1TemporalCNN(model_config(config)).parameter_count,
            'receptive_field_bins': model_config(config).receptive_field,
            'cache_tensor_bytes': sum(c.x.numel()*c.x.element_size()+c.y.numel()*c.y.element_size()+c.valid.numel() for c in caches.values())}
        (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        weights = bounded_class_weights(counts, config['class_weight_maximum_ratio'])
        (OUT / 'class_weights.json').write_text(json.dumps({'training_counts':counts.tolist(),'weights':weights.tolist(),
            'ratio_cap':config['class_weight_maximum_ratio'],'expected_training_bin_weight':float((weights*counts).sum()/counts.sum())},indent=2)+'\n')
        results = {}
        for name, condition_weights in [('unweighted', None), ('weighted', weights)]:
            results[name] = train_experiment(name, caches, config, device, condition_weights, supported, names, OUT/name, provenance)
        if results['unweighted']['initial_parameters_sha256'] != results['weighted']['initial_parameters_sha256']:
            raise AssertionError('Experiments did not start identically')
        histories = {name: json.loads((OUT/name/'history.json').read_text()) for name in results}
        if any(a['trace_order_sha256'] != b['trace_order_sha256'] for a,b in zip(histories['unweighted'],histories['weighted'])):
            raise AssertionError('Experimental trace orders differ')
        winner = max(['unweighted','weighted'], key=lambda name: results[name]['validation']['macro_f1_supported'])
        frozen = OUT / 'frozen'; frozen.mkdir()
        shutil.copy2(OUT/winner/'best.pt', frozen/'baseline.pt')
        (frozen/'config.json').write_text(json.dumps(provenance,indent=2)+'\n')
        freeze = {'selected_experiment':winner,'selection_rule':DIAGNOSTIC_PROTOCOL['selection'],
                  'experiment_tie':'unweighted','checkpoint_sha256':sha(frozen/'baseline.pt'),
                  'configuration_sha256':sha(frozen/'config.json'),'best_epoch':results[winner]['best_epoch'],
                  'validation_macro_f1':results[winner]['validation']['macro_f1_supported'],
                  'normalization_sha256':sha(DEFAULT_STATISTICS),'test_evaluated':False}
        (frozen/'manifest.json').write_text(json.dumps(freeze,indent=2)+'\n')
        for path in frozen.iterdir(): path.chmod(0o444)
        summary = {'status':'trained_evaluated_frozen','selected':freeze,'experiments':results,
                   'weighted_minus_unweighted_macro_f1':results['weighted']['validation']['macro_f1_supported']-results['unweighted']['validation']['macro_f1_supported'],
                   'total_seconds':perf_counter()-start,'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   'warnings':sorted({str(w.message) for w in captured}), 'preservation':preserve_check(),
                   'test_evaluated':False,'original_mixed_artifact_reader_opened':False}
        (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('FROZEN',json.dumps(freeze),flush=True)


if __name__ == '__main__':
    main()
