"""Finalize the independent audit after CPU replay and regression checks.

Reads immutable artifacts and hashes protected bytes; never deserializes test data.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/phase3/cnn_audit'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(name):
    return json.loads((OUT / name).read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert read('verification.json')['status'] in {
        'core_checks_passed_pending_regression_and_interpretation', 'passed_with_qualifications'
    }, 'Independent analysis must succeed before finalization'
    before = read('preservation_before.json')
    changed, missing = [], []
    for relative, expected in before.items():
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
        elif digest(path) != expected:
            changed.append(relative)
    preservation = {'status': 'failed' if changed or missing else 'passed',
                    'existing_files_verified': len(before), 'changed': changed, 'missing': missing,
                    'method': 'SHA256 bytes against pre-audit inventory; test files only hashed, never decoded',
                    'inventory': 'preservation_before.json'}
    save('preservation_after.json', preservation)
    assert not changed and not missing, 'Historical artifact preservation failed'

    tree = ET.parse(OUT / 'regression.xml')
    suites = list(tree.getroot().iter('testsuite'))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ['tests', 'failures', 'errors', 'skipped']}
    counts['passed'] = counts['tests'] - counts['failures'] - counts['errors'] - counts['skipped']
    assert counts['passed'] == 61 and counts['skipped'] == 1
    assert counts['failures'] == counts['errors'] == 0
    legacy = read('legacy_verification/artifact_verification.json')
    assert legacy['status'] == 'passed'
    for name in ['source_integrity', 'metrics_recalculated', 'data_isolation',
                 'training_control_checks', 'training_synthetic_checks', 'temporal_diagnostics']:
        assert read(name + '.json')['status'] == 'passed', name
    for condition, replay in read('cpu_replay.json').items():
        assert replay['status'] == 'passed' and replay['class_disagreements'] == 0, condition

    review = {
        'status': 'passed_for_frozen_configuration',
        'counter_only_forward': ['src/kernel_xray/phase3/cnn_training.py:103',
                                 'src/kernel_xray/phase3/cnn_training.py:136',
                                 'src/kernel_xray/phase3/cnn_training.py:212'],
        'loss': {'reference': 'src/kernel_xray/phase3/cnn.py:76',
                 'definition': 'sum valid-bin cross-entropies, optionally class-weighted, divided by number of valid bins',
                 'unknown_contribution': 0, 'empty_batch': 'None; skip backward and optimizer step',
                 'synthetic_evidence': 'training_synthetic_checks.json'},
        'weighting': {'reference': 'src/kernel_xray/phase3/cnn.py:86',
                      'source': 'scripts/run_cnn_baselines.py:156; training cache labels only',
                      'definition': 'min(4,sqrt(max count/count)) on positive classes; normalize training-bin expectation to one',
                      'zero_support_other_weight': 0, 'other_softmax_output_retained': True,
                      'independent_counts_and_weights': 'data_isolation.json'},
        'optimizer': {'reference': 'src/kernel_xray/phase3/cnn_training.py:196',
                      'implementation': 'AdamW; zero_grad -> valid loss -> backward -> step',
                      'evidence': 'synthetic manual loss/gradient and first AdamW update match; no CNN optimization during audit'},
        'validation_state': {'references': ['src/kernel_xray/phase3/cnn_training.py:126',
                                           'src/kernel_xray/phase3/cnn_training.py:133'],
                             'observed': 'eval/no_grad, parameters and buffers unchanged; no gradients'},
        'selection': {'reference': 'src/kernel_xray/phase3/cnn_training.py:247',
                      'rule': 'strict improvement of fixed training-supported macro11; earliest tie retained; patience5'},
        'normalization': {'fit': 'all official training bins only',
                          'independently_recomputed': True, 'evidence': 'data_isolation.json'},
        'limits': ['Historical nonbest epoch checkpoints were not retained; history consistency is verified, not a retraining replay.',
                   'MPS training emitted a nondeterministic-operation warning; deterministic fixed-checkpoint CPU replay does not prove deterministic training.'],
        'nonblocking_robustness_finding': {
            'description': 'optimizer and cache_traces_in_ram configuration values are descriptive; runner always uses AdamW and RAM caching without rejecting unsupported values',
            'impact_on_actual_run': 'none: frozen config specifies AdamW and true',
            'proposed_fix': 'proposed_config_guard.patch', 'applied': False,
            'suggested_future_tests': 'Reject optimizer=SGD and cache_traces_in_ram=false before any data reads; accept frozen configuration.'}}
    save('training_implementation_review.json', review)

    paths = ['docs/phase3_cnn_independent_audit.md', 'src/kernel_xray/phase3/independent_audit.py',
             'tests/test_independent_cnn_audit.py', 'scripts/audit_cnn_baseline.py',
             'scripts/regress_cnn_audit.py', 'scripts/finalize_cnn_audit.py']
    paths += [str(p.relative_to(ROOT)) for p in OUT.rglob('*')
              if p.is_file() and p.suffix in {'.json', '.xml', '.csv', '.patch'}
              and p.name not in {'verification.json', 'audit_evidence_hashes.json'}]
    save('audit_evidence_hashes.json', {name: digest(ROOT / name) for name in sorted(set(paths))})
    verification = {
        'status': 'passed_with_qualifications',
        'verdict': 'VERIFIED WITH QUALIFICATIONS',
        'meaning': 'Results are numerically correct but there are material generalization or reproducibility limitations.',
        'confirmed_technical_correctness': 'passed: artifact identity, independent confusion arithmetic, CPU frozen replay, training implementation, strict split isolation',
        'reproducibility': 'fixed checkpoint replay passed; seeded initialization and all 18 shuffle orders reproduced; MPS training nondeterminism and single seed qualify end-to-end reproduction',
        'workload_generalization': 'qualified: all 160 validation traces share collector families with training; only two traces contain any individually unseen compiled kernel symbol',
        'scientific_interpretation': 'validation primary L1 recognition only; not unseen-workload/hardware, overlap detection or counter-grounded explanation evidence',
        'hard_failures': [],
        'nonblocking_findings': ['Future unsupported optimizer/cache configuration values are not rejected; separate proposed patch is not applied.'],
        'evidence': {key: value for key, value in [
            ('protocol', 'protocol.json'), ('integrity', 'source_integrity.json'),
            ('metrics', 'metrics_recalculated.json'), ('inference', 'cpu_replay.json'),
            ('data_isolation', 'data_isolation.json'), ('training_review', 'training_implementation_review.json'),
            ('loss_optimizer', 'training_synthetic_checks.json'), ('training_controls', 'training_control_checks.json'),
            ('workloads', 'workload_overlap_analysis.json'), ('temporal', 'temporal_diagnostics.json'),
            ('uncertainty', 'uncertainty.json'), ('preservation', 'preservation_after.json'),
            ('regression', 'regression.xml'), ('historical_verifier', 'legacy_verification/artifact_verification.json'),
            ('evidence_hashes', 'audit_evidence_hashes.json')]},
        'regression': counts,
        'validation_traces': 160, 'valid_validation_bins': 33416,
        'existing_files_unchanged': len(before),
        'retrained': False, 'hyperparameters_tuned': False, 'test_evaluated': False,
        'test_arrays_decoded': False, 'frozen_changed': False,
        'historical_verifier_outputs_redirected': True,
        'team_summary': 'Share confirmed official-validation scores and fixed CPU replay with trace-cluster intervals and workload-overlap qualifications; novel workload/hardware generalization remains preliminary.'}
    save('verification.json', verification)
    print(json.dumps({'verdict': verification['verdict'], 'regression': counts,
                      'existing_files_unchanged': len(before)}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save('verification.json', {'status': 'failed', 'error_type': type(error).__name__,
                                   'error': str(error), 'stage': 'finalization'})
        raise
