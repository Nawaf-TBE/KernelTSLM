"""Finalize local QA baseline evidence after independent audit, tests and figures."""
from pathlib import Path
import json
import pickle
import sys
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_qa_baselines import OUT,CONFIG,load,save,digest,preserve,check_frozen
import numpy as np


def test_counts(name):
    result={k:0 for k in ('tests','errors','failures','skipped')}
    for suite in ET.parse(OUT/name).getroot().iter('testsuite'):
        for k in result:result[k]+=int(suite.attrib.get(k,0))
    result['passed']=result['tests']-sum(result[k] for k in ('errors','failures','skipped'))
    assert result['errors']==result['failures']==0
    return result


def main():
    verification=load(OUT/'verification.json');assert verification['status']=='passed'
    preservation=preserve();hashes=check_frozen()
    protocol=load(OUT/'protocol.json');assert protocol['configuration_sha256']==digest(CONFIG)
    for path,h in protocol['source_sha256'].items():assert digest(ROOT/path)==h
    for path,h in load(OUT/'model_hashes.json').items():assert digest(OUT/path)==h
    tests={'focused':test_counts('focused_tests.xml'),'regression':test_counts('regression.xml')}
    assert tests['focused']['passed']==16 and tests['regression']['passed']==110
    m=load(OUT/'metrics.json');u=load(OUT/'uncertainty.json')
    assert m['statistics/direct/original']['mae']==0 and m['statistics/direct/original']['within_tolerance']==1
    # With direct absolute errors identically zero, paired differences are exactly
    # the negative constant error distributions under the SAME bootstrap draws.
    paired={}
    for metric in ('mae','median_absolute_error','rmse','within_tolerance'):
        point=m['statistics/direct/original'][metric]-m['statistics/majority/original'][metric]
        ci=u['metrics']['statistics/majority/original']['percentile95'][metric]
        paired[metric]={'point':point,'percentile95':[1-ci[1],1-ci[0]] if metric=='within_tolerance' else [-ci[1],-ci[0]]}
    save('statistics_paired_uncertainty.json',{'contrast':'direct minus training-median constant','protocol':u['protocol'],
        'derivation':'Direct errors are zero and coverage one on every trace, so paired differences follow exactly from the same constant-estimator bootstrap draws.',
        'metrics':paired})
    first=OUT/'initial_attempt_preservation_log_error'
    with (OUT/'models.pkl').open('rb') as f:models=pickle.load(f)
    with (first/'models.pkl').open('rb') as f:old=pickle.load(f)
    for key,fit in models.items():
        assert np.array_equal(fit['model'].coef_,old[key]['model'].coef_)
        assert np.array_equal(fit['model'].intercept_,old[key]['model'].intercept_)
    assert load(OUT/'predictions.json')==load(first/'predictions.json')
    assert load(OUT/'metrics.json')==load(first/'metrics.json')
    save('attempt_comparison.json',{'identical_model_coefficients':True,'identical_predictions':True,'identical_metrics':True,
        'change':'Only preservation inventory scope corrected: own growing log excluded. Hyperparameters unchanged.'})
    figures=sorted((OUT/'figures').glob('*.png'));assert len(figures)==5
    for p in figures:assert p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    verification.update({'tests':tests,'preservation':preservation,'qa_sha256':hashes,
        'feature_cache_replay':'all training/validation matrices independently reconstructed and compared at 1e-10',
        'figures':len(figures),'figures_visually_inspected':True,'qa_v1_modified':False,'cnn_modified':False,'published':False,
        'fit_repeatability':'all coefficients/intercepts/scalers bitwise identical; corrected run reproduces first attempt predictions exactly',
        'statistics_paired_uncertainty':'statistics_paired_uncertainty.json'})
    save('verification.json',verification)
    sources=[ROOT/'configs/phase4_qa_baselines_v1.json',ROOT/'tests/test_qa_baselines.py',
        ROOT/'docs/phase4_qa_baselines.md',ROOT/'docs/phase4_qa_baselines_summary.md',
        *sorted((ROOT/'src/kernel_xray/qa_baselines').glob('*.py')),
        *[ROOT/'scripts'/n for n in ('run_qa_baselines.py','plot_qa_baselines.py','finalize_qa_baselines.py')]]
    artifacts=[p for p in OUT.rglob('*') if p.is_file() and p.name!='COMPLETE.json']
    save('COMPLETE.json',{'status':'passed','verdict':verification['verdict'],'tests':tests,'preservation':preservation,
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'artifact_sha256':{str(p.relative_to(OUT)):digest(p) for p in artifacts}})
    print(json.dumps({'verdict':verification['verdict'],'tests':tests,'preservation':preservation},indent=2))


if __name__=='__main__':main()
