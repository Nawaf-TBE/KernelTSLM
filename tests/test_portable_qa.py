"""Published artifacts and safe numerical model interchange; no data access."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from kernel_xray.dataset import DEFAULT_ROOT
from kernel_xray.qa_baselines.portable import PortableModel,feature_names


@pytest.mark.parametrize('task,variant',[('comparison','primary'),('comparison','aware'),('temporal','primary'),('temporal','aware')])
def test_portable(task,variant):
    p=DEFAULT_ROOT/f'models/qa_v1/{task}_{variant}.json';m=PortableModel.load(p)
    x=np.random.default_rng(7).normal(size=(12,len(feature_names(variant))))
    probability=m.predict_proba(x)
    assert np.isfinite(probability).all() and np.allclose(probability.sum(1),1)
    assert m.predict(x).tolist()==[m.metadata['classes'][i] for i in probability.argmax(1)]
    for field in ('feature_names','coefficients'):
        bad=copy.deepcopy(m.metadata);bad[field]=[]
        with pytest.raises(ValueError):PortableModel(bad)
    bad=copy.deepcopy(m.metadata);bad['scaler']['scale'][0]=0
    with pytest.raises(ValueError):PortableModel(bad)
    with pytest.raises(ValueError):m.predict(np.full_like(x,np.nan))


def test_independent_published_metrics():
    from scripts.verify_published_qa import verify
    result=verify()
    assert result['status']=='passed' and result['models_validated']==4


def test_metric_corruption_detected():
    from scripts.verify_published_qa import equal
    with pytest.raises(AssertionError):equal({'accuracy':1.},{'accuracy':0.})


def test_binary_tiny_positive_score_uses_decision_not_rounded_probability():
    p=DEFAULT_ROOT/'models/qa_v1/comparison_primary.json';metadata=json.loads(p.read_text())
    metadata['coefficients']=[[0.]*61];metadata['intercepts']=[1e-17]
    model=PortableModel(metadata)
    assert model.predict(np.zeros((1,61)))[0]==metadata['classes'][1]


def test_artifact_verifier_without_site_packages():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-S',str(DEFAULT_ROOT/'scripts/verify_published_qa.py')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['status']=='passed'
