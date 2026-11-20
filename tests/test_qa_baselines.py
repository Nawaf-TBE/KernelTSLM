"""Synthetic correctness, leakage sentinels, and grouped uncertainty tests."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from kernel_xray.dataset import DEFAULT_ROOT
from kernel_xray.qa_baselines import core,audit


@pytest.fixture
def signals():return np.random.default_rng(9).normal(size=(24,512)).astype(np.float32)
@pytest.fixture
def p():return {'type':'temporal','channels':[2,7],'start':11,'end':200}
@pytest.fixture
def cfg():return json.loads((DEFAULT_ROOT/'configs/phase4_qa_v1.json').read_text())


@pytest.mark.parametrize('aware',[False,True])
def test_features_independent_and_safe(signals,p,aware):
    actual=core.features(signals,p,aware)
    np.testing.assert_allclose(actual,audit.independent_features(signals,p,aware),rtol=1e-12,atol=1e-12)
    assert len(actual)==(70 if aware else 61)
    row={**p,'answer':'secret','oracle':{'maxima':[1e9,2e9]},'trace_id':'leak','split':'test','labels':[99]}
    assert np.array_equal(actual,core.features(signals,core.parameters(row),aware))
    row['answer']='changed';row['oracle']['maxima']=[-1e9,-1e9]
    assert np.array_equal(actual,core.features(signals,core.parameters(row),aware))
    with pytest.raises(ValueError):core.features(signals,row,aware)


def test_feature_boundaries(signals,p):
    original=core.features(signals,p)
    altered=signals.copy();altered[2,:11]+=100;altered[7,200:]+=100;altered[0]+=100
    assert np.array_equal(original,core.features(altered,p))
    for field,value in [('channels',[2,2]),('start',200),('end',513),('type','unknown')]:
        bad={**p,field:value}
        with pytest.raises(ValueError):core.features(signals,bad)


@pytest.mark.parametrize('kind',['comparison','temporal','statistics'])
def test_direct_independent(signals,p,cfg,kind):
    p['type']=kind
    if kind=='statistics':p['channels']=[2]
    assert core.direct(signals,p,cfg)==audit.independent_direct(signals,p,cfg)
    altered=core.reverse_interval(signals,p)
    if kind=='comparison':assert core.direct(altered,p,cfg)==core.direct(signals,p,cfg)
    elif kind=='temporal':
        assert core.direct(altered,p,cfg)=={'Yes':'No','No':'Yes','Same bin':'Same bin'}[core.direct(signals,p,cfg)]
    else:assert np.isclose(core.direct(altered,p,cfg),core.direct(signals,p,cfg))
    assert np.array_equal(signals,core.reverse_interval(altered,p))


@pytest.mark.parametrize('kind',['comparison','temporal'])
def test_ambiguity(signals,p,cfg,kind):
    p['type']=kind;signals[2,11:200]=1;signals[7,11:200]=1
    if kind=='comparison':assert core.direct(signals,p,cfg)=='Tie'
    else:
        with pytest.raises(ValueError):core.direct(signals,p,cfg)
    signals[7,11:200]=1.000001
    with pytest.raises(ValueError):core.direct(signals,p,cfg)


def test_rules_train_only_and_ties():
    rows=[]
    for kind in ('comparison','temporal'):
        for answer,pair in [('B',[0,1]),('A',[0,1]),('A',[1,0])]:rows.append({'type':kind,'channels':pair,'answer':answer})
    rows.extend({'type':'statistics','answer':v} for v in [1,2,7,8])
    fit=core.fit_rules(rows)
    assert fit['constant']==4.5 and fit['majority']['comparison']=='A'
    assert core.rule_predict(fit,'comparison',[0,1],True)=='B'
    assert core.rule_predict(fit,'comparison',[23,22],True)=='A'
    from scripts.run_qa_baselines import independent_rules
    assert fit==independent_rules(rows)


@pytest.mark.parametrize('aware',[False,True])
def test_fit_scaler_and_replay(signals,p,aware):
    matrix=[]
    for c in range(12):
        q={**p,'channels':[c,c+12]};matrix.append(core.features(signals,q,aware))
    matrix=np.array(matrix);answers=['A','B']*6
    cfg={'C':1,'solver':'lbfgs','max_iter':3000,'tol':1e-8,'random_state':7}
    model=core.fit_classifier(matrix,answers,cfg)
    assert model['scaler'].n_samples_seen_==12
    original_mean=model['scaler'].mean_.copy()
    for identity in (False,True):
        assert np.array_equal(core.predict_classifier(model,matrix,identity),audit.linear_replay(model,matrix,identity))
    core.predict_classifier(model,matrix*100)
    assert np.array_equal(model['scaler'].mean_,original_mean)
    repeat=core.fit_classifier(matrix,answers,cfg)
    assert np.array_equal(model['model'].coef_,repeat['model'].coef_)


@pytest.mark.parametrize('truth,pred', [(['A','A','B'],['A','B','B']),(['A','B','C'],['A','A','A'])])
def test_independent_categorical(truth,pred):
    audit.compare(core.categorical_metrics(truth,pred,sorted(set(truth))),audit.categorical(truth,pred,sorted(set(truth))))


def test_independent_numerical(cfg):
    truth=[0,1,2,100];pred=[0,1.000001,1,99]
    audit.compare(core.numerical_metrics(truth,pred,cfg),audit.numerical(truth,pred,cfg))


def test_paired_whole_trace_bootstrap(cfg):
    predictions={}
    for kind in ('comparison','temporal'):
        for model in ('majority','lookup','primary','aware','direct'):
            for condition in ('original','reversed_original'):
                predictions[f'{kind}/{model}/{condition}']=[{'trace_id':t,'reference':y,'prediction':y} for t,y in [('t0','A'),('t0','B'),('t1','B')]]
    for model in ('majority','direct'):
        predictions[f'statistics/{model}/original']=[{'trace_id':t,'reference':v,'prediction':v} for t,v in [('t0',1.),('t0',3.),('t1',5.)]]
    protocol={'seed':7,'replicates':100}
    a=audit.bootstrap(predictions,['t0','t1'],protocol,cfg);b=audit.bootstrap(predictions,['t0','t1'],protocol,cfg)
    assert a==b and a['units']==2
    assert a['metrics']['comparison/direct/original']['percentile95']['accuracy']==[1.,1.]
    assert a['metrics']['statistics/direct/original']['percentile95']['mae']==[0.,0.]
    assert all(v==[0.,0.] for diff in a['paired_differences_percentile95'].values() for v in diff.values())


def test_test_split_rejected():
    from scripts.run_qa_baselines import rows
    with pytest.raises(ValueError):rows('test')
