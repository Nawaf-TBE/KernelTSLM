"""Independent synthetic QA checks; no held-out data or model training."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from kernel_xray.qa.oracle import oracle, Rejected
from kernel_xray.qa.generation import generate_trace, model_input
from kernel_xray.qa.verification import verify_example
from kernel_xray.dataset import DEFAULT_ROOT


@pytest.fixture
def cfg():
    return json.loads((DEFAULT_ROOT/'configs/phase4_qa_v1.json').read_text())


@pytest.fixture
def counters():
    return np.random.default_rng(77).normal(size=(24,512)).astype(np.float32),[f'counter_{i}' for i in range(24)]


def test_mean_half_open(counters,cfg):
    x,n=counters;x[0,:4]=[1,3,7,100]
    answer,values=oracle(x,n,'statistics',[0],0,3,cfg)
    assert answer==11/3 and values=={'sum':11,'count':3}


@pytest.mark.parametrize('delta,expected',[(0,'Tie'),(.1,'Counter B'),(-.1,'Counter A')])
def test_comparison(counters,cfg,delta,expected):
    x,n=counters;x[0,:16]=1;x[1,:16]=1+delta
    assert oracle(x,n,'comparison',[0,1],0,16,cfg)[0]==expected


def test_near_tie(counters,cfg):
    x,n=counters;x[0,:16]=1;x[1,:16]=1.00001
    with pytest.raises(Rejected,match='near_tie'):oracle(x,n,'comparison',[0,1],0,16,cfg)


@pytest.mark.parametrize('b,answer',[(4,'Yes'),(1,'No'),(2,'Same bin')])
def test_temporal(counters,cfg,b,answer):
    x,n=counters;x[:2,:16]=0;x[0,2]=3;x[1,b]=4
    value,operands=oracle(x,n,'temporal',[0,1],0,16,cfg)
    assert value==answer and operands['peak_bins']==[2,b]


@pytest.mark.parametrize('almost',[False,True])
def test_ambiguous_peak(counters,cfg,almost):
    x,n=counters;x[:2,:16]=0;x[0,2:4]=[1,1-1e-7] if almost else [1,1];x[1,2]=2
    with pytest.raises(Rejected):oracle(x,n,'temporal',[0,1],0,16,cfg)


@pytest.mark.parametrize('channels,s,e',[([24],0,5),([-1],0,5),([True],0,5),([0],-1,5),([0],0,513),([0],4,4),([0],5,4),([0],0.,5)])
def test_invalid(counters,cfg,channels,s,e):
    x,n=counters
    with pytest.raises(ValueError):oracle(x,n,'statistics',channels,s,e,cfg)


@pytest.mark.parametrize('bad',['shape','dtype','nan','names','pair'])
def test_malformed(counters,cfg,bad):
    x,n=counters;kind='statistics';cs=[0]
    if bad=='shape':x=x[:23]
    if bad=='dtype':x=x.astype(np.float64)
    if bad=='nan':x[0,0]=np.nan
    if bad=='names':n[1]=n[0]
    if bad=='pair':kind='comparison';cs=[0,0]
    with pytest.raises(ValueError):oracle(x,n,kind,cs,0,16,cfg)


def test_generation_verification_and_serializer(counters,cfg):
    x,n=counters;cfg['attempts_per_type']=40
    args=(x,n,'train','synthetic',cfg,{'sha256':'synthetic'})
    rows,reject,short=generate_trace(*args)
    assert len(rows)==15 and not short
    assert (rows,reject,short)==generate_trace(*args)
    assert len({r['question'] for r in rows})==15
    for row in rows:
        assert verify_example(row,x,n,'train','synthetic',cfg,{'sha256':'synthetic'})
        before=model_input(row,x,n)
        changed=copy.deepcopy(row);changed['answer']='DO NOT LEAK';changed['oracle']={'secret':'never send'}
        assert model_input(changed,x,n)==before
        assert set(before)=={'question','time_series','counter_names'}
        assert np.array_equal(np.asarray(before['time_series'],np.float32),x)


@pytest.mark.parametrize('field,value',[('answer',12345),('start',512),('counter_names',['wrong']),('question','wrong'),('split','test'),('id','wrong'),('extra','secret')])
def test_independent_validator_rejects(counters,cfg,field,value):
    x,n=counters;cfg['attempts_per_type']=10
    row=generate_trace(x,n,'train','synthetic',cfg,{})[0][0]
    row[field]=value
    with pytest.raises(ValueError):verify_example(row,x,n,'train','synthetic',cfg,{})


def test_flat_shortfalls_and_split(counters,cfg):
    x,n=counters;x[:]=0;cfg['attempts_per_type']=20
    rows,reject,short=generate_trace(x,n,'train','flat',cfg,{})
    assert len(rows)==5 and short['comparison']['missing']==short['temporal']['missing']==5
    assert reject['comparison_exact_tie'] and reject['nonunique_or_ambiguous_peak']
    with pytest.raises(ValueError):generate_trace(x,n,'test','forbidden',cfg,{})
    from scripts.build_qa_v1 import authorized_dataset
    with pytest.raises(ValueError):authorized_dataset('test')


def test_single_bin_peak_rejected(counters,cfg):
    x,n=counters
    with pytest.raises(Rejected,match='insufficient_peak_samples'):
        oracle(x,n,'temporal',[0,1],0,1,cfg)
