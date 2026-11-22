"""Independent audit arithmetic checks; no fitting of any baseline."""
import numpy as np
import pytest
import torch
from kernel_xray.phase3.independent_audit import (
    count_confusion, metrics_from_counts, independent_temporal_masks,
    bootstrap_whole_traces, functional_frozen_forward,
)
from kernel_xray.phase3.cnn import L1TemporalCNN


def test_independent_counts_and_harmonic_metrics():
    cm=count_confusion(np.array([0,0,1,1]),np.array([0,1,1,1]))
    assert cm[0,0]==1 and cm[0,1]==1 and cm[1,1]==2
    m=metrics_from_counts(cm,list(range(11)),list(map(str,range(12))))
    assert m['accuracy']==m['micro_f1']==.75
    assert m['per_class'][0]['f1']==pytest.approx(2/3)
    assert m['macro_f1_supported']==pytest.approx((2/3+.8)/11)
    assert m['per_class'][11]['f1']==0
    with pytest.raises(ValueError):count_confusion(np.array([-1]),np.array([0]))


def test_window_diagnostics_unknown_breaks_and_boundary():
    y=np.array([0]*12+[1]*12+[-1]*3+[2]*9)
    interior,boundary=independent_temporal_masks(y)
    assert np.flatnonzero(interior).tolist()==list(range(4,8))+list(range(16,20))+[31]
    assert np.flatnonzero(boundary).tolist()==list(range(8,16))


def test_functional_checkpoint_eval_matches_module_and_has_local_support():
    torch.manual_seed(19);model=L1TemporalCNN().eval();x=torch.randn(1,24,512,requires_grad=True)
    independent=functional_frozen_forward(x,model.state_dict())
    torch.testing.assert_close(independent,model(x),atol=1e-6,rtol=1e-6)
    independent[0,0,256].backward();active=torch.nonzero(x.grad.abs().sum(1)[0]>1e-12).flatten()
    assert (active[0].item(),active[-1].item())==(228,284)


def test_bootstrap_resamples_whole_traces_and_keeps_absent_class_zero():
    traces=np.zeros((3,12,12),dtype=np.int64);traces[0,0,0]=10;traces[1,1,1]=4;traces[2,0,1]=5
    result=bootstrap_whole_traces({'unweighted':traces,'weighted':traces.copy()},list(range(11)),500,7)
    assert result['paired_weighted_minus_unweighted']['macro_f1_percentile95']==[0,0]
    assert result['unweighted']['missing_by_supported_class']['10']==500
    assert result['unweighted']['observed_class_trace_support']['1']==1
