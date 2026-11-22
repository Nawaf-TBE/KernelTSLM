"""Audit both trained checkpoints and replay the frozen model on isolated validation only."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from kernel_xray.phase3.cnn import CNNConfig, L1TemporalCNN
from kernel_xray.phase3.cnn_metrics import classification_metrics, confusion_matrix
from kernel_xray.phase3.cnn_training import CachedTraces, cache_loader, device_batch, isolated_reads
from kernel_xray.phase3.timef_dataset import TimeFTraceDataset
from torch.utils.data import DataLoader
from kernel_xray.phase3 import collate_traces

OUT=ROOT/'outputs/phase3/cnn'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    provenance=json.loads((OUT/'provenance.json').read_text())
    supported,names=provenance['supported_classes'],provenance['names']
    with isolated_reads():
        dataset=TimeFTraceDataset('val')
        x,y,mask,metadata=[],[],[],[]
        try:
            for batch in DataLoader(dataset,batch_size=16,collate_fn=collate_traces):
                x.append(batch['X']);y.append(batch['targets']['workload_l1']);mask.append(batch['masks']['valid_l1']);metadata.extend(batch['metadata'])
        finally:
            dataset.close()
        cache=CachedTraces(torch.cat(x),torch.cat(y),torch.cat(mask),metadata)
        assert cache.fingerprint()==provenance['cache_fingerprints']['val']
        paths=[m['trace_id'] for m in metadata]
        details={}
        for kind in ['unweighted','weighted']:
            folder=OUT/kind
            with np.load(folder/'validation_predictions.npz',allow_pickle=False) as arrays:
                rows={key:arrays[key].copy() for key in arrays.files}
            trace,bin_index=rows['trace_index'],rows['bin_index']
            assert rows['trace_ids'].tolist()==paths
            assert len(set(zip(trace.tolist(),bin_index.tolist())))==len(trace)==int(cache.valid.sum())
            assert np.array_equal(cache.y[trace,bin_index].numpy(),rows['true_class'])
            assert cache.valid[trace,bin_index].all()
            assert rows['probabilities'].shape==(len(trace),12) and np.isfinite(rows['probabilities']).all()
            np.testing.assert_allclose(rows['probabilities'].sum(1),1,atol=1e-6)
            assert np.array_equal(rows['probabilities'].argmax(1),rows['predicted_class'])
            cm=confusion_matrix(rows['true_class'],rows['predicted_class'])
            metrics=classification_metrics(cm,supported,names)
            saved=summary['experiments'][kind]['validation']
            assert metrics['macro_f1_supported']==saved['macro_f1_supported']
            assert metrics['accuracy']==saved['accuracy']
            history=json.loads((folder/'history.json').read_text())
            best=max(history,key=lambda r:r['val_macro_f1'])
            assert best['epoch']==summary['experiments'][kind]['best_epoch']
            ckpt=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
            assert ckpt['epoch']==best['epoch']
            assert all(torch.isfinite(value).all() for value in ckpt['model_state'].values())
            details[kind]={'valid_predictions':len(trace),'recomputed_metrics':'exact','selected_epoch':best['epoch']}
        frozen=OUT/'frozen';manifest=json.loads((frozen/'manifest.json').read_text())
        assert digest(frozen/'baseline.pt')==manifest['checkpoint_sha256']
        assert digest(frozen/'config.json')==manifest['configuration_sha256']
        chosen=manifest['selected_experiment']
        assert digest(frozen/'baseline.pt')==digest(OUT/chosen/'best.pt')
        ckpt=torch.load(frozen/'baseline.pt',map_location='cpu',weights_only=False)
        model=L1TemporalCNN(CNNConfig(**{**ckpt['model_config'],'dilations':tuple(ckpt['model_config']['dilations'])}))
        device=provenance['device']['selected'];model.load_state_dict(ckpt['model_state']);model.to(device).eval()
        probabilities=[]
        with torch.no_grad():
            for batch in cache_loader(cache,16):
                moved=device_batch(batch,device)
                p=model(moved['X']).softmax(1).permute(0,2,1).cpu().numpy()
                probabilities.append(p[batch['masks']['valid_l1'].numpy()])
        with np.load(OUT/chosen/'validation_predictions.npz',allow_pickle=False) as expected:
            np.testing.assert_allclose(np.concatenate(probabilities),expected['probabilities'],rtol=1e-5,atol=1e-6)
        result={'status':'passed','checkpoints':details,'frozen_checkpoint_replay':'all valid validation probabilities match',
                'parameter_count':model.parameter_count,'receptive_field_bins':model.config.receptive_field,
                'test_access':False,'original_mixed_reader_access':False}
        (OUT/'artifact_verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
