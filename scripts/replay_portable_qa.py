"""Read-only portable-model replay on approved isolated validation inputs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from kernel_xray.phase3.timef_dataset import TimeFTraceDataset
from kernel_xray.phase3.cnn_training import isolated_reads
from kernel_xray.qa_baselines.audit import independent_features
from kernel_xray.qa_baselines.portable import PortableModel


def main():
    qa=ROOT/'outputs/phase4/qa_v1/val.jsonl'
    import hashlib
    assert hashlib.sha256(qa.read_bytes()).hexdigest()=='e873d36c89f86619bb29c98b19971e9b574490f48f9f37e94ad8b66c6adcf0df'
    bytrace={}
    for line in qa.read_text().splitlines():
        row=json.loads(line);bytrace.setdefault(row['trace_id'],[]).append(row)
    models={(kind,variant):PortableModel.load(ROOT/f'models/qa_v1/{kind}_{variant}.json') for kind in ('comparison','temporal') for variant in ('primary','aware')}
    predictions=json.loads((ROOT/'results/qa_v1/predictions.json').read_text());expected={key:{r['id']:r['prediction'] for r in rows} for key,rows in predictions.items()}
    dataset=TimeFTraceDataset('val');count=0
    try:
        for start in range(0,len(dataset),16):
            for sample in dataset.__getitems__(list(range(start,min(start+16,len(dataset))))):
                rows=bytrace.pop(sample['metadata']['trace_id']);x=sample['X'].numpy()
                for row in rows:
                    if row['type']=='statistics':continue
                    p={k:row[k] for k in ('type','channels','start','end')}
                    for variant in ('primary','aware'):
                        answer=models[(row['type'],variant)].predict(independent_features(x,p,variant=='aware')[None,:])[0]
                        assert answer==expected[f"{row['type']}/{variant}/original"][row['id']]
                        count+=1
        assert not bytrace
    finally:dataset.close()
    print(json.dumps({'status':'passed','portable_predictions_checked':count,'validation_traces':160}))


if __name__=='__main__':
    with isolated_reads():main()
