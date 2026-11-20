"""Seeded, bounded candidate sampling with no operation-label inputs."""
from collections import Counter
import hashlib
import json
import numpy as np
from .oracle import TYPES, Rejected, oracle, question, validate


def generate_trace(x, names, split, trace_id, config, normalization):
    if split not in ('train','val'):
        raise ValueError('QA v1 permits train/val only')
    validate(x,names,'statistics',[0],0,512)
    # Per-trace seed is independent of traversal/process order, and never sent to a model.
    seed = int.from_bytes(hashlib.sha256(f"{config['seed']}:{split}:{trace_id}".encode()).digest()[:8],'big')
    rng = np.random.default_rng(seed)
    examples, rejected, shortfalls = [], Counter(), {}
    for kind in TYPES:
        pool, seen = [], set()
        for _ in range(config['attempts_per_type']):
            lo, hi = config['length_bands'][int(rng.integers(3))]
            length = int(rng.integers(lo,hi+1));start=int(rng.integers(513-length));end=start+length
            channels = rng.choice(24, size=1 if kind=='statistics' else 2, replace=False).tolist()
            key=(kind,tuple(channels),start,end)
            if key in seen:
                rejected['duplicate_candidate']+=1;continue
            seen.add(key)
            try:
                answer, operands = oracle(x,names,kind,channels,start,end,config)
                if kind=='comparison' and answer=='Tie':
                    rejected['comparison_exact_tie']+=1;continue
            except Rejected as error:
                rejected[str(error)]+=1;continue
            pool.append((channels,start,end,answer,operands))
            # Fixed candidate budget for categorical balancing; stats need only five.
            if kind=='statistics' and len(pool)==config['per_type']:break
        counts=Counter()
        for _ in range(min(config['per_type'],len(pool))):
            # Choose least represented available category, random tie choice; never alter an answer.
            minimum=min(counts[str(row[3])] for row in pool)
            choices=[i for i,row in enumerate(pool) if counts[str(row[3])]==minimum]
            index=int(rng.choice(choices));channels,start,end,answer,operands=pool.pop(index)
            counts[str(answer)]+=1
            params={'type':kind,'channels':channels,'start':start,'end':end}
            qid=hashlib.sha256(json.dumps([config['version'],split,trace_id,params],sort_keys=True).encode()).hexdigest()
            examples.append({'id':qid,'split':split,'trace_id':trace_id,**params,
                'counter_names':[names[c] for c in channels],
                'question':question(kind,names,channels,start,end),'answer':answer,
                'answer_type':'number' if kind=='statistics' else 'category',
                'oracle':{'version':config['version'],'precision':'float32 inputs; float64 arithmetic', 'operands':operands},
                'normalization':normalization})
        if sum(row['type']==kind for row in examples)<config['per_type']:
            shortfalls[kind]={'missing':config['per_type']-sum(row['type']==kind for row in examples),
                             'reason':'bounded candidate pool exhausted; invalid/ambiguous candidates not forced'}
    return examples,dict(rejected),shortfalls


def model_input(example, x, names):
    """Strict output allowlist: no IDs, splits, operands, answers, masks, or labels."""
    validate(x,names,example['type'],example['channels'],example['start'],example['end'])
    if example['counter_names'] != [names[c] for c in example['channels']]:
        raise ValueError('Channel mapping mismatch')
    if example['question']!=question(example['type'],names,example['channels'],example['start'],example['end']):
        raise ValueError('Question text mismatch')
    return {'question':example['question'],'time_series':x.tolist(),'counter_names':list(names)}
