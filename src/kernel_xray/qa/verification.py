"""Independent scalar reference calculation. Does not import or call the oracle."""
import math
import hashlib
import json
import numpy as np


def verify_example(row, x, names, split, trace_id, config, normalization):
    expected={'id','split','trace_id','type','channels','start','end','counter_names','question','answer',
              'answer_type','oracle','normalization'}
    if set(row)!=expected or split not in ('train','val') or row['split']!=split or row['trace_id']!=trace_id:
        raise ValueError('Schema or authorized identity mismatch')
    if x.shape!=(24,512) or x.dtype!=np.float32 or not np.isfinite(x).all() or len(names)!=24 or len(set(names))!=24:
        raise ValueError('Counter schema')
    kind=row['type'];cs=row['channels'];s=row['start'];e=row['end']
    if kind not in ('statistics','comparison','temporal') or len(cs)!=(1 if kind=='statistics' else 2):
        raise ValueError('Question type')
    if any(type(c)!=int or c<0 or c>=24 for c in cs) or len(set(cs))!=len(cs):raise ValueError('Channels')
    if type(s)!=int or type(e)!=int or not 0<=s<e<=512:raise ValueError('Interval')
    if row['counter_names']!=[names[c] for c in cs] or row['normalization']!=normalization:raise ValueError('Mapping/provenance')
    a=names[cs[0]]
    if kind=='statistics':
        text=f'What is the average value of {a} between time bins {s} and {e}? Use [{s}, {e}) in normalized units.'
    elif kind=='comparison':
        text=f'Which counter has the higher maximum value between bins [{s}, {e}): {a} or {names[cs[1]]}? Values are normalized.'
    else:
        text=f'Does {a} reach its maximum before {names[cs[1]]} within bins [{s}, {e})? Values are normalized.'
    if row['question']!=text:raise ValueError('Malformed question text')
    params={k:row[k] for k in ['type','channels','start','end']}
    identity=hashlib.sha256(json.dumps([config['version'],split,trace_id,params],sort_keys=True).encode()).hexdigest()
    if row['id']!=identity:raise ValueError('Question ID')
    values=[[float(v) for v in x[c,s:e]] for c in cs]
    if kind=='statistics':
        total=math.fsum(values[0]);answer=total/(e-s);operands={'sum':total,'count':e-s}
        if type(row['answer']) not in (int,float) or not math.isfinite(row['answer']):raise ValueError('Numeric answer')
        if abs(row['answer']-answer)>config['answer_atol']+config['answer_rtol']*abs(answer):raise ValueError('Mean mismatch')
    else:
        maxima=[max(v) for v in values]
        if kind=='comparison':
            gap=abs(maxima[0]-maxima[1]);threshold=config['comparison_atol']+config['comparison_rtol']*max(abs(v) for v in maxima)
            if gap and gap<=threshold:raise ValueError('Ambiguous comparison')
            answer='Tie' if gap==0 else ('Counter A' if maxima[0]>maxima[1] else 'Counter B')
            operands={'maxima':maxima,'gap':gap,'separation_threshold':threshold}
        else:
            if e-s<2:raise ValueError('Insufficient peak samples')
            peaks=[];second=[];gaps=[];thresholds=[]
            for vals,maximum in zip(values,maxima):
                ordered=sorted(vals,reverse=True);gap=ordered[0]-ordered[1]
                threshold=config['peak_atol']+config['peak_rtol']*abs(maximum)
                indices=[s+i for i,v in enumerate(vals) if v==maximum]
                if len(indices)!=1 or gap<=threshold:raise ValueError('Ambiguous temporal peak')
                peaks.append(indices[0]);second.append(ordered[1]);gaps.append(gap);thresholds.append(threshold)
            answer='Yes' if peaks[0]<peaks[1] else ('No' if peaks[0]>peaks[1] else 'Same bin')
            operands={'maxima':maxima,'peak_bins':peaks,'runner_up':second,'peak_gaps':gaps,'peak_thresholds':thresholds}
        if row['answer']!=answer:raise ValueError('Categorical mismatch')
    if row['answer_type']!=('number' if kind=='statistics' else 'category'):raise ValueError('Answer type')
    reference={'version':config['version'],'precision':'float32 inputs; float64 arithmetic','operands':operands}
    if set(row['oracle'])!=set(reference) or row['oracle']['version']!=reference['version'] or row['oracle']['precision']!=reference['precision']:
        raise ValueError('Oracle metadata')
    def compare(a,b):
        if isinstance(b,dict):
            if not isinstance(a,dict) or set(a)!=set(b):raise ValueError('Operand schema')
            for k in b:compare(a[k],b[k])
        elif isinstance(b,list):
            if not isinstance(a,list) or len(a)!=len(b):raise ValueError('Operand length')
            for v,w in zip(a,b):compare(v,w)
        elif type(a) not in (int,float) or not math.isfinite(a) or abs(a-b)>1e-12*max(1,abs(b)):
            raise ValueError('Operand mismatch')
    compare(row['oracle']['operands'],operands)
    return True
