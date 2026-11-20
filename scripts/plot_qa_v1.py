"""Render verified QA counts and training-only examples; no raw/TimeF reads."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase4/qa_v1';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
q=json.loads((OUT/'quality.json').read_text())
colors=['#19647e','#ef8354']
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
kinds=['statistics','comparison','temporal']
for i,(split,v) in enumerate(q['splits'].items()):
 axes[0].bar(np.arange(3)+i*.36,[v['counts'].get(k,0) for k in kinds],width=.36,label=split,color=colors[i])
 lengths=sorted((int(k),n) for k,n in v['interval_lengths'].items())
 axes[1].hist([a for a,n in lengths],weights=[n for a,n in lengths],bins=np.arange(0,529,16),histtype='step',label=split,lw=2,color=colors[i])
axes[0].set(xticks=np.arange(3)+.18,xticklabels=kinds,title='Questions by type',ylabel='Questions');axes[0].legend()
axes[1].set(title='Interval lengths',xlabel='Bins, half-open intervals',ylabel='Questions');axes[1].legend()
cats=[('comparison','Counter A'),('comparison','Counter B'),('comparison','Tie'),('temporal','Yes'),('temporal','No'),('temporal','Same bin')]
for i,(split,v) in enumerate(q['splits'].items()):
 axes[2].bar(np.arange(6)+i*.36,[v['answer_categories'].get(k,{}).get(a,0) for k,a in cats],width=.36,label=split,color=colors[i])
axes[2].set(xticks=np.arange(6)+.18,xticklabels=[a for k,a in cats],title='Categorical answers',ylabel='Questions');axes[2].tick_params(axis='x',rotation=35);axes[2].legend()
fig.savefig(FIG/'distributions.png',dpi=160);plt.close(fig)
examples=json.loads((OUT/'illustrations.json').read_text())
for item in examples:
 row=item['example'];x=np.asarray(item['X']);s,e=row['start'],row['end'];fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
 ax.axvspan(s,e,color='#ddd',alpha=.4,label=f'Interval [{s}, {e})')
 for c,n in zip(row['channels'],row['counter_names']):
  curve=ax.plot(np.arange(512),x[c],lw=1,label=f'{c}: {n}')[0];color=curve.get_color()
  if row['type']=='statistics':
   ax.hlines(row['answer'],s,e,color=color,ls='--',lw=2,label=f"Mean = {row['answer']:.6g}")
  else:
   v=x[c,s:e];maximum=v.max();indices=np.flatnonzero(v==maximum)+s
   ax.scatter(indices,x[c,indices],color=color,zorder=5,s=22)
   ax.hlines(maximum,s,e,color=color,ls='--',alpha=.7,label=f'Max = {maximum:.6g}')
 title_answer=f"{row['answer']:.6g}" if row['type']=='statistics' else row['answer']
 ax.set(title=f"Training example: {row['type']} | Reference answer: {title_answer}",xlabel='Original time-bin index',ylabel='Train-normalized counter value (z-score)',xlim=(0,511))
 ax.legend(fontsize=7,loc='upper right');fig.savefig(FIG/f"example_{row['type']}.png",dpi=160);plt.close(fig)
print('Saved',len(list(FIG.glob('*.png'))),'figures')
