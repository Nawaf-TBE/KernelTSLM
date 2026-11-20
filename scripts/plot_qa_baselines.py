"""Validation figures from independently verified, frozen QA baseline results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/phase4/qa_baselines_v1';FIG=OUT/'figures'

def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='passed'
    FIG.mkdir(exist_ok=True);m=json.loads((OUT/'metrics.json').read_text());u=json.loads((OUT/'uncertainty.json').read_text())
    models=['majority','lookup','primary','aware','direct'];labels=['Majority','Counter pair','Primary LR','Aware LR','Direct calculation']
    colors=['#737373','#bd8b28','#237b9f','#8666a6','#2d8a62']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for ax,kind in zip(axes,['comparison','temporal']):
        values=[m[f'{kind}/{s}/original']['accuracy'] for s in models]
        ci=np.asarray([u['metrics'][f'{kind}/{s}/original']['percentile95']['accuracy'] for s in models])
        ax.bar(np.arange(5),values,color=colors)
        ax.errorbar(np.arange(5),values,yerr=[np.maximum(0,np.array(values)-ci[:,0]),np.maximum(0,ci[:,1]-values)],fmt='none',ecolor='black',capsize=4)
        ax.set(xticks=np.arange(5),xticklabels=labels,ylim=(0,1.08),ylabel='Accuracy (95% whole-trace CI)',title=f'Validation: {kind}')
        ax.tick_params(axis='x',rotation=25)
        for i,v in enumerate(values):ax.text(i,v+.025,f'{100*v:.1f}%',ha='center',fontsize=9)
    fig.savefig(FIG/'accuracy.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for row,kind in enumerate(['comparison','temporal']):
        for col,model in enumerate(['primary','aware']):
            ax=axes[row,col];item=m[f'{kind}/{model}/original'];cm=np.asarray(item['confusion_matrix'])
            ax.imshow(cm,cmap='Blues');ax.set(xticks=range(len(cm)),yticks=range(len(cm)),xticklabels=item['classes'],yticklabels=item['classes'],xlabel='Predicted answer',ylabel='Reference answer',title=f'Validation: {kind}, {model} LR')
            for i in range(len(cm)):
                for j in range(len(cm)):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()*.55 else 'black')
    fig.savefig(FIG/'confusions.png',dpi=180);plt.close(fig)
    pred=json.loads((OUT/'predictions.json').read_text());fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for model,label,color in [('majority','Training-median constant','#737373'),('direct','Direct calculation','#2d8a62')]:
        r=pred[f'statistics/{model}/original'];error=np.sort([abs(v['prediction']-v['reference']) for v in r])
        ax.step(error,np.arange(1,len(error)+1)/len(error),where='post',label=label,color=color,lw=2)
    ax.set(xscale='symlog',xlabel='Absolute error, normalized units (symlog)',ylabel='Cumulative fraction',title='Validation statistics QA: numerical-error distribution',ylim=(0,1.03));ax.legend()
    fig.savefig(FIG/'statistics_errors.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained');names=['lookup','primary','aware','direct'];short=['Counter pair','Primary LR','Aware LR','Direct calc.']
    for ax,kind in zip(axes,['comparison','temporal']):
        for i,(cond,label,color) in enumerate([('original','Original input / original labels','#237b9f'),('reversed_original','Reversed input / original labels','#bd8b28'),('reversed_altered','Reversed input / altered labels','#2d8a62')]):
            ax.bar(np.arange(4)+(i-1)*.26,[m[f'{kind}/{s}/{cond}']['accuracy'] for s in names],width=.26,label=label,color=color)
        ax.set(xticks=np.arange(4),xticklabels=short,ylabel='Accuracy',ylim=(0,1.18),title=f'Validation reverse-time control: {kind}');ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=7)
    fig.savefig(FIG/'temporal_ablation.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');ys=[];names=[]
    for kind in ['comparison','temporal']:
        for model in ['primary','aware','direct']:
            key=f'{kind}/{model}/original minus {kind}/lookup/original';ci=u['paired_differences_percentile95'][key]['accuracy']
            point=m[f'{kind}/{model}/original']['accuracy']-m[f'{kind}/lookup/original']['accuracy'];i=len(names)
            ax.plot(ci,[i,i],color='#237b9f',lw=2);ax.scatter(point,i,color='#237b9f');names.append(f'{kind}: {model} − counter pair')
    ax.axvline(0,color='gray',ls='--');ax.set(yticks=range(len(names)),yticklabels=names,xlabel='Paired accuracy difference (95% trace-bootstrap CI)',title='Validation improvements relative to identity shortcut')
    fig.savefig(FIG/'paired_intervals.png',dpi=180);plt.close(fig)
    print('Saved five validation figures')

if __name__=='__main__':main()
