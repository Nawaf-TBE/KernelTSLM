"""Render saved CNN metrics with the existing Phase 1 plotting environment."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/phase3/cnn'


def csv_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    provenance = json.loads((OUT / 'provenance.json').read_text())
    names = provenance['names']
    label_names = [name.replace('_', '\n') for name in names]
    for kind in ['unweighted', 'weighted']:
        folder = OUT / kind
        result = json.loads((folder / 'results.json').read_text())
        metrics = result['validation']
        csv_rows(folder / 'per_class_metrics.csv', metrics['per_class'])
        csv_rows(folder / 'per_trace_metrics.csv', metrics['per_trace'])
        (folder / 'validation_results.json').write_text(json.dumps(metrics, indent=2) + '\n')
        cm = np.asarray(metrics['confusion_matrix'])
        with (folder / 'confusion_matrix.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['true/predicted'] + names)
            writer.writerows([[name] + row.tolist() for name, row in zip(names, cm)])
        normalized = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(1,keepdims=True)>0)
        fig, ax = plt.subplots(figsize=(11, 9), layout='constrained')
        img = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        ax.set(xticks=range(12), yticks=range(12), xticklabels=label_names, yticklabels=label_names,
               xlabel='Predicted L1 class', ylabel='True L1 class',
               title=f'{kind.title()} CNN — validation (valid primary bins only)\nRow-normalized color; annotations are counts')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        for i in range(12):
            for j in range(12):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=6,
                        color='white' if normalized[i,j]>.55 else '#172b4d')
        fig.colorbar(img, ax=ax, label='Fraction of true-class bins', shrink=.8)
        fig.savefig(folder / 'confusion_matrix.png', dpi=160)
        plt.close(fig)
        history = json.loads((folder / 'history.json').read_text())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        epochs = [row['epoch'] for row in history]
        axes[0].plot(epochs, [r['train_unweighted_ce'] for r in history], label='Train CE (online, dropout on)')
        axes[0].plot(epochs, [r['val_ce'] for r in history], label='Validation CE')
        if kind == 'weighted':
            axes[0].plot(epochs, [r['train_objective'] for r in history], '--', label='Train weighted objective')
        axes[0].set(xlabel='Epoch', ylabel='Masked loss per valid bin')
        axes[1].plot(epochs, [r['train_online_macro_f1'] for r in history], label='Train online')
        axes[1].plot(epochs, [r['val_macro_f1'] for r in history], label='Validation')
        axes[1].set(xlabel='Epoch', ylabel='Macro-F1, fixed 11 supported classes', ylim=(0,1))
        for ax in axes:
            ax.axvline(result['best_epoch'], color='0.4', linestyle=':', label='Selected epoch')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle(f'{kind.title()} CNN — same seed, architecture and schedule')
        fig.savefig(folder / 'learning_curves.png', dpi=160); plt.close(fig)
        truth_share = cm.sum(1) / cm.sum()
        pred_share = cm.sum(0) / cm.sum()
        fig, ax = plt.subplots(figsize=(11,4), layout='constrained')
        xs = np.arange(12)
        ax.bar(xs-.2, truth_share, width=.4, label='Validation truth')
        ax.bar(xs+.2, pred_share, width=.4, label='Prediction')
        ax.set(xticks=xs, xticklabels=label_names, ylabel='Fraction of valid bins',
               title=f'{kind.title()} CNN — prediction distribution')
        ax.tick_params(axis='x', labelsize=8); ax.legend(); ax.grid(axis='y',alpha=.2)
        fig.savefig(folder/'prediction_distribution.png',dpi=160);plt.close(fig)
    # A deterministic qualitative diagnostic, not a selection metric: the first
    # validation trace containing at least two valid L1 classes.
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    loaded = {}
    for kind in ['unweighted', 'weighted']:
        with np.load(OUT/kind/'validation_predictions.npz', allow_pickle=False) as arrays:
            loaded[kind] = {key: arrays[key].copy() for key in arrays.files}
    rows = loaded['unweighted']
    trace = next(i for i in range(len(rows['trace_ids']))
                 if len(np.unique(rows['true_class'][rows['trace_index']==i])) >= 2)
    matrix = np.full((3,512), -1, dtype=int)
    for index, kind in enumerate(['unweighted','weighted'], start=1):
        item = loaded[kind]
        mask = item['trace_index'] == trace
        matrix[0, item['bin_index'][mask]] = item['true_class'][mask]
        matrix[index, item['bin_index'][mask]] = item['predicted_class'][mask]
    colors = plt.get_cmap('tab20')(np.arange(12))
    cmap = ListedColormap(colors); cmap.set_bad('#dedede')
    fig, ax = plt.subplots(figsize=(12,4), layout='constrained')
    ax.imshow(np.ma.masked_less(matrix,0), aspect='auto', interpolation='nearest', cmap=cmap, vmin=-.5, vmax=11.5)
    ax.set(yticks=[0,1,2], yticklabels=['Primary truth','Unweighted','Weighted'],
           xlabel='Original input bin index (no resampling)',
           title='First multiclass validation trace — deterministic qualitative selection\n'+str(rows['trace_ids'][trace]))
    present = sorted(set(matrix[matrix>=0].tolist()))
    legend = [Patch(color=colors[i],label=names[i]) for i in present]
    legend.append(Patch(color='#dedede',label='Unknown / no scored prediction'))
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5,-.18), ncol=min(4,len(legend)),fontsize=8)
    fig.savefig(OUT/'temporal_diagnostic.png',dpi=160);plt.close(fig)
    fig, ax = plt.subplots(figsize=(11,5), layout='constrained')
    x = np.arange(12)
    for offset, kind in [(-.2,'unweighted'),(.2,'weighted')]:
        values = [r['f1'] for r in summary['experiments'][kind]['validation']['per_class']]
        ax.bar(x+offset,values,.4,label=kind)
    ax.set(xticks=x,xticklabels=label_names,ylabel='Validation F1',ylim=(0,1),
           title='Controlled weighting comparison — original 12-class vocabulary')
    ax.tick_params(axis='x',labelsize=8);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'per_class_comparison.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    main()
