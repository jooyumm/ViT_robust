"""Export plain English bar charts for the non-adaptive region-switch experiment."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def bar_chart(folder,name,labels,values,colors,*,title,ylabel='Accuracy (%)',reference=None):
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False}):
        fig,ax=plt.subplots(figsize=(7.2,5))
        bars=ax.bar(np.arange(len(labels)),np.asarray(values)*100,color=colors,width=.6,zorder=3)
        ax.set_xticks(np.arange(len(labels)),labels)
        if any(max(map(len,label.split('\n')))>18 for label in labels):
            plt.setp(ax.get_xticklabels(),rotation=35,ha='right')
        ax.set(title=title,ylabel=ylabel,ylim=(0,110),yticks=np.arange(0,101,20))
        ax.grid(axis='y',alpha=.2,zorder=0)
        ax.bar_label(bars,labels=[f'{v*100:.2f}%' for v in values],padding=5)
        if reference is not None:
            ax.axhline(reference*100,color='#477ca8',ls='--',lw=1.2,label=f'Original clean: {reference*100:.1f}%')
            ax.legend(loc='upper left',fontsize=10)
        fig.tight_layout()
        fig.savefig(Path(folder)/name,dpi=200,bbox_inches='tight')
        plt.close(fig)


def render_summary(report,folder,baseline=None):
    """Read only the three non-adaptive rows; never plot adaptive results."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows={r['key']:r for r in report['rows'] if r['key'] in ('none','random','detected')}
    d=rows['detected'];n=report['n']
    if baseline is not None and (baseline['n']!=n or baseline['source_manifest_sha256']!=report['source_manifest_sha256']):
        raise ValueError('Main and localization results have different sources')
    bar_chart(folder,'comparison.png',['Clean','Attacked','Defended'],
              [d['clean_accuracy'],d['attack_accuracy'],d['defended_accuracy']],
              ['#477ca8','#d86a4a','#278776'],title=f'Classification accuracy (n={n})')
    bar_chart(folder,'localization_ablation.png',['No defense','Random region','Detected region'],
              [rows[k]['defended_accuracy'] for k in ('none','random','detected')],
              ['#d86a4a','#8c6bb1','#278776'],title=f'Region selection (n={n})',reference=d['clean_accuracy'])
    bar_chart(folder,'detector_metrics.png',['TPR\n(all attacks)','FPR\n(clean images)','Region hit rate\n(flagged attacks)'],
              [d['tpr'],d['fpr'],d['flagged_region_accuracy']],['#278776','#d86a4a','#477ca8'],
              title=f'Detection and localization (n={n})',ylabel='Rate (%)')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=HERE/'results')
    args=parser.parse_args()
    report=json.loads((HERE/'results/defense_comparison.json').read_text())
    baseline=json.loads((HERE/'results/comparison.json').read_text())
    render_summary(report,args.output_dir,baseline)
