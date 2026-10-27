"""Generate exportable figures only from completed, frozen test predictions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from metrics import strategy

ROOT=Path(__file__).resolve().parents[1]
MODELS=['base','fine_tuned','class_frequency','numeric_logistic']
COLORS=['#4477AA','#228833','#EEAA33','#AA3377']
LABELS={'base':'Base decision model','fine_tuned':'Fine-tuned decision head','class_frequency':'Training class frequency','numeric_logistic':'Numeric logistic baseline'}


def render():
    # No mock measurements or unfinished-run placeholders are accepted here.
    summary=json.loads((ROOT/'results/summary.json').read_text())
    metrics=pd.read_csv(ROOT/'results/forecast_metrics.csv')
    predictions=pd.read_parquet(ROOT/'results/test_predictions.parquet')
    stock=predictions[predictions.symbol=='AAPL'].sort_values('target_date').reset_index(drop=True)
    if len(stock)!=summary['primary_stock_test_examples'] or stock.target_date.duplicated().any():
        raise ValueError('The primary-stock figure requires the complete unique test series')
    figures=ROOT/'results/figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','axes.grid':True,'grid.alpha':.2,'axes.axisbelow':True})
    paths=[]
    def save(fig,name):
        fig.savefig(figures/f'{name}.png',dpi=180,bbox_inches='tight')
        fig.savefig(figures/f'{name}.svg',bbox_inches='tight')
        plt.close(fig)
        paths.append(str(figures/f'{name}.png'))

    primary=metrics[metrics.group=='AAPL'].set_index('model').loc[MODELS]
    labels=['Base','Fine-tuned\nhead','Class\nfrequency','Numeric\nlogistic']
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,column,title in zip(axes,['log_loss','brier'],['Five-bin log loss','Multiclass Brier score']):
        ax.bar(labels,primary[column],color=COLORS,width=.65)
        ax.set_title(title+' — lower is better')
        ax.set_ylabel('Mean loss per forecast')
        for i,value in enumerate(primary[column]):
            ax.annotate(f'{value:.3f}',(i,value),ha='center',xytext=(0,5),textcoords='offset points',fontsize=9)
        ax.margins(y=.18)
    fig.suptitle(f'AAPL held-out prediction quality · {len(stock)} sessions',fontsize=14)
    save(fig,'forecast_quality')

    dates=pd.to_datetime(stock.target_date)
    curves=pd.DataFrame({'date':stock.target_date})
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    for name,color in zip(MODELS,COLORS):
        probabilities=stock[[f'{name}_p{k}' for k in range(5)]].to_numpy()
        _,net=strategy(stock,probabilities)
        wealth=10_000*np.cumprod(1+net)
        curves[name]=wealth
        ax.plot(dates,wealth,label=LABELS[name],color=color,linewidth=1.7)
    # Every position is closed daily, including this constant-long comparator.
    passive=np.expm1(stock.target_log_return.to_numpy())-2*.0005
    wealth=10_000*np.cumprod(1+passive)
    curves['always_long_intraday']=wealth
    ax.plot(dates,wealth,label='Always long intraday',color='#555555',linestyle='--',linewidth=1.4)
    ax.axhline(10_000,color='#aaaaaa',linewidth=.7)
    ax.set(title='AAPL intraday research replay after costs',ylabel='Wealth from 10,000 normalized currency units',xlabel='Held-out session date')
    ax.legend(loc='best',fontsize=8,frameon=False)
    if all(np.all(curves[name].to_numpy()==10_000) for name in MODELS):
        ax.text(.02,.03,'All four forecast rules abstained (0 trades); their curves overlap at 10,000.',transform=ax.transAxes,fontsize=9,color='#555555',bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
    fig.text(.01,-.035,'Signals use prior-session observations. Costs: 5 bps per side. No overnight holdings or broker-fill assumptions.',fontsize=9,color='#555555')
    save(fig,'aapl_intraday_equity')
    curves.to_csv(ROOT/'results/aapl_equity_curves.csv',index=False)

    y=stock.label.to_numpy(int)
    fig,ax=plt.subplots(figsize=(6.8,5.5),layout='constrained')
    ax.plot([0,1],[0,1],color='#777777',linestyle='--',linewidth=1,label='Ideal calibration')
    calibration=[]
    edges=np.linspace(0,1,11)
    for name,color in zip(MODELS,COLORS):
        p=stock[[f'{name}_p{k}' for k in range(5)]].to_numpy()
        confidence=p.max(1)
        correct=p.argmax(1)==y
        xs,ys,counts=[],[],[]
        for low,high in zip(edges[:-1],edges[1:]):
            selected=(confidence>low)&(confidence<=high)
            if not selected.any():continue
            x=float(confidence[selected].mean());accuracy=float(correct[selected].mean());count=int(selected.sum())
            xs.append(x);ys.append(accuracy);counts.append(count)
            calibration.append({'model':name,'bin_low':low,'bin_high':high,'n':count,'mean_confidence':x,'accuracy':accuracy})
        ax.scatter(xs,ys,s=30+np.asarray(counts)/len(stock)*180,color=color,label=LABELS[name],alpha=.85,edgecolors='white',linewidths=.6)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Reported probability of the most likely five-bin answer',ylabel='Observed accuracy in that confidence bin',title='AAPL calibration on newer held-out sessions')
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=8,frameon=False,loc='upper left')
    fig.text(.01,-.025,'Marker area indicates sample count. Sparse bins provide weak calibration evidence.',fontsize=9,color='#555555')
    save(fig,'aapl_calibration')
    pd.DataFrame(calibration).to_csv(ROOT/'results/calibration_bins.csv',index=False)

    (ROOT/'results/figures_manifest.json').write_text(json.dumps({'source':'completed test_predictions.parquet and forecast_metrics.csv','primary_stock_test_examples':len(stock),'png_files':paths,'svg_files':[str(Path(p).with_suffix('.svg')) for p in paths]},indent=2))
    return paths


if __name__=='__main__':
    for path in render():print(path)
