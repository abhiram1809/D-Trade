"""Replay frozen forecasts using full allocation to the most probable direction."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import directional_positions,strategy

ROOT=Path(__file__).resolve().parents[1]
MODELS=['base','fine_tuned','class_frequency','numeric_logistic']


def run():
    source=ROOT/'results/test_predictions.parquet'
    original=ROOT/'results/trading.json'
    predictions=pd.read_parquet(source)
    stock=predictions[predictions.symbol=='AAPL'].sort_values('target_date').reset_index(drop=True)
    summary=json.loads((ROOT/'results/summary.json').read_text())
    if len(stock)!=summary['primary_stock_test_examples'] or stock.target_date.duplicated().any():
        raise ValueError('Replay requires the complete unique primary-stock test series')
    original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
    curves=pd.DataFrame({'date':stock.target_date})
    models={};trades=[]
    costs=stock.market.map({'India':.001,'US':.0005,'Global':.0005,'FX':.0001}).to_numpy()
    returns=np.expm1(stock.target_log_return.to_numpy())
    for name in MODELS:
        p=stock[[f'{name}_p{k}' for k in range(5)]].to_numpy()
        p3=np.c_[p[:,:2].sum(1),p[:,2],p[:,3:].sum(1)]
        position=directional_positions(p,confidence_gate=None)
        standard,net=strategy(stock,p,confidence_gate=None)
        stress,_=strategy(stock,p,2.,confidence_gate=None)
        if np.any(net<=-1):raise ValueError('Synthetic exposure depleted its equity')
        wealth=10_000*np.cumprod(1+net)
        curves[name]=wealth
        models[name]={'standard_cost':standard,'double_cost':stress,'long_sessions':int((position>0).sum()),
                      'short_sessions':int((position<0).sum()),'cash_sessions':int((position==0).sum()),
                      'round_trip_trades':int((position!=0).sum()),
                      'gross_compounded_return':float(np.prod(1+position*returns)-1),
                      'net_win_rate_on_active_sessions':float((net[position!=0]>0).mean()) if (position!=0).any() else None}
        unique=np.isclose(p3,p3.max(1,keepdims=True),atol=1e-12,rtol=0).sum(1)==1
        selected=np.array(['down','flat','up'])[p3.argmax(1)]
        selected[~unique]='tie'
        rows=stock[['id','origin_date','target_date','symbol']].copy()
        rows['model']=name
        for k,direction in enumerate(('down','flat','up')):rows[f'p_{direction}']=p3[:,k]
        rows['highest_probability']=p3.max(1);rows['winner']=selected
        rows['position']=position;rows['allocation_fraction']=abs(position)
        rows['underlying_intraday_return']=returns
        rows['gross_strategy_return']=position*returns
        rows['round_trip_cost']=2*costs*abs(position)
        rows['net_strategy_return']=net;rows['wealth_from_10000']=wealth
        trades.append(rows)
    curves['always_long_intraday']=10_000*np.cumprod(1+returns-2*costs)
    pd.concat(trades,ignore_index=True).to_csv(ROOT/'results/highest_confidence_trades.csv',index=False)
    curves.to_csv(ROOT/'results/aapl_highest_confidence_equity_curves.csv',index=False)
    report={'rule':'highest_probability_direction','confidence_gate':None,
            'direction_probabilities':{'down':'p0+p1','flat':'p2','up':'p3+p4'},
            'allocation':'100% long for unique up winner, 100% short for unique down winner; flat or tied winner stays cash',
            'entry_exit':'Next session open to close, signal from prior completed session; reset exposure daily',
            'evaluation':'User-requested exploratory replay after the original test; no retraining or probability changes',
            'source_predictions_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'original_gated_backtest_sha256':original_hash,'symbol':'AAPL','sessions':len(stock),
            'target_first':stock.target_date.min(),'target_last':stock.target_date.max(),
            'cost_per_side_bps':5,'stress_cost_per_side_bps':10,'models':models}
    (ROOT/'results/trading_highest_confidence.json').write_text(json.dumps(report,indent=2)+'\n')
    render(curves)
    lines=['# Highest-probability allocation intraday replay','',
           'For each session, add the two down-bin probabilities and the two up-bin probabilities, then compare down, flat and up. Allocate 100% of that strategy’s equity long if up wins, or short if down wins. Flat and tied winners stay in cash. There is no minimum confidence threshold and confidence is not used to scale position size.',
           '',f"AAPL: {len(stock)} sessions, {report['target_first']} through {report['target_last']}. Signals use the previous completed session; entry is at the next open and exit at that close. Costs are 5 basis points per side, with a 10-basis-point-per-side stress case.",'',
           '| Model | Long | Short | Cash | Net return | Worst drawdown | Sharpe | Doubled-cost return |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,item in models.items():
        a=item['standard_cost'];stress=item['double_cost'];sharpe=a['annualized_sharpe']
        s='—' if sharpe is None else f'{sharpe:.2f}'
        lines.append(f"| {name} | {item['long_sessions']} | {item['short_sessions']} | {item['cash_sessions']} | {a['total_return']:.2%} | {a['worst_drawdown']:.2%} | {s} | {stress['total_return']:.2%} |")
    passive=models['fine_tuned']['standard_cost']['passive_intraday']
    lines+=['',f"Always-long intraday comparator after the same standard costs: {passive['total_return']:.2%} return and {passive['worst_drawdown']:.2%} worst drawdown.",'',
            f"The fine-tuned rule's compounded return before costs is {models['fine_tuned']['gross_compounded_return']:.2%}. Its net return includes two transaction-cost charges on each of its {models['fine_tuned']['round_trip_trades']} active sessions.",'',
            'This is the user-requested exploratory execution rule applied after the original test was observed. Forecasts, model weights, calibration, dates and labels are unchanged. The original 55% gate results remain in `trading.json` and are not overwritten. This replay is not a new unseen-data validation of the rule.',
            '', 'Positions are synthetic unlevered daily exposures. Borrow fees, financing, taxes, slippage beyond the fixed cost, and short-sale constraints are not fully modeled. Only AAPL has a complete continuous test series; sparse cross-market prediction samples are not used as continuous portfolios.',
            '', 'Per-session allocations and returns: `highest_confidence_trades.csv`. Equity curves: `aapl_highest_confidence_equity_curves.csv`. Full statistics and source hashes: `trading_highest_confidence.json`.',
            '', '![Highest-probability allocation equity](figures/aapl_highest_confidence_equity.png)','']
    (ROOT/'results/BACKTEST_HIGHEST_CONFIDENCE.md').write_text('\n'.join(lines))
    assert hashlib.sha256(original.read_bytes()).hexdigest()==original_hash
    print(json.dumps({name:{'return':r['standard_cost']['total_return'],'trades':r['round_trip_trades']} for name,r in models.items()}))
    return report


def render(curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_results import LABELS,COLORS
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    dates=pd.to_datetime(curves.date)
    for name,color in zip(MODELS,COLORS):ax.plot(dates,curves[name],label=LABELS[name],color=color,linewidth=1.7)
    ax.plot(dates,curves.always_long_intraday,label='Always long intraday',color='#555555',linestyle='--',linewidth=1.4)
    ax.axhline(10_000,label='Original 55% gate (all cash)',color='#888888',linestyle=':',linewidth=1)
    ax.set(title='AAPL intraday replay · highest-probability allocation',xlabel='Held-out session date',ylabel='Wealth from 10,000 normalized currency units')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False,loc='best')
    fig.text(.01,-.035,'100% long/short when that direction wins; flat/ties stay cash. Costs: 5 bps per side. Exploratory replay.',fontsize=9,color='#555555')
    folder=ROOT/'results/figures';folder.mkdir(exist_ok=True)
    for extension in ('png','svg'):
        target=folder/f'aapl_highest_confidence_equity.{extension}'
        fig.savefig(target,dpi=180,bbox_inches='tight')
        if extension=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':run()
