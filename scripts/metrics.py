"""Forecast and executable next-session intraday diagnostics."""
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss


def score(y, p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-9, 1)
    p /= p.sum(axis=1, keepdims=True)
    y = np.asarray(y, dtype=int)
    pred = p.argmax(1)
    p3 = np.c_[p[:, :2].sum(1), p[:, 2], p[:, 3:].sum(1)]
    y3 = np.array([0, 0, 1, 2, 2])[y]
    q = p.max(1)
    correct = pred == y
    ece = 0.
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        m = (q > lo) & (q <= hi)
        if m.any():
            ece += m.mean() * abs(q[m].mean() - correct[m].mean())
    central = np.isin(y, [1, 2, 3])
    nonflat=y!=2
    binary={"nonflat_n":int(nonflat.sum()),"nonflat_directional_accuracy":None,"nonflat_binary_log_loss":None,"nonflat_binary_brier":None}
    if nonflat.any():
        up=p[nonflat,3:].sum(1)/(p[nonflat,:2].sum(1)+p[nonflat,3:].sum(1))
        truth=(y[nonflat]>=3).astype(int)
        binary.update(nonflat_directional_accuracy=float(np.mean((up>=.5)==truth)),nonflat_binary_log_loss=float(log_loss(truth,np.c_[1-up,up],labels=[0,1])),nonflat_binary_brier=float(np.mean((up-truth)**2)))
    return {'n': len(y), 'five_bin_accuracy': accuracy_score(y, pred), 'five_bin_macro_f1': f1_score(y,pred,labels=np.arange(5),average='macro',zero_division=0), 'three_class_accuracy': accuracy_score(y3,p3.argmax(1)), 'three_class_macro_f1': f1_score(y3,p3.argmax(1),labels=np.arange(3),average='macro',zero_division=0), 'log_loss': log_loss(y,p,labels=np.arange(5)), 'brier': float(np.mean(np.sum((p-np.eye(5)[y])**2,axis=1))), 'ece':float(ece), 'actual_central_interval_coverage':float(central.mean()), 'predicted_central_interval_mass':float(p[:,1:4].sum(1).mean()), **binary}


def directional_positions(p, confidence_gate=.55):
    """Full allocation to the leading direction; None removes the confidence gate.

    In the ungated rule, flat and ties stay in cash. Down/up aggregate their two
    range bins, so allocation uses directional probabilities rather than one bin.
    """
    p=np.asarray(p)
    if p.ndim!=2 or p.shape[1]!=5 or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('Expected finite nonnegative five-bin probabilities')
    if not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):
        raise ValueError('Forecast probability rows must sum to one')
    p3=np.c_[p[:,:2].sum(1),p[:,2],p[:,3:].sum(1)]
    choice=p3.argmax(1)
    if confidence_gate is None:
        unique=np.isclose(p3,p3.max(1,keepdims=True),atol=1e-12,rtol=0).sum(1)==1
        return np.where(unique&(choice==2),1.,np.where(unique&(choice==0),-1.,0.))
    if not 0<=confidence_gate<=1:raise ValueError('Confidence gate must be in [0,1] or None')
    return np.where((choice==2)&(p3[:,2]>=confidence_gate),1.,np.where((choice==0)&(p3[:,0]>=confidence_gate),-1.,0.))


def strategy(frame, p, cost_multiplier=1., *, confidence_gate=.55):
    """Intraday exposure only; two sides paid each active session, no overnight holding."""
    position = directional_positions(p,confidence_gate)
    costs = frame.market.map({'India':.001,'US':.0005,'Global':.0005,'FX':.0001}).to_numpy()*cost_multiplier
    r = np.expm1(frame.target_log_return.to_numpy())
    net = position * r - 2 * costs * abs(position)
    passive = r - 2 * costs
    def stats(values):
        # Equal-weight observed instruments each date, local-currency normalized returns.
        daily = frame[['target_date']].assign(r=values).groupby('target_date').r.mean().sort_index().to_numpy()
        equity=np.cumprod(1+daily);peak=np.maximum.accumulate(np.r_[1.,equity])[1:]
        return {'total_return':float(equity[-1]-1) if len(equity) else 0.,'worst_drawdown':float(np.min(equity/peak-1)) if len(equity) else 0.,'annualized_sharpe':float(daily.mean()/daily.std(ddof=1)*np.sqrt(252)) if len(daily)>1 and daily.std(ddof=1)>0 else None,'observed_sessions':len(daily)}
    # Annualized Sharpe is meaningful only for a continuous series; expose this explicitly.
    result=stats(net)
    result.update(active_fraction=float(np.mean(position!=0)),two_sided_turnover=float(np.sum(2*abs(position))),passive_intraday=stats(passive))
    return result, net


def block_bootstrap_loss_difference(frame,p_a,p_b,seed=20261025,repetitions=1000):
    y=frame.label.to_numpy(int)
    delta=-np.log(np.clip(p_a[np.arange(len(y)),y],1e-9,1))+np.log(np.clip(p_b[np.arange(len(y)),y],1e-9,1))
    rng=np.random.default_rng(seed);groups=[np.asarray(ix) for ix in frame.reset_index(drop=True).groupby('symbol',sort=False).indices.values()]
    means=[]
    for _ in range(repetitions):
        selected=[]
        for ids in groups:
            n=len(ids);starts=rng.integers(0,n,int(np.ceil(n/20)));local=((starts[:,None]+np.arange(20))%n).ravel()[:n];selected.append(delta[ids[local]])
        means.append(np.concatenate(selected).mean())
    return {'mean_log_loss_difference_a_minus_b':float(delta.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'repetitions':repetitions,'block_observations_per_symbol':20,'note':'For subsampled symbols these are 20 observed origins, not 20 consecutive market sessions; primary AAPL is continuous.'}
