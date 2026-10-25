"""Check every stored OHLCV/return window against the original Yahoo snapshots."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
start=time.time()
df=pd.read_parquet(ROOT/'data/processed/examples.parquet')
with np.load(ROOT/'data/processed/arrays.npz') as archive:
    windows=archive['windows']
    returns=archive['returns_bp']
assert windows.shape==(len(df),100,5)
assert returns.shape==(len(df),100)
assert np.isfinite(windows).all()
checked=[]
for symbol,group in df.groupby('symbol',sort=False):
    frame=pd.read_parquet(ROOT/'data/raw/prices'/f'{symbol}.parquet').sort_values('date').drop_duplicates('date')
    good=(frame[['Open','High','Low','Close','Adj Close']]>0).all(axis=1)&(frame.High>=frame.Low)
    frame=frame.loc[good].reset_index(drop=True)
    positions=frame.set_index('date').index.get_indexer(group.origin_date)
    assert (positions>=100).all()
    factor=(frame['Adj Close']/frame.Close).to_numpy()
    prices=frame[['Open','High','Low','Close']].to_numpy()*factor[:,None]
    volume=frame.Volume.fillna(0).to_numpy()
    source=np.column_stack([prices,volume])
    expected=np.lib.stride_tricks.sliding_window_view(source,100,axis=0).swapaxes(1,2)[positions-99].astype('float32')
    actual=windows[group.index]
    np.testing.assert_array_equal(actual,expected,err_msg=f'{symbol}: an OHLCV window differs from source')
    logret=np.r_[np.nan,np.diff(np.log(prices[:,3]))]
    expected_returns=np.rint(np.lib.stride_tricks.sliding_window_view(logret,100)[positions-99]*10000).astype('int32')
    np.testing.assert_array_equal(returns[group.index],expected_returns,err_msg=f'{symbol}: a return window differs from source')
    prompt_returns=np.array([list(map(int,state.split('r_bp=',1)[1].split('\n',1)[0].split(','))) for state in group.state],dtype='int32')
    np.testing.assert_array_equal(prompt_returns,expected_returns,err_msg=f'{symbol}: prompt numbers differ from source')
    checked.append({'symbol':symbol,'windows':len(group),'ohlcv_values':int(actual.size),'return_values':int(expected_returns.size)})
    print(symbol,len(group),'complete source windows matched',flush=True)
report={'passed':True,'windows_checked':len(df),'ohlcv_values_checked':int(windows.size),'return_values_checked':int(returns.size),'prompt_return_values_checked':int(returns.size),'per_symbol':checked,'dataset_sha256':hashlib.sha256((ROOT/'data/processed/examples.parquet').read_bytes()).hexdigest(),'seconds':time.time()-start,'completed_unix':time.time()}
(ROOT/'results/context_values_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='per_symbol'},indent=2))
