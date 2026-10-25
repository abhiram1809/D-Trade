"""Audit every target, label, context length and selected-news cutoff against source snapshots."""
import gzip,hashlib,json,time
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from news import render_news
ROOT=Path(__file__).resolve().parents[1]
start=time.time();df=pd.read_parquet(ROOT/'data/processed/examples.parquet');stocks=[]
assert df.id.is_unique and (df.n_points==100).all() and (df.origin_date<df.target_date).all()
for symbol,group in df.groupby('symbol',sort=False):
 raw=pd.read_parquet(ROOT/'data/raw/prices'/f'{symbol}.parquet').drop_duplicates('date').set_index('date')
 expected=np.log(raw.loc[group.target_date,'Close'].to_numpy()/raw.loc[group.target_date,'Open'].to_numpy())
 actual=group.target_log_return.to_numpy();np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=0)
 bins=np.searchsorted([-1.,-.25,.25,1.],expected/group.sigma.to_numpy(),side='right')
 np.testing.assert_array_equal(bins,group.label.to_numpy())
 stocks.append({'symbol':symbol,'targets_checked':len(group),'maximum_return_error':float(np.max(abs(actual-expected)))})
for state in df.state:
 values=state.split('r_bp=',1)[1].split('\n',1)[0].split(',');assert len(values)==100
 assert all(v.lstrip('-').isdigit() for v in values)
news_count=0
for row in df.loc[df.news_json!='[]',['origin_utc','news_json','state']].itertuples(index=False):
 origin=datetime.fromisoformat(row.origin_utc)
 assert row.state.endswith(render_news(json.loads(row.news_json),origin))
 for article in json.loads(row.news_json):
  available=datetime.fromisoformat(article['available_utc']);published=datetime.fromisoformat(article['published_utc'])
  assert available<=origin and (origin-available).total_seconds()<=72*3600
  assert (available-published).total_seconds()>=24*3600
  news_count+=1
count=0
by_id=df.set_index('id').state
with gzip.open(ROOT/'data/processed/aapl_train.jsonl.gz','rt') as f:
 for line in f:
  row=json.loads(line);assert row['symbol']=='AAPL' and row['target_date']<'2024-01-01'
  assert len(row['input']['ohlcv_100'])==100;count+=1
  assert row['state']==row['input']['state']==by_id.loc[row['id']]
assert count>=10000 and count==len(df[(df.symbol=='AAPL')&(df.split=='train')])
report={'passed':True,'all_examples_checked':len(df),'all_target_symbols_checked':len(stocks),'all_prompt_context_lengths_checked':len(df),'selected_news_records_checked':news_count,'aapl_training_export_records_checked':count,'per_symbol_target_reconstruction':stocks,'dataset_sha256':hashlib.sha256((ROOT/'data/processed/examples.parquet').read_bytes()).hexdigest(),'seconds':time.time()-start,'completed_unix':time.time()}
(ROOT/'results/dataset_full_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='per_symbol_target_reconstruction'},indent=2))
