"""Final causal-data audit before any model scoring; keep special Indian sessions."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from news import POLICY
ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'data/processed'
df=pd.read_parquet(folder/'examples.parquet')
a=np.load(folder/'arrays.npz')
# Yahoo's three Saturday FX tail bars are outside normal FX sessions.
bad=(df.market=='FX')&((pd.to_datetime(df.origin_date).dt.day_of_week>=5)|(pd.to_datetime(df.target_date).dt.day_of_week>=5))
keep=~bad
arrays={key:a[key][keep] for key in a.files}
df=df.loc[keep].reset_index(drop=True)
# Future split/dividend adjustment changes absolute historical price levels.
# Ratio/return features cancel the common future factor. Never expose that level in prompts.
df['state']=df.state.str.replace(r' Last close=[^.]*? [A-Z]{3}\.', '', regex=True)
# The decimal price contains periods; use an exact field boundary instead.
df['state']=df.state.str.replace(r' Last close=.*?\. News coverage=', ' News coverage=', regex=True)
df.to_parquet(folder/'examples.parquet',index=False)
np.savez_compressed(folder/'arrays.npz',**arrays)
selection=[]
for (symbol,split),group in df[df.split!='train'].groupby(['symbol','split']):
 count=12 if split=='validation' else 24
 selection.extend(group.iloc[np.unique(np.linspace(0,len(group)-1,min(count,len(group))).astype(int))].index.tolist())
(folder/'evaluation_indices.json').write_text(json.dumps(selection))
aapl=df[(df.symbol=='AAPL')&(df.split=='train')]
with gzip.open(folder/'aapl_train.jsonl.gz','wt') as f:
 for idx,row in aapl.iterrows():
  obj=row.to_dict();obj['input']={'policy':POLICY,'state':row.state,'ohlcv_100':arrays['windows'][idx].tolist(),'news':json.loads(row.news_json)};obj['target']={'bin':row.label_name,'next_session_open_to_close_log_return':row.target_log_return}
  f.write(json.dumps(obj,ensure_ascii=False)+'\n')
m=json.loads((folder/'manifest.json').read_text());m['total_examples']=len(df);m['examples_sha256']=hashlib.sha256((folder/'examples.parquet').read_bytes()).hexdigest();m['quality_audit']={'removed_fx_weekend_examples':int(bad.sum()),'special_indian_weekend_sessions_retained':True,'absolute_adjusted_price_removed_from_prompt':True};m['split_counts']=df.groupby('split').size().to_dict();(folder/'manifest.json').write_text(json.dumps(m,indent=2))
print(json.dumps(m['quality_audit']));print(m['split_counts']);print('AAPL train',len(aapl))
