"""Causal recency annotation before fine-tuning; preserve all completed cached inputs."""
import gzip,hashlib,json,time
from pathlib import Path
import pandas as pd
from news import render_news
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'data/processed/examples.parquet';table=pd.read_parquet(p);before=table.state.copy();old_sha=hashlib.sha256(p.read_bytes()).hexdigest()
assert not (table.loc[table.split=='validation','news_json']!='[]').any(), 'Validation news would require restarting the comparison'
updates={}
for row in table.loc[table.news_json!='[]',['state','news_json','origin_utc']].itertuples():
 updates[row.Index]=row.state.split('\nNews',1)[0]+render_news(json.loads(row.news_json),pd.Timestamp(row.origin_utc))
changed=[i for i,s in updates.items() if s!=before.loc[i]]
if not changed:
 print('Recency already present; no dataset mutation needed');raise SystemExit(0)
table.loc[changed,'state']=[updates[i] for i in changed]
assert table.loc[table.split=='validation','state'].equals(before.loc[table.split=='validation'])
verified=0
for cache in (ROOT/'results/cache').glob('*'):
 for f in cache.glob('[0-9]*.npz'):
  if not f.stem.isdigit():continue
  idx=int(f.stem);assert table.loc[idx,'state']==before.loc[idx],f'Cached input changed: {f}';verified+=1
tmp=p.with_suffix('.tmp.parquet');table.to_parquet(tmp,index=False);tmp.replace(p);new_sha=hashlib.sha256(p.read_bytes()).hexdigest()
for mp in (ROOT/'results/cache').glob('*/manifest.json'):
 m=json.loads(mp.read_text());assert m['dataset_sha256']==old_sha
 m['dataset_sha256']=new_sha;m['recency_augmentation_cached_inputs_verified']=verified;mp.write_text(json.dumps(m,indent=2))
# Update the detailed exported training input text, preserving the same rows and all numbers.
export=ROOT/'data/processed/aapl_train.jsonl.gz';temp_export=export.with_suffix('.tmp.gz');mapping=table.set_index('id').state
with gzip.open(export,'rt') as source,gzip.open(temp_export,'wt') as target:
 for line in source:
  row=json.loads(line);state=mapping.loc[row['id']];row['state']=state;row['input']['state']=state;target.write(json.dumps(row,ensure_ascii=False)+'\n')
temp_export.replace(export)
mp=ROOT/'data/processed/manifest.json';m=json.loads(mp.read_text());m['examples_sha256']=new_sha;m['news_recency_annotation']={'examples_changed':len(changed),'validation_inputs_unchanged':True,'completed_cached_inputs_verified':verified,'clock':'per-example origin_utc; never current wall clock'};mp.write_text(json.dumps(m,indent=2))
audit={'old_dataset_sha256':old_sha,'new_dataset_sha256':new_sha,**m['news_recency_annotation'],'completed_unix':time.time()};(ROOT/'results/news_recency_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
