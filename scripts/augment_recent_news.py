"""Add archive text without changing examples, prices, targets, splits or sample selection."""
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from news import NewsIndex, render_news

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'data/processed'
    path=folder/'examples.parquet'
    table=pd.read_parquet(path)
    original=table.state.copy()
    old_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    index=NewsIndex(ROOT/'data/raw/news')
    matched=0
    updates={}
    # The new source is confined to test-period dates. Keep all training/validation bytes unchanged.
    for idx,row in table[table.split=='test'].iterrows():
        if row.market!='US':continue
        selected,coverage=index.at(row.symbol,pd.Timestamp(row.origin_utc))
        state=row.state.split('\nNews',1)[0]
        state=state.split('News coverage=',1)[0]+'News coverage='+coverage['status']+'.'+state.split('News coverage=',1)[1].split('.',1)[1]
        if selected:
            matched+=1
        state+=render_news(selected,pd.Timestamp(row.origin_utc))
        updates[idx]=(state,json.dumps(selected),json.dumps(coverage))
    # Pandas 3 stores strings in immutable Arrow arrays: individual .at writes
    # copy the entire large column. Assign each updated column in one operation.
    update_ids=list(updates)
    for column_number,column in enumerate(['state','news_json','news_coverage_json']):
        table.loc[update_ids,column]=[updates[idx][column_number] for idx in update_ids]
    non_test=table.split!='test'
    assert table.loc[non_test,'state'].equals(original.loc[non_test])
    # Resuming precomputed validation is safe only if each cached input is identical.
    cached_checked=0
    for cache in (ROOT/'results/cache').glob('*'):
        if not cache.is_dir():continue
        for feature in cache.glob('[0-9]*.npz'):
            if not feature.stem.isdigit():
                continue
            idx=int(feature.stem)
            if table.loc[idx,'state']!=original.loc[idx]:raise RuntimeError('A cached input changed; isolate its cache before proceeding')
            cached_checked+=1
    temporary=path.with_suffix('.tmp.parquet')
    table.to_parquet(temporary,index=False)
    temporary.replace(path)
    new_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    for manifest_path in (ROOT/'results/cache').glob('*/manifest.json'):
        manifest=json.loads(manifest_path.read_text());assert manifest['dataset_sha256']==old_sha
        manifest['dataset_sha256']=new_sha
        manifest['unchanged_cached_inputs_news_augmentation']={'old_dataset_sha256':old_sha,'new_dataset_sha256':new_sha,'verified_cached_inputs':cached_checked}
        manifest_path.write_text(json.dumps(manifest,indent=2))
    manifest=json.loads((folder/'manifest.json').read_text())
    manifest['examples_sha256']=new_sha
    manifest['news_coverage']=index.coverage
    manifest['news_matched_examples']=int((table.news_json!='[]').sum())
    manifest['test_us_news_matched_examples']=matched
    manifest['recent_news_archive']=json.loads((ROOT/'data/raw/news/recent_archive_manifest.json').read_text())
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    audit={'old_dataset_sha256':old_sha,'new_dataset_sha256':new_sha,'train_and_validation_states_unchanged':True,'cached_inputs_verified':cached_checked,'US_test_examples_with_news':matched,'labels_prices_splits_unchanged':True,'updated_unix':time.time()}
    (ROOT/'results/news_augmentation_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
