"""Supplement 2025–2026 US headlines from a pinned Yahoo-news archive mirror."""
import json
from pathlib import Path

import pyarrow.parquet as pq
import requests

from fetch_news_parquet import RemoteFile

ROOT=Path(__file__).resolve().parents[1]


def main():
    repo='defeatbeta/yahoo-finance-data'
    revision=requests.get(f'https://huggingface.co/api/datasets/{repo}',timeout=30).json()['sha']
    files=requests.get(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/data/US',timeout=30).json()
    file=next(f for f in files if f['path'].endswith('/stock_news.parquet'))
    raw=RemoteFile(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{file["path"]}',file['size'])
    parquet=pq.ParquetFile(raw)
    symbols=set(json.loads((ROOT/'config.json').read_text())['symbols']['US'])
    outputs={s:[] for s in symbols}
    symbol_col=parquet.schema.names.index('symbol')
    columns=['uuid','symbol','title','publisher','report_date','link']
    for group in range(parquet.num_row_groups):
        stats=parquet.metadata.row_group(group).column(symbol_col).statistics
        if stats and stats.has_min_max and not any(stats.min<=s<=stats.max for s in symbols):continue
        frame=parquet.read_row_group(group,columns=columns).to_pandas()
        for row in frame[frame.symbol.isin(symbols)].itertuples():
            if not ('2025-01-01'<=row.report_date<'2026-10-25'):continue
            outputs[row.symbol].append({'Date':row.report_date+' 23:59:59 UTC','published_date':row.report_date,'Article_title':row.title,'Stock_symbol':row.symbol,'Url':row.link,'Publisher':row.publisher,'source':'defeatbeta_yahoo_archive','timestamp_quality':'date_only_end_of_day_plus_24h','publication_time_precision':'date','uuid':row.uuid})
        print('row_group',group,'rows',sum(map(len,outputs.values())),'download_MB',round(raw.downloaded/1e6,2),flush=True)
    folder=ROOT/'data/raw/news'
    for symbol,rows in outputs.items():
        (folder/f'{symbol}_supplement.jsonl').write_text(''.join(json.dumps(r,default=str)+'\n' for r in rows))
    manifest={'repo':repo,'revision':revision,'file':file['path'],'file_size':file['size'],'bytes_downloaded':raw.downloaded,'counts':{s:len(rows) for s,rows in outputs.items()},'publication_time_precision':'date_only; conservative 23:59:59UTC plus 24h availability delay','first':min(r['published_date'] for rows in outputs.values() for r in rows),'last':max(r['published_date'] for rows in outputs.values() for r in rows)}
    (folder/'recent_archive_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
