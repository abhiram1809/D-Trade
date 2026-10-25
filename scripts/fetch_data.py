"""Cache auditable Yahoo daily bars and timestamped news; no synthetic replacement."""
import concurrent.futures as cf
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config.json').read_text())
RAW = ROOT / 'data/raw'


def fetch_one(item):
    market, symbol = item
    target = RAW / 'prices' / (symbol + '.parquet')
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {'symbol': symbol, 'market': market, 'retrieved_utc': datetime.now(timezone.utc).isoformat()}
    try:
        if target.exists():
            frame = pd.read_parquet(target)
            meta_file = target.with_suffix('.json')
            if meta_file.exists():
                record.update(json.loads(meta_file.read_text()))
        else:
            ticker = yf.Ticker(symbol)
            frame = ticker.history(start='1960-01-01', end=CFG['as_of_exclusive'], interval='1d', auto_adjust=False, actions=True, raise_errors=True)
            if frame.empty:
                raise ValueError('Empty history')
            record['timezone'] = str(frame.index.tz)
            try:
                record['currency'] = ticker.history_metadata.get('currency')
            except Exception:
                record['currency'] = None
            frame = frame.reset_index()
            frame['date'] = pd.to_datetime(frame['Date']).dt.strftime('%Y-%m-%d')
            frame['close_utc'] = pd.to_datetime(frame['Date'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            frame.to_parquet(target, index=False)
        frame = frame[frame['date'] < CFG['as_of_exclusive']]
        record.update(rows=len(frame), first=frame['date'].min(), last=frame['date'].max(), sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        target.with_suffix('.json').write_text(json.dumps(record, indent=2))
        print(symbol, record['rows'], record['first'], record['last'], flush=True)
    except Exception as exc:
        record['error'] = repr(exc)
        print(symbol, 'FAILED', repr(exc), flush=True)
    return record


def fetch_news(symbol='AAPL'):
    """HF filter avoids downloading the entire 30GB FNSPID corpus."""
    folder = RAW / 'news'
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f'{symbol}_fnspid.jsonl'
    status_path = folder / f'{symbol}_fnspid_status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('complete'):
        return
    offset = 0
    if out.exists():
        offset = len(out.read_text().splitlines())
    with out.open('a') as f:
        while True:
            params = {'dataset': 'Zihan1004/FNSPID', 'config': 'default', 'split': 'train', 'where': f'"Stock_symbol"=\'{symbol}\'', 'offset': offset, 'length': 100}
            response = None
            for attempt in range(3):
                try:
                    response = requests.get('https://datasets-server.huggingface.co/filter', params=params, timeout=60)
                    response.raise_for_status()
                    payload = response.json()
                    break
                except Exception as exc:
                    if attempt == 2:
                        status_path.write_text(json.dumps({'complete': False, 'rows': offset, 'error': repr(exc), 'source': 'Zihan1004/FNSPID'}))
                        return
                    time.sleep(2 * (attempt + 1))
            rows = payload.get('rows', [])
            for row in rows:
                f.write(json.dumps(row['row'], ensure_ascii=False) + '\n')
            f.flush()
            offset += len(rows)
            total = payload.get('num_rows_total', offset)
            print('news', symbol, offset, '/', total, flush=True)
            if not rows or offset >= total:
                status_path.write_text(json.dumps({'complete': True, 'rows': offset, 'source': 'Zihan1004/FNSPID', 'retrieved_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
                return
            time.sleep(.15)


def current_news(symbol):
    target = RAW / 'news' / f'{symbol}_yahoo_current.json'
    if target.exists():
        return
    try:
        rows = yf.Ticker(symbol).get_news(count=30)
        target.write_text(json.dumps({'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'symbol': symbol, 'items': rows}, indent=2))
    except Exception as exc:
        target.with_suffix('.error.json').write_text(json.dumps({'error': repr(exc)}))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--news-only', action='store_true')
    parser.add_argument('--prices-only', action='store_true')
    args = parser.parse_args()
    if not args.news_only:
        items = [(m, s) for m, ss in CFG['symbols'].items() for s in ss]
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            records = list(pool.map(fetch_one, items))
        (RAW / 'price_manifest.json').write_text(json.dumps(records, indent=2))
    if not args.prices_only:
        for symbol in ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'JPM', 'XOM', 'KO', 'IBM']:
            fetch_news(symbol)
        for symbols in CFG['symbols'].values():
            for symbol in symbols:
                current_news(symbol)
