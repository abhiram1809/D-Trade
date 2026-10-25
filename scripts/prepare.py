"""Generate causal 100-point examples and exports; labels are always separate."""
import gzip
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from news import NewsIndex, POLICY, render_news

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'config.json').read_text())
NAMES = ['large_down', 'down', 'flat', 'up', 'large_up']


def origin_time(day, market, timezone):
    # Conservative session completion (one hour after major cash-market close).
    hour, minute = {'US': (17, 0), 'India': (16, 30), 'Global': (18, 0), 'FX': (23, 59)}[market]
    return pd.Timestamp(f'{day} {hour:02}:{minute:02}', tz=ZoneInfo(timezone or 'UTC')).tz_convert('UTC')


def split_target(day):
    if day < CFG['train_end_exclusive']:
        return 'train'
    if day < CFG['validation_end_exclusive']:
        return 'validation'
    return 'test'


def numeric_features(window):
    """Fixed-width features consume ALL 100 candles; no fitted global statistics."""
    close = window[:, 3]
    # Input columns: relative open, high, low, close, relative log-volume.
    flat = window.reshape(-1)
    changes = np.diff(np.log(close))
    return np.r_[flat, [changes[-k:].mean() for k in (5, 20, 60)], [changes[-k:].std() for k in (5, 20, 60)], close[-1] / close[0] - 1].astype('float32')


def main():
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    news = NewsIndex(ROOT / 'data/raw/news')
    records, windows, returns, xs, manifest = [], [], [], [], []
    markets = {s: m for m, ss in CFG['symbols'].items() for s in ss}
    for symbol, market in markets.items():
        path = ROOT / 'data/raw/prices' / (symbol + '.parquet')
        if not path.exists():
            continue
        frame = pd.read_parquet(path).sort_values('date').drop_duplicates('date')
        before = len(frame)
        frame = frame[(frame['date'] < CFG['as_of_exclusive']) & (frame[['Open', 'High', 'Low', 'Close', 'Adj Close']] > 0).all(axis=1)].copy()
        frame = frame[frame['High'] >= frame['Low']].reset_index(drop=True)
        meta = json.loads(path.with_suffix('.json').read_text())
        # A per-day common adjustment factor cancels in intraday returns.
        factor = (frame['Adj Close'] / frame['Close']).to_numpy()
        prices = frame[['Open', 'High', 'Low', 'Close']].to_numpy() * factor[:, None]
        volume = frame['Volume'].fillna(0).to_numpy()
        logret = np.r_[np.nan, np.diff(np.log(prices[:, 3]))]
        count_start = len(records)
        for t in range(100, len(frame) - 1):
            target_date = str(frame['date'].iloc[t + 1])
            if target_date >= CFG['test_end_exclusive']:
                continue
            r = logret[t - 99:t + 1]
            if not np.isfinite(r).all():
                continue
            sigma = max(float(r.std(ddof=1)), .001)
            label_return = float(np.log(prices[t + 1, 3] / prices[t + 1, 0]))
            label = int(np.searchsorted(CFG['bin_edges_sigma'], label_return / sigma, side='right'))
            day = str(frame['date'].iloc[t])
            if market == 'FX' and (pd.Timestamp(day).dayofweek >= 5 or pd.Timestamp(target_date).dayofweek >= 5):
                continue
            origin = origin_time(day, market, meta.get('timezone'))
            selected, coverage = news.at(symbol, origin)
            # All features are constructed using observations <= t.
            raw_window = np.column_stack([prices[t - 99:t + 1], volume[t - 99:t + 1]])
            normalized = raw_window.copy()
            normalized[:, :4] /= prices[t, 3]
            normalized[:, 4] = np.log1p(volume[t - 99:t + 1]) - np.log1p(np.median(volume[t - 99:t + 1]))
            bps = np.rint(r * 10000).astype(int)
            narrative = f'{symbol} {market}. 100 daily close-to-close returns, oldest first, integer basis points. sigma={sigma:.5f}. News coverage={coverage["status"]}.'
            state = narrative + '\nr_bp=' + ','.join(map(str, bps))
            state += render_news(selected, origin)
            record = {'id': f'{symbol}:{day}', 'symbol': symbol, 'market': market, 'currency': meta.get('currency'), 'origin_date': day, 'origin_utc': origin.isoformat(), 'target_date': target_date, 'split': split_target(target_date), 'sigma': sigma, 'label': label, 'label_name': NAMES[label], 'target_log_return': label_return, 'state': state, 'n_points': 100, 'news_json': json.dumps(selected), 'news_coverage_json': json.dumps(coverage), 'window_start': str(frame['date'].iloc[t-99]), 'window_end': day}
            records.append(record)
            windows.append(raw_window.astype('float32'))
            returns.append(bps)
            xs.append(numeric_features(normalized))
        manifest.append({'symbol': symbol, 'market': market, 'valid_bars': len(frame), 'invalid_removed': before - len(frame), 'examples': len(records) - count_start})
        print(symbol, 'examples', len(records) - count_start, flush=True)
    table = pd.DataFrame(records)
    table.to_parquet(out / 'examples.parquet', index=False)
    np.savez_compressed(out / 'arrays.npz', windows=np.array(windows), returns_bp=np.array(returns, dtype='int32'), numeric=np.array(xs))
    aapl = table[(table.symbol == 'AAPL') & (table.split == 'train')]
    if len(aapl) < 10000:
        raise ValueError(f'Only {len(aapl)} AAPL training examples: requirement unsatisfied')
    with gzip.open(out / 'aapl_train.jsonl.gz', 'wt') as f:
        for idx, row in aapl.iterrows():
            obj = row.to_dict()
            obj['input'] = {'policy': POLICY, 'state': row.state, 'ohlcv_100': np.asarray(windows[idx]).tolist(), 'news': json.loads(row.news_json)}
            obj['target'] = {'bin': row.label_name, 'next_session_open_to_close_log_return': row.target_log_return}
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    report = {'config': CFG, 'universe': manifest, 'total_examples': len(table), 'aapl_training_examples': len(aapl), 'splits': table.groupby(['market', 'split']).size().to_dict().__str__(), 'news_coverage': news.coverage, 'news_matched_examples': int((table.news_json != '[]').sum()), 'examples_sha256': hashlib.sha256((out/'examples.parquet').read_bytes()).hexdigest()}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    # Deterministic evenly spaced validation/test sample per symbol, before scores.
    selections = []
    for (symbol, split), group in table[table.split != 'train'].groupby(['symbol', 'split']):
        count = 12 if split == 'validation' else 24
        selections.extend(group.iloc[np.unique(np.linspace(0, len(group)-1, min(count, len(group))).astype(int))].index.tolist())
    (out / 'evaluation_indices.json').write_text(json.dumps(selections))
    print(json.dumps({k:v for k,v in report.items() if k not in ['universe','config','news_coverage']}, indent=2))


if __name__ == '__main__':
    main()
