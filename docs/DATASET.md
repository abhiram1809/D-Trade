# Dataset specification

Frozen research cutoff: 25 October 2026, Asia/Kolkata. Dates before that cutoff only. Details and actual counts: `data/processed/manifest.json`.

## Files

| File | Contents |
|---|---|
| `data/raw/prices/SYMBOL.parquet` | Original Yahoo daily bars, adjusted close, corporate actions and source timestamp |
| `data/raw/prices/SYMBOL.json` | Symbol, market, timezone, currency, retrieval time and SHA-256 |
| `data/raw/price_manifest.json` | All requested instruments and explicit fetch failures |
| `data/raw/news/*_fnspid.jsonl` | Timestamped real headlines, source links and publisher where available |
| `data/raw/news/*_status.json` | Source revision, date span, row count and retrieval/completion status |
| `data/raw/news/*_yahoo_current.json` | Current-news endpoint snapshot, including empty responses; never backfilled into old dates |
| `data/processed/examples.parquet` | Input text, target labels, dates, provenance and news coverage |
| `data/processed/arrays.npz` | Aligned 100×5 OHLCV windows, 100 integer returns, and a complete numeric comparator representation |
| `data/processed/aapl_train.jsonl.gz` | Detailed text-plus-number examples for every distinct AAPL training origin |
| `data/processed/evaluation_indices.json` | Fixed, evenly spaced multi-market evaluation sample selected before scores |

## Example table fields

`id` is `SYMBOL:origin_date`, unique per instrument/session. `origin_utc` is a conservative post-close information cutoff. The raw source column named `close_utc` preserves Yahoo’s bar-index timestamp; it is not an independently verified exchange closing time. FX daily-session boundaries are vendor-defined and are not treated as executable venue timestamps. `target_date` is the next observed market session, not necessarily the next calendar day. `split` is based on the target date. `n_points` is always 100. `window_start` and `window_end` identify the historical span.

`state` contains the symbol/market, 100 ordered daily close-to-close returns rounded to integer basis points, trailing sigma, explicit news coverage and at most three timestamped headlines. `news_json` preserves selected evidence with salience and availability metadata. `news_coverage_json` distinguishes a known archive span from unknown coverage; neither guarantees a complete record of real events.

`label` is in 0–4 and `label_name` is `large_down`, `down`, `flat`, `up`, or `large_up`. `target_log_return = log(next_close / next_open)`. `sigma` is the sample standard deviation of the 100 known close-to-close returns, floored at 0.001. Bin edges are `[-1, -.25, .25, 1] × sigma`, with the lower edge included in the upper bin. Targets are stored separately and are never inserted into the model state.

The OHLCV array uses adjusted OHLC for comparability over splits/dividends and original reported volume. Numeric features divide prices by the last context close and center log-volume within the window. Future common adjustment factors cancel in return and relative-price features. Absolute adjusted historical price levels are deliberately removed from model prompts. No test-period fitted normalization is used.

## News handling

FNSPID headlines are obtained from a pinned parquet mirror using HTTP range reads of only needed columns. The original source and mirror revision are preserved. Headline-level deduplication is conservative. Scraped publication timestamps are not independently verified; availability is delayed 24 hours, and the lookback is 72 hours.

Salience is a transparent regex heuristic, not a labeled oracle. Material-event terms take precedence over price-recap terms. The model is asked to consider magnitude, surprise, credibility and horizon. Unknown cases retain uncertainty. No future price movement is used to choose or label news relevance.

Headline text also includes publication-timestamp age and conservative-availability age, both computed from that example's historical forecast origin. The current wall clock is never used to compute these fields. This provides recency context without adding an absolute forecast date that could encourage historical date memorization. Timestamp precision remains uncertain, and dated reports may already be priced in.

The FNSPID archive spans roughly 2020–2023 and seven US tickers. A second pinned archive, defeatbeta/yahoo-finance-data, adds 2025–2026 Yahoo headlines for all twelve US instruments. Its dates have no exact intraday time: publication is bounded at 23:59:59 UTC of the reported date, then availability is delayed a further 24 hours. The stored time is a conservative bound, not an asserted true publication instant. Gaps, early 2025, 2024, and non-US historical news remain unknown. A numeric narrative is text, but must not be misrepresented as an independent news report. The dataset has over 10,000 real AAPL market observations/windows, not a claim of 10,000 independently timestamped AAPL news events.

## Evaluation sample

All AAPL training windows and every AAPL validation/test session are used for the selected head. Cross-market model comparison uses 12 evenly spaced 2024 validation origins per instrument; cross-market testing uses 24 evenly spaced later origins per instrument, plus the complete AAPL test series. These are small diagnostic samples outside AAPL; no universal market superiority follows.

The overlapping context is intentional. Statistical uncertainty uses blocks instead of pretending every neighboring origin is independent. Source selection includes survivors and does not reconstruct historical index membership. FX volume and currencies are not directly comparable to equity exchange volume or a unified-currency portfolio.
