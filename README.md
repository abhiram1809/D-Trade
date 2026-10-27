# D-Trade: bounded decision models for market forecasting

A reproducible experiment comparing open decision models below **800 million parameters**, building historical text-and-price examples, fine-tuning an original decision head on the local AMD GPU through **Vulkan**, and evaluating on a newer chronological period.

**Run status is in `results/run_status.json`.** A running worker is not a finished experiment. `results/summary.json` and `results/REPORT.md` are created only after the measured fine-tuning and test stages finish.

Read [the model research](docs/DECISION_MODELS.md), [the predeclared experiment protocol](docs/PROTOCOL.md), and [the dataset specification](docs/DATASET.md).

## Data and prediction task

- 34 deliberately chosen instruments: 12 US, 8 Indian, 8 other global equities, and 6 FX pairs.
- Yahoo daily OHLCV, dividends and splits; source snapshots and hashes retained.
- More than 10,000 distinct AAPL training windows. The complete dataset stores exactly 100 candles per example.
- Model text contains all 100 daily close-to-close returns in integer basis points, trailing volatility, and available timestamped headlines. Full OHLCV remains in the dataset and numeric comparator.
- Target: **next trading session's open-to-close return**, represented by five volatility-scaled ranges and down/flat/up probabilities. This excludes the overnight gap.
- Training targets before 2024; validation in 2024; testing from 2025 to the frozen October 2026 cutoff. Missing historical news is explicit.

## Models and GPU scope

Candidates are pinned **Laya** (421M) and **d3-edge** (594.6M including its readout). Validation five-bin log loss selects the practical candidate; this does not prove a globally best model. Jev's proprietary architecture/size cannot be verified for the local parameter-constrained fine-tune.

The backbone stays frozen. Native Vulkan computes the final scorer/readout forward pass, softmax gradients, gradient reduction and Adam weight updates. This is genuine **head-only parameter-efficient fine-tuning**, not full-backbone training or LoRA. Frozen feature extraction also offloads supported linear layers through Vulkan; attention and other operations remain in PyTorch CPU. GPU forward/gradient/update parity checks are saved in `results/vulkan_parity.json`.

The RX 560 is old enough that feature extraction is a long-running stage. Per-example representations use file and directory fsync around atomic writes. On startup, the worker checks cache contents and quarantines damaged files for regeneration. Rerunning the worker resumes valid completed representations, and a process lock prevents duplicate GPU workers.

For this checkout at `~/Documents/ChatGPT/D-Trade`, `ops/d-trade.service` runs the pipeline as a user service. Install it under `~/.config/systemd/user/`, run `systemctl --user daemon-reload`, then `systemctl --user enable --now d-trade.service`. It resumes after the user logs in and retries process failures up to three times per hour. Stop an existing manually launched worker before starting the service; its GPU lock prevents concurrent runners. Inspect it with `systemctl --user status d-trade.service` and `journalctl --user -u d-trade.service`. A completed experiment with a matching dataset is not retrained on subsequent logins. Adapt the two paths in the unit for another checkout location.

## Reproduce

Python 3.14 and a working Vulkan driver were used on this host. Exact Python versions are in `requirements.lock.txt`; install the CPU PyTorch wheel separately using its official index when creating a fresh environment.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
bash scripts/build_vulkan.sh
.venv/bin/python scripts/vulkan_scorer.py
.venv/bin/python scripts/download_models.py
.venv/bin/python scripts/fetch_data.py --prices-only
.venv/bin/python scripts/fetch_news_parquet.py
.venv/bin/python scripts/fetch_recent_archive.py
.venv/bin/python scripts/prepare.py
.venv/bin/python scripts/run_pipeline.py
```

`build_vulkan.sh` uses `glslc`, a C++ compiler, Vulkan headers and `libvulkan.so.1`. Override `VULKAN_HEADERS` with a local include path. On this machine the script reuses existing headers from the adjacent Timestral experiment. The linear-offload implementation was adapted from that local project; its provenance is preserved in `docs/IMPLEMENTATION.md`.

After completion, the worker writes validation comparisons, exact adapter weights/config, per-origin test probabilities, per-market/per-symbol metrics, cost-sensitive AAPL backtests, a paired block bootstrap and a readable report. To apply an adapter to a documented JSON example:

```bash
.venv/bin/python scripts/predict.py --input example.json --adapter models/laya-market-head
```

Use the selected model's actual adapter directory. A bare input has `{"input":{"state":"...\nr_bp=...100 integer returns..."}}`; exported dataset records already contain this structure. Predictions reject a sequence with fewer or more than 100 observations.

Historical retrospective scores can overlap unknown foundation-model pretraining data. News timestamps, survivorship, limited universe coverage and illustrative execution assumptions constrain interpretation. The experiment reports measured evidence rather than asserting profitable trading.
