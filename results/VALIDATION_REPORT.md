# Frozen base-model validation — 2024

408 predetermined daily forecast origins across 34 instruments. Each model receives the same 100 return observations, the same volatility-scaled next-session open-to-close question, and the same evidence-availability policy. This is a small convenience sample. The 2025–2026 holdout has not been scored.

| Model | Five-range accuracy | Three-class accuracy | Log loss | Brier score |
|---|---:|---:|---:|---:|
| laya | 20.59% | 31.86% | 1.6891 | 0.8357 |
| d3-edge | 38.73% | 33.33% | 1.5023 | 0.7510 |
| AAPL training class frequencies | 38.73% | 31.86% | 1.4561 | 0.7421 |

D3-edge is the selected foundation checkpoint because its five-range log loss is lower than Laya’s. The class-frequency comparator still has lower loss than either foundation model, so this comparison does not establish a market-prediction edge.

D3’s most likely five-range answers, ordered large down / down / flat / up / large up: **[0, 0, 408, 0, 0]**.
Its five-range accuracy comes entirely from always choosing the flat category; that accuracy should not be presented as evidence of learned return predictability. Aggregating range probabilities into up/down/flat probabilities can produce different three-class answers.

The class-frequency comparator uses only earlier AAPL training labels and provides no new market information. It is a control, not a claim of being an optimal forecasting model.

All three distributions have 52% directional accuracy conditional on non-flat outcomes in this sample. This coincides with the training-frequency control and does not establish a useful directional advantage.

Fine-tuning will update d3’s five existing answer-code readout rows (5,120 weights) using all 10,752 distinct AAPL training windows on native Vulkan. The backbone stays frozen. The newer-period performance remains unknown.

The full representation pass is running and is resumable per example. Its initial measured throughput suggests roughly a day of preparation on this RX 560; that estimate may change for longer news contexts.
