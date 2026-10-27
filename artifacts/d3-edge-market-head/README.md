# Trained d3-edge market head

This adapter contains 5,120 fine-tuned FP32 readout weights, not the frozen 594.6M-parameter base model. It was trained on 10,752 AAPL examples with native Vulkan on the Radeon RX 560. Download the pinned base using `scripts/download_models.py`, then run `scripts/predict.py --input results/inference_input.json --adapter artifacts/d3-edge-market-head`.

`weight.npy` is the normalized head used by prediction. The initial head and original-row safetensors export are retained for comparison. `adapter_config.json` records the base revision, training scope, calibration, hashes and measured device. Read `results/REPORT.md` for held-out results and limitations.
