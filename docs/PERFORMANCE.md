# Runtime acceleration — 27 October 2026

The user requested maximum available compute. The Ryzen 5 3550H has four physical CPU cores/eight hardware threads, 24 GB RAM, a Radeon RX 560 and integrated Vega 8 graphics. The system was already in its performance power profile. Baloo desktop indexing was suspended temporarily; the user service restores it on exit. No overclocking was applied.

## Measurements

The original Vulkan linear offload is synchronous: each linear operation transfers activations, dispatches a tiled matrix product, waits for completion and returns to CPU attention/normalization. A profile found most wall time in these linear calls. Increasing CPU threads alone did not improve that path. Sixteen alternative GPU tile configurations offered only small changes and were not selected for production.

`results/d3_runtime_benchmark.json` compares CPU and original Vulkan extraction on three real cached contexts. CPU extraction with four threads took about 5.1 seconds/input; the original GPU path took about 15 seconds/input. Increasing a single CPU worker to eight threads and batching 2/4/8 contexts did not improve throughput. `results/d3_batch_benchmark.json` records padded-batch parity checks.

The concurrent comparison uses eight real contexts repeated three times, without using market labels or changing model inputs. `results/parallel_runtime_extended.json` records:

| Extraction workers | Seconds per input |
|---|---:|
| Two CPU workers, two threads each | 4.91 |
| Two CPU workers plus original Vulkan GPU extraction | 6.96 |
| Two CPU workers, four threads each | 4.95 |

These are short measured runs, not guarantees for every input length or background load. The first configuration was selected by throughput. It fills the four physical cores using independent contexts; additional hardware threads did not improve this comparison. The integrated GPU was not separately benchmarked. CPU/GPU extraction contention on this machine made the measured mixed setup slower.

## Preserved experiment

All contexts still contain the same 100 points and historical news. The model revision, question, target bins, chronological splits, stock universe and training settings are unchanged. FP32 CPU features agree with the existing Vulkan features: maximum observed feature difference 0.0000201 and decision-probability difference 0.0000021 in the extended comparison. The acceptance tolerance is 0.0001 for probabilities. This is floating-point parity, not bitwise identity. The benchmark has no financial-model selection authority.

`scripts/parallel_features.py` starts independent frozen models, validates each against the saved original readout, and dynamically assigns pending examples. Only missing or quarantined inputs are regenerated. New cache files record their extraction backend; existing valid Vulkan representations are retained. All writes use the durable atomic writer, and a worker failure aborts the phase for recovery through the user service.

Only frozen feature preparation moves to the measured CPU runtime. Head forward/backward/Adam operations still use the native Vulkan scorer on the RX 560. Four CPU threads are available for training-phase CPU work and numeric-baseline fitting. The saved adapter records `runtime_config.json` alongside its GPU training provenance.

## Reproduce the timing comparisons

Stop the production service before GPU benchmarks so simultaneous jobs do not contaminate timings. Use the same prepared dataset and cached reference vectors.

```bash
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_d3_runtime.py
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_d3_batches.py
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/parallel_features.py --benchmark --extended
```

The original GPU-thread comparison ran before the passive OpenMP wait setting; treat cross-script timing ratios as approximate. The longer concurrent configurations share the same process pool and passive wait setting. Production throughput and remaining time are written to `results/d3_feature_progress.json`.
