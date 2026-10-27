# Completed experiment — 27 October 2026

The experiment is **complete and independently audited**. All 12,646 selected-model representations were checked. Vulkan fine-tuning used 10,752 distinct AAPL examples and 5,120 original readout weights; the selected checkpoint's 1,008 GPU updates were replayed independently with bitwise-identical weights. All 160 test metric rows were recomputed across 1,246 newer-period origins, including 454 continuous AAPL sessions. A fresh saved-adapter prediction matched its stored held-out probabilities within 0.0000004. The three result figures were visually reviewed. See `results/completed_run_audit.json`, `results/inference_audit.json` and `results/REPORT.md`.

Fine-tuning improved AAPL log loss from 1.5906 to 1.5348 but did not beat the frequency baseline (1.5268) or numeric logistic baseline (1.4980). The paired interval includes zero. All four forecast rules abstained under the frozen confidence gate, so zero backtest return means zero trades. The trained adapter is exported to `artifacts/d3-edge-market-head/`. The service exited normally and restored desktop indexing.

The sections below preserve historical progress and recovery notes; their pending-work statements describe earlier stages.

The user subsequently requested a replay without the directional confidence gate. `results/BACKTEST_HIGHEST_CONFIDENCE.md` records full allocation to the largest down/flat/up probability, with cash for flat or ties. The fine-tuned model takes 312 long and 4 short positions, stays cash on 138 sessions, and returns −24.98% after 5 bps per side (gross +2.92%; drawdown −38.43%). The original gated outputs and all forecasts are retained. All 19 tests pass; `results/highest_confidence_audit.json` independently checks all 1,816 model/session allocation rows.

## Recovery on 27 October

The machine booted at 01:36 IST on 26 October, within a minute of the previous worker's final progress entry. It stopped at 6,167 / 10,752 AAPL training representations and 6,815 / 12,646 total representations. The log had no recorded exception. A resumed runner on 27 October passed all 11 integrity tests, reused validation caches, selected the same d3-edge candidate, and continued from the saved inputs. Its initial throughput after restart is slower than the earlier run; use fresh progress rather than the previous ETA.

`ops/d-trade.service` provides a user service for this checkout. Its installed service resumes after login and restarts failed processes, with a limit of three starts per hour. Use `systemctl --user status d-trade.service` and live `results/run_status.json` PIDs to check it. An unchanged completed dataset is not trained again on a later login. The actual Vulkan fine-tuning and newer-period test are still pending at this milestone.

A full cache check after the reboot found five empty d3 files at indices 6162–6166. The remaining 6,820 checked representations passed shape, finiteness, token count and dataset-identity checks. Preserve the empty files under `results/cache/d3-edge/quarantine/` and regenerate those inputs. `scripts/feature_cache.py` now checks existing cache entries on startup, quarantines invalid ones, and uses file/directory fsync around atomic writes. `results/cache_resume_audit.json` preserves the initial failure evidence; subsequent recovery reports record quarantine and regeneration. This changes storage reliability, not model inputs or forecasting settings.

All five damaged inputs were regenerated successfully and independently validated; `results/reboot_cache_repair_verified.json` records this. All 15 integrity/storage tests passed. The service is actively extracting more inputs. Initial throughput after recovery is about 15 seconds/input, roughly twice as slow as the prior run, giving about 24 hours remaining at this milestone. Treat this as a changing throughput estimate, not a completion promise.

The user subsequently authorized maximum compute. Runtime profiling selected two independent CPU workers with two threads each for frozen extraction (4.91 seconds/input in the longer comparison). Adding GPU extraction was slower; head training still uses Vulkan. `runtime_config.json` and `docs/PERFORMANCE.md` record the measured configuration and FP32 parity. The service now starts `scripts/parallel_features.py` for the full d3 extraction phase. It increases the job's CPU scheduling weight and temporarily suspends/resumes Baloo indexing. Do not confuse CPU feature preparation with CPU optimizer training; the optimizer is unchanged and remains Vulkan.

At this earlier milestone, actual selected-model fine-tuning and newer-period evaluation were still pending. The completion evidence is now recorded at the top of this document.

## Authoritative current work

The dataset has 268,166 distinct 100-point examples across 34 instruments, with 10,752 AAPL training examples before 2024. The recent-news merge completed with 4,447 US test examples carrying eligible news. `results/news_augmentation_audit.json` verifies training/validation inputs unchanged and the 17 pre-existing cache inputs unchanged. All 11 integrity tests passed after the merge and recency annotation.

The previous process did not survive the earlier interruption. A detached sequential runner was started on this turn; find its PID in `results/runner_process.json`. `results/run_status.json` now includes both the runner and current child PID. Confirm those live with `ps` and matching command lines; a status file alone is not evidence of a running process. Do not restart on a timeout or stale observation without verifying termination. `results/pipeline.lock` prevents duplicate runners.

The worker phases are: Laya validation (408 fixed 2024 origins), d3 validation (same origins), candidate selection by validation log loss, complete selected-model representations (10,752 AAPL training rows + all AAPL later sessions + cross-market samples), Vulkan head fine-tuning, newer test evaluation, and final report. Expected runtime is many hours on this RX 560. Cache writes are atomic per input and are resumable. Do not reduce the training-example count, chronological splits, 100-point context, or original scope to make the run finish sooner.

## Evidence and cautions

- Research notes: `docs/DECISION_MODELS.md`; exact task and split policy: `docs/PROTOCOL.md`.
- Laya retains 421M parameters; d3-edge retains 594.6M including its code readout. Jev cannot be verified under the ceiling or fine-tuned locally.
- `results/vulkan_parity.json` demonstrates GPU forward, backward and Adam updates for 1,024/5,120-weight heads. This is a kernel check, **not** the requested full financial fine-tune.
- This implementation fine-tunes the existing final scorer/readout with the rest frozen; explicitly call it head-only fine-tuning, never full-backbone or LoRA training.
- `results/extraction_profile.json` verified identical Laya features with four CPU threads and modest speed improvement. A shared-memory padding experiment was slower and rejected; the original validated GPU shader remains in use.
- `scripts/audit_dataset.py` checks every target and label against all 34 raw sources, every 100-point prompt, news availability and the 10,752-row exported training file. The full audit passed: 268,166 targets/contexts, all 34 symbols, 24,070 selected-news records and 10,752 exported AAPL training examples. Authoritative evidence is `results/dataset_full_audit.json`.
- Conditional non-flat up/down metrics were added to `scripts/metrics.py` and prediction output before any test scoring. No model/prompt choice used test scores.
- News remains unknown for many non-US and early-period cases; date-only news uses conservative end-of-day +24h availability. Pretraining overlap makes the holdout retrospective, not a claim of prospective financial skill.

## Next actions

1. Check live runner and child, progress timestamps and cache counts. If a child failed, inspect that phase log and correct the specific problem; preserve completed valid representations.
2. If all processes are terminal, run `.venv/bin/python scripts/run_pipeline.py` detached with output appended to `results/pipeline.log`; keep the PID recorded. Its file lock rejects duplicates.
3. After actual completion, inspect `results/model_selection.json`, saved adapter config and changed weights, `training_history.json`, all test prediction rows, metrics, bootstrap and backtest outputs. Verify >=10,000 distinct AAPL training windows, 800M ceiling, 100 inputs, real Vulkan training and newer chronological test independently against underlying artifacts.
4. Exercise saved-adapter prediction on a real exported 100-point input, check probabilities/ranges, fix any genuine issues and write the final measured report.
5. Only then mark the goal complete, and give the user concise measured findings plus links to notes, dataset and report.

## Subsequent progress in the same live run

`results/news_recency_audit.json` records explicit headline age added to 8,756 news-bearing inputs, with all validation inputs and the 144 then-completed representations unchanged. The current dataset SHA-256 is `1033c942a1b3ebc240315b15cc988cf7c5c55a6a4759081fe5d591f2d765386a`. The full dataset audit was rerun successfully against that hash, including exact equality of exported training input text and the example table. The live worker did not need restarting.

Final-report rendering now creates three standard Matplotlib PNG/SVG figures from completed test predictions (forecast losses, continuous AAPL net equity and calibration), plus curve/bin CSV files. The plotting dependency is installed and locked. These figures have **not** been rendered or visually verified yet because the real test results are still pending; inspect them after completion before delivering the report. Training provenance now records dataset/training-ID hashes and Vulkan shader/library hashes. Coverage diagnostics are descriptive date groups, not evidence of a causal news benefit.

`results/context_values_audit.json` additionally proves exact source equality for every OHLCV window (134,083,000 values), every stored return window and all 26,816,600 integer return values in model prompts. The audit passed against the current dataset SHA. Training history now records actual successful GPU update counts, examples seen and unique example counts per epoch; the learning settings are unchanged. The saved-prediction interface returns an undefined conditional non-flat probability as null if all non-flat probability underflows to zero.

## First complete base-model result

Laya finished all 408 predetermined 2024 validation origins. `results/laya_base_validation.json` records raw five-bin accuracy 20.59%, log loss 1.6891 and Brier 0.8357; the AAPL training-frequency comparator has 38.73%, 1.4561 and 0.7421 on the same sample. These are validation diagnostics, not the newer held-out test and not a final model selection. Do not change prompts, data or selection settings in response to these scores.

The runner automatically advanced to d3 validation; its current PID is read from `results/run_status.json` and its progress from `results/d3_feature_progress.json`. Runner PID 6753 and d3 child PID 12649 were confirmed live at this milestone. Laya child 6797 completed normally; do not restart it merely because it is now absent.

`results/model_parameter_audit.json` counts every tensor element in the actual downloaded model files. Conservative runtime bounds (including retained old readout, adapter vector and extra temperature) are 421,294,855 for Laya and 594,587,841 for d3-edge. Both remain below 800M, including d3 vision weights.

## Candidate selection completed

Both 408-case validation passes finished normally. `results/model_selection.json` selects d3-edge by log loss: 1.5023 versus Laya 1.6891. The class-frequency control is better at 1.4561. `results/VALIDATION_REPORT.md` explains these measured results and the constant-flat five-range predictions; this is not evidence of a market edge. The 2025–2026 test is still unscored.

The runner advanced automatically to `selected_model_full_features`. Read its actual child PID from `results/run_status.json` (16799 at this milestone; runner 6753). The selected model is d3-edge. The full pass has 12,238 representations remaining after the first 408 validation caches. It first finishes the additional AAPL validation sessions, then all 10,752 training examples, then the newer test inputs. The initial measured throughput estimates about a day for this stage. Actual head fine-tuning and held-out scoring have NOT happened yet. Do not mark complete or blocked while this verified live process is making progress.
