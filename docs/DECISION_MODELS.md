# Decision models and the market experiment

Research date: **25 October 2026**, Asia/Kolkata. Statements about released models refer to pinned snapshots in `vendor/*/metadata.json`, not an assumption that today's latest checkpoint will remain unchanged.

## Why the recent interest

Many applications need a bounded judgment—route a request, choose a tool, decide whether evidence supports a claim—rather than a paragraph. A decision model maps a state and a typed question directly to probabilities over allowed answers. Code can use the distribution without parsing generated prose. The technique extends classification and representation learning; the current product category emphasizes runtime-specified choices, calibrated uncertainty and integration into ordinary programs.

TypeSafe announced **Jev on 15 September 2026**, describing reinforcement learning for calibrated decisions (RLCD) and parallel structured outputs. Its speed and quality comparisons are vendor results; they do not establish forecasting ability. Source: [TypeSafe announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev?render=full).

Laya's open release and vLLM Semantic Router's Decision 3.0 family give developers inspectable alternatives. Public benchmark comparisons quickly followed. Adoption counts from aggregators are volatile and are deliberately not used as evidence of model quality.

## How they work

A common interface has:

- **State**: text, JSON, numeric observations or other evidence.
- **Question**: instructions and named alternatives, an ordered rubric, or a yes/no proposition.
- **Readout**: a learned score for each allowed answer, normalized into probabilities.
- **Application policy**: thresholds, abstention, review and downstream actions.

For logits z and temperature T, `p_i = exp(z_i/T) / sum_j exp(z_j/T)`. A higher maximum probability is not automatically a trustworthy estimate of correctness. Calibration must be checked on the target distribution.

**Laya.** The English checkpoint has **421,293,830 parameters**, confirmed by the Hugging Face safetensors metadata. Its implementation uses a bidirectional ModernBERT encoder, two Transformer head layers, and a shared scorer at per-option mask markers. It also has a separate act/escalation head. The last scorer layer is a 1,024-dimensional linear projection shared across choices; this is the parameter-efficient fine-tuning target in this project. The released English wrapper defaults to 512 tokens even though the encoder supports longer contexts. We must explicitly extend the runtime limit and verify complete input preservation. The multilingual release is about 322M with a longer default input budget. Model-specific domain training matters; general workflow scores are not finance evidence. [Model card](https://huggingface.co/convaiinnovations/laya), [author implementation](https://github.com/NandhaKishorM/laya).

**d3-edge.** The released model contains **594,321,600 backbone parameters**, plus a small 255-code readout, and fits the 800M ceiling. Its Qwen3.5-based package includes vision parameters even for this text-only task. The checkpoint config specifies a noncausal full-attention text path. A typed question is rendered with answer codes; the final prompt representation is projected onto those codes, and the applicable scores are normalized. Multiple questions in one API request can require separate forward passes. Its advertised accelerator latency is measured on an MI325X, not an RX 560. We use the official pinned rendering and runtime rather than treating ordinary next-token generation as equivalent. [Model card](https://huggingface.co/vllm-sr/d3-edge), [vLLM structured decisions documentation](https://docs.vllm.ai/en/latest/serving/online_serving/structured_decisions/).

**Jev.** This is a hosted proprietary service returning typed choices, ordered scores and yes/no probabilities. Its exact architecture, weights and parameter count cannot be independently inspected here. RLCD and calibration are developer descriptions, not guarantees on financial time series. It is unsuitable for the required local Vulkan fine-tune and cannot be verified under 800M parameters. It remains relevant background, not a locally measured candidate. [Official announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev?render=full), [official documentation](https://docs.typesafe.ai/).

## What RLCD changes—and what it cannot guarantee

A proper scoring rule rewards reporting the actual conditional probability in expectation. For log scoring, the reward is `log p(observed outcome)`; minimizing cross-entropy is the corresponding supervised objective. Ranked probability scores additionally respect ordered choices. Training on such objectives can encourage useful uncertainty rather than confident-sounding prose.

The mathematical optimum assumes appropriate data, sufficient capacity and successful optimization. Distribution shift, label noise, clipping, limited samples, option ordering and mis-specified questions still matter. Structured outputs prevent arbitrary output strings; they do not prevent wrong decisions, misleading probabilities or harmful actions.

The independent [matched decision-gate study](https://arxiv.org/abs/2610.00346) reports that label-trained classifiers can outperform general decision models and that benchmark exposure can explain apparent advantages. A [paired Laya/Jev audit](https://arxiv.org/abs/2610.02267) finds task-dependent gaps and sensitivity to option order, while also documenting mistakes in its own original analysis. Neither study evaluates the forecasting experiment here.

## Why markets are harder than classification demos

Next-session prices depend on information absent from a candle window, changing regimes, liquidity and other participants' expectations. A material announcement may already be priced in. A dramatic headline may have little incremental effect. Even reliable sentiment extraction is a different task from predicting returns.

The prompt therefore asks for novelty, credibility, surprise, magnitude and the forecast horizon. Disasters, financing, regulation, earnings/guidance and innovations are candidates for analysis, not a whitelist of guaranteed price movers. Promotional headlines and analyst recaps receive low priority. Missing news is marked unknown rather than interpreted as reassuring silence. See `scripts/news.py` for the exact reproducible policy and its deliberately heuristic salience metadata.

## Candidate choice and limitations

The experiment evaluates open candidates under the actual hardware and input constraints. Validation scores, local latency and measured parameter counts determine the practical choice. There is no basis for claiming the globally best sub-800M model from a small convenience sample. Price, momentum and class-frequency baselines must accompany results.

The chosen daily interval supports >10,000 AAPL windows and long market coverage. [yfinance documents limited intraday retention](https://github.com/ranaroussi/yfinance/blob/main/yfinance/multi.py), making minute candles a poor primary source for this historical sample. Historical US news comes from [FNSPID](https://github.com/zGeneral/fnspid) through a pinned Nasdaq-100 parquet mirror. A second pinned [Yahoo-news archive](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data) supplies 2025–2026 US headlines. Its date-only timestamps use conservative end-of-day plus 24-hour availability. These are scraped/mirrored records, not audited original publication timestamps; timing uncertainty remains. Outside available archive coverage, historical news is unknown.

A 2025–2026 holdout is newer than the fine-tuning examples, but can overlap the foundation checkpoint's unknown pretraining corpus. This is a retrospective experiment. It cannot establish prospective trading skill without future observations collected after the model and evaluation policy are frozen.
