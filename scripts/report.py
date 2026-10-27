"""Human-readable measured results; fails if the real run has not finished."""
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    summary=json.loads((ROOT/'results/summary.json').read_text())
    data=json.loads((ROOT/'data/processed/manifest.json').read_text())
    selection=json.loads((ROOT/'results/model_selection.json').read_text())
    metrics=pd.read_csv(ROOT/'results/forecast_metrics.csv')
    primary=metrics[metrics.group=='AAPL']
    c=summary['checkpoint']
    lines=['# Measured decision-model market experiment','',f"Selected **{c['model']}**, {c['base_parameters']:,} parameters. Selection used a predetermined 2024 validation sample, not test returns.",'',f"Dataset: **{data['total_examples']:,}** 100-point examples across {len(data['universe'])} instruments. **{data['aapl_training_examples']:,} distinct AAPL training examples**. Historical headlines are available only for a subset of US instruments; all coverage gaps are explicit.",'',f"Fine-tuning changed **{c['trainable_parameters']:,} original scorer/readout weights** on **{c['device']} through Vulkan**. The backbone remained frozen. Training ends {c['training_target_last']}; test targets span {summary['test_target_first']} to {summary['test_target_last']}.",'','| Model | AAPL observations | Five-bin accuracy | Three-class accuracy | Log loss | Brier |','|---|---:|---:|---:|---:|---:|']
    for r in primary.itertuples():lines.append(f'| {r.model} | {r.n} | {r.five_bin_accuracy:.3f} | {r.three_class_accuracy:.3f} | {r.log_loss:.4f} | {r.brier:.4f} |')
    baseline=float(primary[primary.model=='class_frequency'].log_loss.iloc[0]);tuned=float(primary[primary.model=='fine_tuned'].log_loss.iloc[0])
    lines+=['',('The fine-tuned model improved' if tuned<baseline else 'The fine-tuned model did not improve')+f' AAPL log loss versus the training-frequency baseline ({tuned:.4f} versus {baseline:.4f}). This is the primary predictive comparison.']
    boot=summary['bootstrap_fine_tuned_vs_frequency_AAPL'];lines+=['',f"Paired block-bootstrap loss difference (fine-tuned minus baseline): {boot['mean_log_loss_difference_a_minus_b']:.4f}, 95% interval [{boot['ci95'][0]:.4f}, {boot['ci95'][1]:.4f}]. Negative favors fine-tuning.",'','## Illustrative continuous AAPL intraday backtest','', '| Model | Net return | Worst drawdown | Active fraction | Doubled-cost return |','|---|---:|---:|---:|---:|']
    if boot['ci95'][0]<=0<=boot['ci95'][1]:
        lines.insert(lines.index('## Illustrative continuous AAPL intraday backtest')-1,'The interval includes zero: this test does not provide clear evidence of a predictive advantage over class frequencies.')
    trading=json.loads((ROOT/'results/trading.json').read_text())
    for model,r in trading.items():
        a=r['AAPL_standard_cost'];s=r['AAPL_double_cost'];lines.append(f"| {model} | {a['total_return']:.2%} | {a['worst_drawdown']:.2%} | {a['active_fraction']:.2%} | {s['total_return']:.2%} |")
    if 'feature_runtime' in c:
        lines+=['','Frozen features used the benchmark-selected FP32 CPU/Vulkan runtime in `runtime_config.json`; parity checks against the original cached encoder outputs are recorded in the runtime benchmark files. Weight updates use the native Vulkan scorer.']
    lines+=['','## Interpretation and limits','']+[f'- {x}.' for x in summary['limitations']]+['- Reported backtests are open-to-close synthetic exposures with explicit costs, not broker executions.','- Missing archive news does not mean no market-moving event occurred.','- The noise filter is heuristic salience metadata, not a human-verified relevance dataset.','- A correct typed answer format is not evidence of financial accuracy.','','Full per-market/per-symbol results: `forecast_metrics.csv`. Per-origin probabilities and labels: `test_predictions.parquet`. Parameter updates and exact scope: the saved `adapter_config.json`. Vulkan forward/backward/update checks: `vulkan_parity.json`.','','## Validation candidate comparison','','| Candidate | Parameters | Validation log loss |','|---|---:|---:|']
    for r in selection['comparisons']:lines.append(f"| {r['model']} | {r['parameters']:,} | {r['log_loss']:.4f} |")
    from plot_results import render
    figures=render()
    lines+=['','## Measured test figures','']
    for figure in figures:
        label=Path(figure).stem.replace('_',' ')
        lines+=[f'![{label}]({figure})','']
    (ROOT/'results/REPORT.md').write_text('\n'.join(lines)+'\n')
    print(ROOT/'results/REPORT.md')


if __name__=='__main__':main()
