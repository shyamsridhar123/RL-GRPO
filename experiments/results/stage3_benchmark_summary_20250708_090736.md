# Stage 3 Progressive Training Model Benchmark Report

**Analysis Date:** 2025-07-08
**Session ID:** 20250708_090736
**Test Dataset:** GSM8K Mathematical Reasoning
**Problems Tested:** 50 of 50
**Training Context:** Stage 3 model used progressive training strategy

## Performance Results

**Model:** stage3_final
**Accuracy:** 0.040 (2/50 correct)
**Avg Generation Time:** 35.91 seconds
**Avg Tokens/Second:** 4.00
**Load Time:** 10.15 seconds
**Memory Used:** 0.18 GB

## Progressive Training Analysis

### Training Strategy Context
The stage3_final model was trained using a progressive training strategy:
- **Stage 1:** Basic reasoning task adaptation
- **Stage 2:** Intermediate complexity reasoning
- **Stage 3:** Advanced mathematical reasoning focus

### Comparison with Previous Models
Previous benchmark results (from benchmark_summary_20250708_064555.md):
- **Base Qwen:** 0.060 accuracy, 11.29 tokens/sec
- **Ultra_fast:** 0.000 accuracy, 10.83 tokens/sec
- **Extreme_fast:** 0.060 accuracy, 9.50 tokens/sec
- **Hardware_accelerated:** 0.000 accuracy, 7.77 tokens/sec

**Stage3_final performance:** 0.040 accuracy, 4.00 tokens/sec

[ANALYSIS] Progressive training needs further analysis

## Data Sources

- Detailed results: `stage3_benchmark_results_20250708_090736.json`
- Stage 3 training logs: `models/stage3/training.log`
- Training summary: `models/stage3/training_summary.json`
- Test dataset: GSM8K (grade school math problems)
- Evaluation metric: Exact numerical answer matching
- Benchmark integrity: Identical parameters to original benchmark

