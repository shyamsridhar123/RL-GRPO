# Stage 3 Progressive Training Model Benchmark Report

**Analysis Date:** 2025-07-08
**Session ID:** 20250708_082926
**Test Dataset:** GSM8K Mathematical Reasoning
**Problems Tested:** 50 of 50
**Training Context:** Stage 3 model used progressive training strategy

## Performance Results

**Model:** stage3_final
**Accuracy:** 0.040 (2/50 correct)
**Avg Generation Time:** 12.94 seconds
**Avg Tokens/Second:** 12.72
**Load Time:** 3.97 seconds
**Memory Used:** 0.06 GB

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

**Stage3_final performance:** 0.040 accuracy, 12.72 tokens/sec

