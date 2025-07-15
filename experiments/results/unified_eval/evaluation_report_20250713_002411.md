# Unified Progressive Training Evaluation Report

**Date:** 2025-07-13 00:24:11

**Base Model:** Qwen/Qwen2-0.5B-Instruct
**Trained Model:** Qwen/Qwen2-0.5B-Instruct + Unified Progressive Training

## Overall Performance

| Metric | Base Model | Trained Model | Improvement |
|--------|------------|--------------|-------------|
| Correctness Score | 0.87 | 0.89 | 0.02 |
| Step-by-Step Reasoning | 71.7% | 80.0% | 8.3% |
| Average Calculation Steps | 17.6 | 19.1 | 1.5 |
| Relative Improvement | - | - | 2.3% |

## Performance by Difficulty Level

### Basic Problems

| Metric | Base Model | Trained Model | Improvement |
|--------|------------|--------------|-------------|
| Correctness Score | 0.87 | 0.85 | -0.02 |
| Step-by-Step Reasoning | 70.0% | 75.0% | 5.0% |
| Average Steps | 2.1 | 3.2 | 1.1 |

### Intermediate Problems

| Metric | Base Model | Trained Model | Improvement |
|--------|------------|--------------|-------------|
| Correctness Score | 0.89 | 0.89 | -0.00 |
| Step-by-Step Reasoning | 70.0% | 80.0% | 10.0% |
| Average Steps | 2.7 | 3.5 | 0.8 |

### Advanced Problems

| Metric | Base Model | Trained Model | Improvement |
|--------|------------|--------------|-------------|
| Correctness Score | 0.85 | 0.93 | 0.08 |
| Step-by-Step Reasoning | 75.0% | 85.0% | 10.0% |
| Average Steps | 4.8 | 3.8 | -1.0 |

## Conclusion

The Unified Progressive Training shows moderate improvement over the base model, with better performance in some areas.
