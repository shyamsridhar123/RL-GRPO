# AI Assistant Instructions for GRPO Training System

## Project Overview

This is a **CPU-optimized Group Relative Policy Optimization (GRPO) training system** for mathematical reasoning on consumer hardware. The project demonstrates feasibility of LLM fine-tuning without GPU requirements, targeting educational/research accessibility.

**Key Achievement**: 494M parameter model training on 14-core CPU in ~75 minutes with 3.32GB memory usage.

## Architecture & Critical Files

### Core Training Pipeline
- **`optimization/ultra_fast_training.py`** - Main training entry point with 12-core CPU optimization
- **`src/training/grpo_trainer.py`** - Core GRPO trainer using TRL library
- **`src/training/lightning_fisher.py`** - Fisher Information approximation (EWC memory)
- **`src/training/progressive_training.py`** - Multi-stage curriculum learning
- **`app.py`** - Gradio web interface for accessible experimentation

### Performance Optimization Components
- **`src/training/advanced_memory_optimization.py`** - Memory management for CPU constraints
- **`optimization/ultra_optimized_training.py`** - Combines Fisher + memory optimization
- **`src/agents/grpo_agent.py`** - GRPO algorithm implementation

## Development Patterns

### CPU-First Design Philosophy
All components assume **CPU-only training** with aggressive optimization:
```python
# Environment setup pattern (found in multiple files)
os.environ['OMP_NUM_THREADS'] = '12'  # Use physical cores
torch.set_num_threads(14)  # Use all logical cores
```

### Configuration Management
- **YAML configs** in `configs/` for training parameters
- **Dataclass configs** in Python for runtime parameters
- Example: `CPUGRPOConfig` in `grpo_trainer.py` with CPU-specific defaults

### Training Scale Reality
- **Small datasets**: 10-200 samples (not GPU-scale 10,000+)
- **Conservative parameters**: Learning rates 1e-5, batch size 1-2
- **Time constraints**: Target <300s total training time

## Key Workflows

### Run Main Training
```bash
python optimization/ultra_fast_training.py
```

### Launch Web Interface
```bash
python app.py  # Gradio interface on localhost:7860
```

### Run System Tests
```bash
python experiments/test_complete_ultra_optimized_system.py
python experiments/test_progressive_ultra_optimized.py
```

### Progressive Training (3-stage curriculum)
```bash
python src/training/progressive_training.py
```

## Critical Implementation Details

### Model Persistence Issues
**Watch for quantization serialization problems**:
- Use `safe_serialization=True` for model saves
- Fallback to `safetensors` if standard save fails
- Models in `models/*/` directories often have serialization workarounds

### Memory Management Pattern
```python
# Standard memory optimization pattern used throughout
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
psutil.virtual_memory()  # Monitor usage
```

### GRPO-Specific Requirements
- **Batch size must equal num_generations** (TRL requirement)
- **2+ generations required** for GRPO algorithm
- **CPU-only training** - all CUDA disabled

## Testing & Validation

### Experiment Structure
- **`experiments/`** - All validation and testing scripts
- **`experiments/results/`** - Benchmark outputs and analysis
- **`documentation/`** - Research findings and performance analysis

### Key Test Files
- `test_complete_ultra_optimized_system.py` - Full system integration
- `baseline_accuracy_validation.py` - Model accuracy benchmarks
- `test_progressive_ultra_optimized.py` - Progressive training validation

### Performance Benchmarks
- **GSM8K mathematical reasoning** - 50 problem evaluation set
- **Memory usage tracking** - Peak 3.32GB documented
- **Training throughput** - 0.004 samples/second baseline

## Integration Points

### TRL Library Integration
- Uses `GRPOTrainer` and `GRPOConfig` from TRL
- Custom CPU optimizations wrapped around TRL components
- Careful parameter mapping between custom configs and TRL

### Gradio Web Interface
- **`app.py`** integrates with `ultra_fast_training.py`
- Real-time training monitoring and model comparison
- Educational interface for non-technical users

### Experimental Framework
- Comprehensive logging and timing measurement
- Automated benchmark comparison between model variants
- Progressive training stages with rollback capabilities

## Common Pitfalls

1. **Path Issues**: Use `sys.path.append()` for cross-module imports
2. **Memory Leaks**: Always call `gc.collect()` after training
3. **Windows Paths**: Use `os.path.abspath()` for model saving
4. **Quantization**: Disable for debugging, enable for production
5. **CPU Threads**: Set early before PyTorch imports

## Project Goals & Constraints

**Mission**: Democratize AI training by proving CPU feasibility
**Hardware Target**: 14-core consumer CPU, 16GB RAM
**Time Constraint**: <6 hours for full training pipeline
**Accuracy Goal**: Maintain GSM8K reasoning capability through training

When working on this codebase, prioritize CPU optimization, memory efficiency, and educational accessibility over raw performance numbers.
