# GRPO Training System Utility Scripts

This directory contains utility scripts for the CPU-optimized Group Relative Policy Optimization (GRPO) training system.

## Available Scripts

### 1. Monitor Training (`monitor_training.py`)

Monitors system resources during training runs.

**Usage:**
```bash
python scripts/monitor_training.py --output ./logs/monitoring --interval 5
```

**Features:**
- Tracks CPU usage, memory consumption, and process statistics
- Saves monitoring logs in JSON format for later analysis
- Configurable sampling interval

### 2. Visualize Results (`visualize_results.py`)

Generates visualizations from training logs and metrics.

**Usage:**
```bash
python scripts/visualize_results.py --log-dir ./logs --output-dir ./documentation/results
```

**Features:**
- Creates training loss plots over time
- Visualizes resource usage from monitoring logs
- Generates PNG images for inclusion in documentation

### 3. Checkpoint Manager (`checkpoint_manager.py`)

Manages model checkpoints for better organization and disk space efficiency.

**Usage:**
```bash
# List all checkpoints
python scripts/checkpoint_manager.py list

# List checkpoints from a specific stage
python scripts/checkpoint_manager.py list --stage stage3

# Create a backup of a checkpoint
python scripts/checkpoint_manager.py backup ./models/stage3/final_model

# Clean up old checkpoints (dry run by default)
python scripts/checkpoint_manager.py cleanup --keep 2

# Actually delete old checkpoints
python scripts/checkpoint_manager.py cleanup --keep 2 --execute

# Convert a model to SafeTensors format
python scripts/checkpoint_manager.py convert ./models/stage3/final_model
```

**Features:**
- Lists all checkpoints with metadata
- Creates backups before making changes
- Cleans up old checkpoints while keeping the N most recent ones
- Converts PyTorch models to SafeTensors format for better compatibility

### 4. Memory Profiler (`memory_profiler.py`)

Profiles memory usage to help identify and fix memory leaks or excessive usage.

**Usage:**
```bash
# Run in automatic mode
python scripts/memory_profiler.py --output ./logs/memory_profiles

# Run in interactive mode
python scripts/memory_profiler.py --output ./logs/memory_profiles --interactive
```

**Features:**
- Takes detailed memory snapshots during training
- Identifies top memory consumers
- Compares snapshots to detect memory leaks
- Provides optimization recommendations

## Common Workflows

### Training with Resource Monitoring

```bash
# Start monitoring in a separate terminal
python scripts/monitor_training.py &

# Run training
python optimization/ultra_fast_training.py

# Stop monitoring with Ctrl+C
```

### Cleanup and Organization

```bash
# Backup important models
python scripts/checkpoint_manager.py backup ./models/stage3/final_model

# Clean up old checkpoints
python scripts/checkpoint_manager.py cleanup --keep 2 --execute

# Convert models to SafeTensors for better compatibility
python scripts/checkpoint_manager.py convert ./models/stage3/final_model
```

### Post-Training Analysis

```bash
# Generate visualizations
python scripts/visualize_results.py --log-dir ./logs

# Examine memory usage patterns
python scripts/memory_profiler.py --interactive
```

## Best Practices

1. Always monitor system resources during training to identify bottlenecks
2. Regularly clean up old checkpoints to conserve disk space
3. Create backups before making any significant changes
4. Use the memory profiler when experiencing unexpected memory issues
5. Generate visualizations for documentation and analysis
