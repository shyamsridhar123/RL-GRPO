#!/usr/bin/env python3
"""
Scaled-Up Training and Evaluation Runner

This script runs the scaled-up unified progressive training followed by
a comprehensive evaluation against the base model, with detailed metrics
and visualizations to demonstrate the effectiveness of the training approach.

Usage:
    python run_scaled_training_and_eval.py
"""

import os
import sys
import time
import subprocess
import psutil
import torch
import gc
from datetime import datetime

# Fix console encoding for emoji support on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='backslashreplace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='backslashreplace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def print_header(message):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)

def optimize_environment():
    """Set up environment for optimal CPU performance"""
    # Configure CPU threads
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    
    torch.set_num_threads(logical_cores)
    
    # Environment variables
    os.environ['OMP_NUM_THREADS'] = str(physical_cores)
    os.environ['MKL_NUM_THREADS'] = str(physical_cores)
    os.environ['NUMEXPR_NUM_THREADS'] = str(physical_cores)
    
    # Enable Intel MKL-DNN if available
    if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True
    
    # Extra optimization for Intel CPUs
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    
    print(f"[OK] Environment optimized with {logical_cores} logical cores")
    
    # Clean memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"[OK] Initial memory usage: {psutil.Process(os.getpid()).memory_info().rss / (1024**2):.2f} MB")

def run_scaled_training():
    """Run the scaled up unified progressive training"""
    print_header("RUNNING SCALED UNIFIED PROGRESSIVE TRAINING")
    
    # Execute the training script
    training_script = os.path.join(project_root, "optimization", "unified_progressive_training.py")
    
    print(f"Starting training at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Running: {training_script}")
    
    try:
        # Set environment variables for Unicode support
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run the training script as a subprocess and capture output
        process = subprocess.Popen(
            [sys.executable, training_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env
        )
        
        # Stream the output in real-time with proper encoding
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                # Decode with utf-8 and handle errors
                try:
                    decoded_line = output.decode('utf-8', errors='replace')
                    print(decoded_line, end='')
                except Exception as e:
                    print(f"[Error decoding output: {e}]")
        
        # Wait for completion
        process.wait()
        
        if process.returncode != 0:
            print(f"Training failed with exit code {process.returncode}")
            return False
        
        print(f"Training completed successfully at {datetime.now().strftime('%H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"Error running training script: {e}")
        return False

def run_evaluation():
    """Run evaluation against the base model"""
    print_header("RUNNING COMPARATIVE EVALUATION")
    
    # Execute the evaluation script
    eval_script = os.path.join(project_root, "experiments", "evaluate_unified_model.py")
    
    print(f"Starting evaluation at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Running: {eval_script}")
    
    try:
        # Set environment variables for Unicode support
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Run the evaluation script as a subprocess and capture output
        process = subprocess.Popen(
            [sys.executable, eval_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env
        )
        
        # Stream the output in real-time with proper encoding
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
            if output:
                # Decode with utf-8 and handle errors
                try:
                    decoded_line = output.decode('utf-8', errors='replace')
                    print(decoded_line, end='')
                except Exception as e:
                    print(f"[Error decoding output: {e}]")
        
        # Wait for completion
        process.wait()
        
        if process.returncode != 0:
            print(f"Evaluation failed with exit code {process.returncode}")
            return False
        
        print(f"Evaluation completed successfully at {datetime.now().strftime('%H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"Error running evaluation script: {e}")
        return False

def save_summary(training_success, evaluation_success, total_time):
    """Save a summary of the run to a markdown file"""
    summary_path = os.path.join(project_root, "experiments", "results", "scaled_training_summary.md")
    
    # Get memory usage information
    memory_info = psutil.Process(os.getpid()).memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)
    
    # Create content
    content = [
        "# Scaled Training and Evaluation Summary",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Runtime:** {total_time:.1f} seconds ({total_time/60:.1f} minutes)",
        f"**Memory Usage:** {memory_usage_mb:.2f} MB",
        "",
        "## Status",
        f"- **Training:** {'[SUCCESS]' if training_success else '[FAILED]'}",
        f"- **Evaluation:** {'[SUCCESS]' if evaluation_success else '[FAILED]'}",
        "",
        "## Results Location",
        f"- **Training Models:** {os.path.join(project_root, 'models', 'unified_progressive')}",
        f"- **Evaluation Results:** {os.path.join(project_root, 'experiments', 'results', 'unified_eval')}",
        "",
        "## System Information",
        f"- **CPU:** {psutil.cpu_count(logical=True)} logical cores, {psutil.cpu_count(logical=False)} physical cores",
        f"- **Total Memory:** {psutil.virtual_memory().total / (1024**3):.1f} GB",
        f"- **Python Version:** {sys.version}",
        f"- **PyTorch Version:** {torch.__version__}",
        "",
        "## Next Steps",
        "",
        "- Review the evaluation report in the results directory",
        "- Compare performance metrics between base and trained models",
        "- Analyze accuracy across different difficulty levels",
        "- Consider additional hyperparameter tuning based on results"
    ]
    
    # Write to file
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    
    print(f"\n[OK] Summary saved to {summary_path}")

def main():
    """Run the full training and evaluation pipeline"""
    start_time = time.time()
    
    print_header("SCALED TRAINING AND EVALUATION PIPELINE")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"CPU: {psutil.cpu_count(logical=True)} logical cores, {psutil.cpu_count(logical=False)} physical cores")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB total")
    
    # Optimize environment for CPU performance
    optimize_environment()
    
    # Step 1: Run scaled training
    training_success = run_scaled_training()
    if not training_success:
        print("Training failed. Stopping pipeline.")
        save_summary(False, False, time.time() - start_time)
        return
    
    # Step 2: Run evaluation
    evaluation_success = run_evaluation()
    if not evaluation_success:
        print("Evaluation failed.")
    
    # Summary
    total_time = time.time() - start_time
    print_header("PIPELINE COMPLETE")
    print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Training: {'[SUCCESS]' if training_success else '[FAILED]'}")
    print(f"Evaluation: {'[SUCCESS]' if evaluation_success else '[FAILED]'}")
    print(f"\nResults available in:")
    print(f"- Training: {os.path.join(project_root, 'models', 'unified_progressive')}")
    print(f"- Evaluation: {os.path.join(project_root, 'experiments', 'results', 'unified_eval')}")
    
    # Save summary to markdown file
    save_summary(training_success, evaluation_success, total_time)
    
    if training_success and evaluation_success:
        print("\n[SUCCESS] Pipeline completed successfully!")
    else:
        print("\n[WARNING] Pipeline completed with errors.")

if __name__ == "__main__":
    main()
