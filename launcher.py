#!/usr/bin/env python3
"""
GRPO Training System Launcher
Provides a convenient interface for running various components of the system
"""

import os
import sys
import argparse
import time
import subprocess
import webbrowser
from pathlib import Path
import json

# Fix console encoding for emoji support
if sys.platform == 'win32':
    # Force UTF-8 encoding for Windows console
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='backslashreplace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='backslashreplace')
    os.system('chcp 65001 > NUL')  # Set Windows console to UTF-8

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import project utilities
from src.utils.cpu_utils import setup_cpu_optimization, print_system_info


def run_command(cmd, description=None, wait=True, shell=False):
    """Run a command and return the process"""
    if description:
        print(f"\n>> {description}")
    
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    # If this is a Python script, wrap it with the encoding fix
    if isinstance(cmd, list) and cmd[0] == sys.executable:
        # Replace python with: python fix_encoding.py python
        fix_cmd = [
            sys.executable,
            os.path.join(project_root, "fix_encoding.py")
        ] + cmd
        cmd = fix_cmd
    
    # Set environment variables for UTF-8 encoding
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    if shell:
        process = subprocess.Popen(cmd, shell=True, text=True, env=env)
    else:
        process = subprocess.Popen(cmd, env=env)
    
    if wait:
        process.wait()
        return process.returncode
    else:
        return process


def run_training(args):
    """Run the training component"""
    # Set up CPU optimization
    setup_cpu_optimization(args.physical_cores, args.logical_cores)
    print_system_info()
    
    # Build command
    cmd = [
        sys.executable, 
        os.path.join(project_root, "optimization", "ultra_fast_training.py"),
        f"--physical_cores={args.physical_cores}",
        f"--logical_cores={args.logical_cores}",
        f"--batch_size={args.batch_size}",
        f"--num_generations={args.num_generations}",
        f"--learning_rate={args.learning_rate}"
    ]
    
    # Add optional arguments
    if args.config_path:
        cmd.append(f"--config_path={args.config_path}")
    if args.output_dir:
        cmd.append(f"--output_dir={args.output_dir}")
    
    # Run with monitoring if requested
    if args.monitor:
        monitor_cmd = [
            sys.executable,
            os.path.join(project_root, "scripts", "monitor_training.py"),
            "--output", os.path.join(project_root, "logs", "monitoring"),
            "--interval", "3"
        ]
        
        monitor_process = run_command(
            monitor_cmd, 
            "Starting system monitoring", 
            wait=False
        )
        
        try:
            return_code = run_command(cmd, "Starting GRPO training")
        finally:
            # Stop monitoring
            if monitor_process:
                monitor_process.terminate()
    else:
        return_code = run_command(cmd, "Starting GRPO training")
    
    return return_code


def run_progressive_training(args):
    """Run progressive training with multiple stages"""
    # Set up CPU optimization
    setup_cpu_optimization(args.physical_cores, args.logical_cores)
    print_system_info()
    
    # Build command
    cmd = [
        sys.executable, 
        os.path.join(project_root, "src", "training", "progressive_training.py"),
        f"--physical_cores={args.physical_cores}",
        f"--logical_cores={args.logical_cores}"
    ]
    
    # Run with monitoring if requested
    if args.monitor:
        monitor_cmd = [
            sys.executable,
            os.path.join(project_root, "scripts", "monitor_training.py"),
            "--output", os.path.join(project_root, "logs", "monitoring"),
            "--interval", "3"
        ]
        
        monitor_process = run_command(
            monitor_cmd, 
            "Starting system monitoring", 
            wait=False
        )
        
        try:
            return_code = run_command(cmd, "Starting progressive training")
        finally:
            # Stop monitoring
            if monitor_process:
                monitor_process.terminate()
    else:
        return_code = run_command(cmd, "Starting progressive training")
    
    return return_code


def run_demo_app(args):
    """Run the Gradio demo app"""
    # Build command
    cmd = [
        sys.executable, 
        os.path.join(project_root, "app.py")
    ]
    
    # Run the app
    app_process = run_command(cmd, "Starting Gradio demo app", wait=False)
    
    # Wait for app to start
    print("Waiting for app to start...")
    time.sleep(3)
    
    # Open browser if requested
    if not args.no_browser:
        print("Opening browser...")
        webbrowser.open("http://127.0.0.1:7860")
    
    try:
        print("\nPress Ctrl+C to stop the app")
        app_process.wait()
    except KeyboardInterrupt:
        print("Stopping app...")
        app_process.terminate()
    
    return 0


def run_evaluation(args):
    """Run model evaluation"""
    # Build command
    cmd = [
        sys.executable, 
        os.path.join(project_root, "experiments", "balanced_evaluation.py"),
        f"--model_path={args.model_path}"
    ]
    
    return run_command(cmd, "Running model evaluation")


def run_benchmark(args):
    """Run benchmarking and comparison"""
    # Build command
    cmd = [
        sys.executable, 
        os.path.join(project_root, "experiments", "benchmark_analysis.py")
    ]
    
    if args.target_model:
        cmd.append(f"--target_model={args.target_model}")
    
    return run_command(cmd, "Running benchmark analysis")


def run_visualization(args):
    """Run visualization of training results"""
    # Build command
    cmd = [
        sys.executable, 
        os.path.join(project_root, "scripts", "visualize_results.py"),
        f"--log-dir={args.log_dir}",
    ]
    
    if args.output_dir:
        cmd.append(f"--output-dir={args.output_dir}")
    
    return run_command(cmd, "Generating visualizations")


def list_models(args):
    """List all available models"""
    models_dir = Path(args.models_dir)
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return 1
    
    print("\nAvailable Models:")
    print("=" * 60)
    
    # Look for model directories
    model_dirs = []
    
    # Look in stage directories
    for stage_dir in models_dir.glob("stage*"):
        if stage_dir.is_dir():
            final_model = stage_dir / "final_model"
            if final_model.exists() and final_model.is_dir():
                model_dirs.append((final_model, f"{stage_dir.name}/final_model"))
    
    # Look in other directories
    for subdir in models_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("stage") and subdir.name != "archive":
            # Check if it contains config.json
            if (subdir / "config.json").exists():
                model_dirs.append((subdir, subdir.name))
            
            # Check for final_model subdirectory
            final_model = subdir / "final_model"
            if final_model.exists() and final_model.is_dir():
                model_dirs.append((final_model, f"{subdir.name}/final_model"))
    
    if not model_dirs:
        print("No models found.")
        return 0
    
    # Display model information
    for i, (path, name) in enumerate(model_dirs):
        # Check for metadata
        metadata_file = path / "training_metadata.json"
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception:
                pass
        
        # Calculate model size
        model_bin = path / "pytorch_model.bin"
        safetensors_model = path / "model.safetensors"
        
        if model_bin.exists():
            size_mb = model_bin.stat().st_size / (1024 * 1024)
            model_format = "PyTorch"
        elif safetensors_model.exists():
            size_mb = safetensors_model.stat().st_size / (1024 * 1024)
            model_format = "SafeTensors"
        else:
            size_mb = 0
            model_format = "Unknown"
        
        # Display info
        print(f"{i+1}. {name}")
        print(f"   Path: {path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Format: {model_format}")
        
        if metadata:
            if "training_time_seconds" in metadata:
                print(f"   Training Time: {metadata['training_time_seconds']:.2f} seconds")
            if "throughput_samples_per_second" in metadata:
                print(f"   Throughput: {metadata['throughput_samples_per_second']:.4f} samples/second")
        
        print("")
    
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GRPO Training System Launcher")
    
    # Create subparsers for different components
    subparsers = parser.add_subparsers(dest="component", help="Component to run")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Run GRPO training")
    train_parser.add_argument("--physical_cores", type=int, default=12, help="Number of physical CPU cores")
    train_parser.add_argument("--logical_cores", type=int, default=14, help="Number of logical CPU cores")
    train_parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    train_parser.add_argument("--num_generations", type=int, default=2, help="Number of generations")
    train_parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    train_parser.add_argument("--config_path", type=str, help="Path to config file")
    train_parser.add_argument("--output_dir", type=str, help="Output directory for trained model")
    train_parser.add_argument("--monitor", action="store_true", help="Enable resource monitoring")
    
    # Progressive training command
    prog_parser = subparsers.add_parser("progressive", help="Run progressive training")
    prog_parser.add_argument("--physical_cores", type=int, default=12, help="Number of physical CPU cores")
    prog_parser.add_argument("--logical_cores", type=int, default=14, help="Number of logical CPU cores")
    prog_parser.add_argument("--monitor", action="store_true", help="Enable resource monitoring")
    
    # Demo app command
    demo_parser = subparsers.add_parser("demo", help="Run Gradio demo app")
    demo_parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Run model evaluation")
    eval_parser.add_argument("--model_path", type=str, required=True, help="Path to model to evaluate")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark analysis")
    bench_parser.add_argument("--target_model", type=str, help="Path to target model for comparison")
    
    # Visualization command
    viz_parser = subparsers.add_parser("visualize", help="Run visualization of training results")
    viz_parser.add_argument("--log-dir", type=str, default="./logs", help="Directory containing training logs")
    viz_parser.add_argument("--output-dir", type=str, help="Directory to save visualizations")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List all available models")
    list_parser.add_argument("--models-dir", type=str, default="./models", help="Directory containing model checkpoints")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the selected component
    if args.component == "train":
        return run_training(args)
    elif args.component == "progressive":
        return run_progressive_training(args)
    elif args.component == "demo":
        return run_demo_app(args)
    elif args.component == "evaluate":
        return run_evaluation(args)
    elif args.component == "benchmark":
        return run_benchmark(args)
    elif args.component == "visualize":
        return run_visualization(args)
    elif args.component == "list-models":
        return list_models(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
