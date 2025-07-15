"""
Automated Scaling Pipeline
Orchestrates scaled training and evaluation with monitoring
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

def run_command(command, description):
    """Run a command with monitoring"""
    print(f"\n🚀 {description}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Completed in {elapsed:.1f}s")
            return True, result.stdout
        else:
            print(f"❌ Failed after {elapsed:.1f}s")
            print(f"Error: {result.stderr}")
            return False, result.stderr
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False, str(e)

def check_prerequisites():
    """Check if system is ready for scaled training"""
    print("🔍 Checking Prerequisites")
    print("=" * 30)
    
    checks = [
        ("Python environment", "python --version"),
        ("Required packages", "python -c 'import torch, transformers, trl'"),
        ("Disk space", "python -c 'import shutil; print(f\"Free space: {shutil.disk_usage(\".\")[2]//1024**3} GB\")'")
    ]
    
    all_good = True
    for check_name, command in checks:
        success, output = run_command(command, f"Checking {check_name}")
        if not success:
            all_good = False
            print(f"⚠️  {check_name} check failed")
        else:
            print(f"✅ {check_name}: OK")
    
    return all_good

def main():
    """Run complete scaling pipeline"""
    
    print("🎯 SCALED TRAINING PIPELINE")
    print("=" * 50)
    print("This will run 450 total samples across 3 stages")
    print("Estimated time: 3-5 hours")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please fix issues and retry.")
        return False
    
    pipeline_start = time.time()
    
    # Step 1: Run scaled training
    print(f"\n📊 Step 1: Scaled Progressive Training")
    success, output = run_command(
        "python optimization/scaled_progressive_training.py",
        "Running scaled training (450 samples)"
    )
    
    if not success:
        print("❌ Scaled training failed")
        return False
    
    # Step 2: Run evaluation
    print(f"\n📈 Step 2: Model Evaluation")
    success, output = run_command(
        "python experiments/evaluate_scaled_models.py",
        "Evaluating scaled models"
    )
    
    if not success:
        print("⚠️  Evaluation failed, but training succeeded")
    
    # Step 3: Generate report
    total_time = time.time() - pipeline_start
    
    print("\n" + "=" * 50)
    print("🎉 SCALING PIPELINE COMPLETE")
    print("=" * 50)
    print(f"⏱️  Total pipeline time: {total_time/60:.1f} minutes")
    print(f"📊 Sample size increased: 15 → 450 (30x increase)")
    print(f"📁 Results in: ./experiments/results/scaled_*/")
    
    # Create pipeline summary
    summary = {
        'pipeline_completion_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_duration_minutes': total_time / 60,
        'sample_size_increase': '30x (15 → 450)',
        'training_successful': True,
        'evaluation_successful': success
    }
    
    os.makedirs("./experiments/results", exist_ok=True)
    with open("./experiments/results/scaling_pipeline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Pipeline summary saved")
    print("\n🎯 Next Steps:")
    print("   1. Review results in ./experiments/results/scaled_*/")
    print("   2. Compare performance improvements")
    print("   3. Consider Phase 2: 1000+ samples for research publication")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
