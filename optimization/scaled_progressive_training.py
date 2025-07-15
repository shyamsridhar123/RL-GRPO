"""
Scaled Progressive Training Pipeline
Implements larger-scale training with 100+ samples per stage
"""

import os
import sys
import time
import gc
import psutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from optimization.unified_progressive_training import (
    UnifiedProgressiveTrainer,
    UnifiedProgressiveConfig
)

class ScaledProgressiveConfig(UnifiedProgressiveConfig):
    """Configuration for scaled progressive training"""
    
    def __init__(self):
        super().__init__()
        
        # Scaled sample sizes
        self.stage1_samples = 100  # Basic arithmetic
        self.stage2_samples = 150  # Multi-step problems  
        self.stage3_samples = 200  # Complex reasoning
        
        # Conservative parameters for stability
        self.learning_rate = 5e-6  # Even more conservative
        self.batch_size = 1
        self.num_train_epochs = 1
        
        # Memory optimization
        self.gradient_accumulation_steps = 2
        self.max_memory_mb = 8000  # 8GB limit
        
        # Progress tracking
        self.log_every_n_steps = 10
        self.save_intermediate_checkpoints = True

def create_scaled_training_data():
    """Generate training data for scaled experiments"""
    
    # Stage 1: Basic arithmetic (100 samples)
    stage1_data = [
        {"problem": f"What is {i} + {j}?", "solution": str(i + j)}
        for i in range(1, 21) for j in range(1, 6)
    ]
    
    # Stage 2: Multi-step problems (150 samples)  
    stage2_data = [
        {"problem": f"If I have {a} apples and buy {b} more, then eat {c}, how many do I have?", 
         "solution": str(a + b - c)}
        for a in range(5, 20) for b in range(2, 8) for c in range(1, 5)
    ][:150]
    
    # Stage 3: Complex reasoning (200 samples)
    stage3_data = [
        {"problem": f"A train travels at {speed} mph for {time} hours. How far does it travel?",
         "solution": f"{speed * time} miles"}
        for speed in range(30, 80, 5) for time in range(1, 6)
    ][:200]
    
    return stage1_data, stage2_data, stage3_data

def monitor_resources():
    """Monitor system resources during training"""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    return {
        "memory_used_mb": (memory.total - memory.available) // (1024 * 1024),
        "memory_percent": memory.percent,
        "cpu_percent": cpu_percent
    }

def main():
    """Run scaled progressive training"""
    
    print("🚀 Starting Scaled Progressive Training")
    print("=" * 50)
    
    # Configuration
    config = ScaledProgressiveConfig()
    
    print(f"Sample sizes: Stage 1: {config.stage1_samples}, "
          f"Stage 2: {config.stage2_samples}, Stage 3: {config.stage3_samples}")
    print(f"Total samples: {config.stage1_samples + config.stage2_samples + config.stage3_samples}")
    
    # Create training data
    print("\n📊 Generating scaled training data...")
    stage1_data, stage2_data, stage3_data = create_scaled_training_data()
    
    # Initialize trainer
    trainer = UnifiedProgressiveTrainer(config)
    
    # Track overall progress
    start_time = time.time()
    initial_resources = monitor_resources()
    
    print(f"\n💾 Initial memory usage: {initial_resources['memory_used_mb']} MB")
    print(f"🖥️  Initial CPU usage: {initial_resources['cpu_percent']:.1f}%")
    
    results = {}
    
    try:
        # Stage 1: Basic Arithmetic
        print(f"\n🎯 Stage 1: Basic Arithmetic ({len(stage1_data)} samples)")
        stage1_start = time.time()
        
        stage1_result = trainer.train_stage(
            stage_name="stage1_scaled",
            training_data=stage1_data,
            stage_config=config.get_stage1_config()
        )
        
        stage1_time = time.time() - stage1_start
        stage1_resources = monitor_resources()
        
        print(f"✅ Stage 1 completed in {stage1_time:.1f}s")
        print(f"   Memory: {stage1_resources['memory_used_mb']} MB")
        
        results['stage1'] = {
            'time': stage1_time,
            'samples': len(stage1_data),
            'memory_mb': stage1_resources['memory_used_mb'],
            'result': stage1_result
        }
        
        # Cleanup
        gc.collect()
        
        # Stage 2: Multi-step Problems
        print(f"\n🎯 Stage 2: Multi-step Problems ({len(stage2_data)} samples)")
        stage2_start = time.time()
        
        stage2_result = trainer.train_stage(
            stage_name="stage2_scaled", 
            training_data=stage2_data,
            stage_config=config.get_stage2_config(),
            previous_model_path="./models/unified_progressive/stage1_scaled"
        )
        
        stage2_time = time.time() - stage2_start
        stage2_resources = monitor_resources()
        
        print(f"✅ Stage 2 completed in {stage2_time:.1f}s")
        print(f"   Memory: {stage2_resources['memory_used_mb']} MB")
        
        results['stage2'] = {
            'time': stage2_time,
            'samples': len(stage2_data),
            'memory_mb': stage2_resources['memory_used_mb'],
            'result': stage2_result
        }
        
        # Cleanup
        gc.collect()
        
        # Stage 3: Complex Reasoning
        print(f"\n🎯 Stage 3: Complex Reasoning ({len(stage3_data)} samples)")
        stage3_start = time.time()
        
        stage3_result = trainer.train_stage(
            stage_name="stage3_scaled",
            training_data=stage3_data, 
            stage_config=config.get_stage3_config(),
            previous_model_path="./models/unified_progressive/stage2_scaled"
        )
        
        stage3_time = time.time() - stage3_start
        stage3_resources = monitor_resources()
        
        print(f"✅ Stage 3 completed in {stage3_time:.1f}s")
        print(f"   Memory: {stage3_resources['memory_used_mb']} MB")
        
        results['stage3'] = {
            'time': stage3_time,
            'samples': len(stage3_data),
            'memory_mb': stage3_resources['memory_used_mb'],
            'result': stage3_result
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    # Final summary
    total_time = time.time() - start_time
    total_samples = len(stage1_data) + len(stage2_data) + len(stage3_data)
    final_resources = monitor_resources()
    
    print("\n" + "=" * 50)
    print("🎉 SCALED TRAINING COMPLETE")
    print("=" * 50)
    print(f"📊 Total samples: {total_samples}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"💾 Peak memory: {final_resources['memory_used_mb']} MB")
    print(f"🚀 Throughput: {total_samples/total_time:.4f} samples/second")
    
    # Save detailed results
    import json
    results_summary = {
        'total_time': total_time,
        'total_samples': total_samples,
        'throughput': total_samples/total_time,
        'peak_memory_mb': final_resources['memory_used_mb'],
        'stages': results
    }
    
    os.makedirs("./experiments/results/scaled_training", exist_ok=True)
    with open("./experiments/results/scaled_training/performance_metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n📁 Results saved to: ./experiments/results/scaled_training/")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Ready for evaluation with scaled model!")
        print("Next: Run evaluation script on the scaled models")
    else:
        print("\n❌ Training failed - check logs for details")
