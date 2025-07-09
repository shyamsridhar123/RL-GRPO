#!/usr/bin/env python3
"""
Fast Startup GRPO Training: Minimize Initialization Overhead
Addresses the real bottleneck: 41+ seconds of library loading and model initialization
"""

import os
import sys
import time
import warnings
from pathlib import Path

# Suppress warnings to speed up imports
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configure hardware acceleration before any imports
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['NUMEXPR_NUM_THREADS'] = '12'
os.environ['OPENBLAS_NUM_THREADS'] = '12'

print("🚀 FAST STARTUP GRPO TRAINING")
print("=" * 50)
startup_timer = time.time()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Optimize imports - import only what's needed, when needed
def lazy_import_torch():
    """Lazy import torch with optimization"""
    import torch
    torch.set_num_threads(12)
    torch.manual_seed(42)
    return torch

def lazy_import_transformers():
    """Lazy import transformers with caching"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import logging
    logging.set_verbosity_error()  # Reduce verbosity
    return AutoTokenizer, AutoModelForCausalLM

def lazy_import_trl():
    """Lazy import TRL components"""
    from trl import GRPOTrainer, GRPOConfig
    return GRPOTrainer, GRPOConfig

def create_minimal_dataset():
    """Create minimal dataset to reduce initialization time"""
    print("📊 Creating minimal dataset...")
    dataset_timer = time.time()
    
    # Use minimal math problems for fastest startup
    prompts = [
        "Solve: 2+3=",
        "Solve: 5-1=", 
        "Solve: 4÷2=",
        "Solve: 3×2=",
        "Solve: 7+1=",
        "Solve: 9-3=",
        "Solve: 8÷4=",
        "Solve: 5×1="
    ]
    
    # Use datasets for compatibility but minimize overhead
    from datasets import Dataset
    dataset = Dataset.from_dict({"prompt": prompts})
    
    print(f"   Dataset created in {time.time() - dataset_timer:.2f}s")
    return dataset

def load_model_cached():
    """Load model with caching optimizations"""
    print("🧠 Loading model (with caching)...")
    model_timer = time.time()
    
    torch = lazy_import_torch()
    AutoTokenizer, AutoModelForCausalLM = lazy_import_transformers()
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    cache_dir = Path("models/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load with optimizations
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=False  # Allow download if not cached
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=False
    )
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   Model loaded in {time.time() - model_timer:.2f}s")
    return model, tokenizer

def create_fast_grpo_config():
    """Create optimized GRPO config for fastest training"""
    print("⚙️  Creating GRPO config...")
    
    GRPOTrainer, GRPOConfig = lazy_import_trl()
    
    # Use absolute minimal config to avoid parameter errors
    config = GRPOConfig(
        output_dir="models/fast_startup",
        overwrite_output_dir=True,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        num_train_epochs=0.1,  # Very short training
        logging_steps=1,
        save_steps=10,
        dataloader_num_workers=0,
        # CPU-specific settings to avoid bf16/GPU errors
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
    )
    
    return config

def simple_reward_function(responses):
    """Ultra-simple reward function to minimize computation"""
    # Just return small positive rewards to make training work
    return [0.1] * len(responses)

def run_fast_startup_training():
    """Run GRPO training with minimal startup time"""
    print("\n🏃 Starting GRPO training...")
    training_timer = time.time()
    
    # Load components
    model, tokenizer = load_model_cached()
    dataset = create_minimal_dataset()
    config = create_fast_grpo_config()
    
    # Import TRL after model loading
    GRPOTrainer, GRPOConfig = lazy_import_trl()
    
    # Create trainer
    print("🎯 Initializing GRPO trainer...")
    trainer_timer = time.time()
    
    trainer = GRPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_function=simple_reward_function,
    )
    
    print(f"   Trainer initialized in {time.time() - trainer_timer:.2f}s")
    
    # Run training
    print("🚂 Running GRPO training...")
    actual_training_timer = time.time()
    
    trainer.train()
    
    actual_training_time = time.time() - actual_training_timer
    print(f"   Actual training completed in {actual_training_time:.2f}s")
    
    # Save model
    print("💾 Saving model...")
    save_timer = time.time()
    trainer.save_model()
    print(f"   Model saved in {time.time() - save_timer:.2f}s")
    
    total_training_time = time.time() - training_timer
    print(f"\n✅ Total training pipeline: {total_training_time:.2f}s")
    
    return trainer, actual_training_time

def main():
    """Main execution with detailed timing"""
    try:
        # Run the training
        trainer, actual_training_time = run_fast_startup_training()
        
        total_time = time.time() - startup_timer
        initialization_time = total_time - actual_training_time
        
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE BREAKDOWN")
        print("=" * 60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Initialization overhead: {initialization_time:.2f}s ({initialization_time/total_time*100:.1f}%)")
        print(f"Actual training time: {actual_training_time:.2f}s ({actual_training_time/total_time*100:.1f}%)")
        
        # Calculate improvement
        baseline_init = 41  # From performance analysis
        improvement = baseline_init - initialization_time
        print(f"\n🎯 OPTIMIZATION RESULTS")
        print(f"Baseline initialization: {baseline_init:.0f}s")
        print(f"Optimized initialization: {initialization_time:.2f}s")
        print(f"Improvement: {improvement:.2f}s ({improvement/baseline_init*100:.1f}% faster)")
        
        if improvement > 20:
            print("✅ SIGNIFICANT IMPROVEMENT ACHIEVED!")
        elif improvement > 10:
            print("✅ Good improvement achieved")
        else:
            print("⚠️  Limited improvement - need more optimization")
            
        print(f"\n💡 For longer training runs, initialization overhead will be {initialization_time:.1f}s regardless of training duration")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
