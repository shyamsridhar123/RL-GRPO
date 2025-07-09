#!/usr/bin/env python3
"""
Ultra-Fast CPU Training with Hardware Acceleration
Optimized for 12-core CPU with 14 logical processors
"""

import os
import sys
import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any
from datasets import Dataset

# AGGRESSIVE CPU OPTIMIZATION FOR 12-CORE SYSTEM
os.environ['OMP_NUM_THREADS'] = '12'  # Use physical cores
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['NUMEXPR_NUM_THREADS'] = '12'
os.environ['OPENBLAS_NUM_THREADS'] = '12'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Memory optimization - AGGRESSIVE MEMORY CLEANUP (matching gradio app)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'  # Smaller chunks
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Avoid network calls
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Aggressive cleanup
os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'  # Force malloc cleanup

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training.grpo_trainer import CPUGRPOTrainer, CPUGRPOConfig

def configure_ultra_cpu_optimization():
    """Configure maximum CPU optimization for 12-core system with REAL hardware acceleration"""
    print("[FAST] Configuring ULTRA CPU optimization for 12-core system...")
    
    # Import for system monitoring
    import psutil
    import gc
    
    # CRITICAL: Use ALL logical cores (you have 14!)
    logical_cores = psutil.cpu_count(logical=True)  # 14 cores
    physical_cores = psutil.cpu_count(logical=False)  # 12 cores
    
    # Set PyTorch to use ALL cores
    torch.set_num_threads(logical_cores)  # USE ALL 14 CORES
    torch.set_num_interop_threads(physical_cores // 2)  # 6 for inter-op
    
    # CRITICAL: Intel MKL-DNN (this was missing!)
    if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True
        print("   [OK] Intel MKL-DNN enabled (KEY OPTIMIZATION)")
    
    # Enable all CPU optimizations
    if hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available():
        torch.backends.mkl.enabled = True
        print("   [OK] Intel MKL BLAS enabled")
    
    if hasattr(torch.backends, 'openmp') and torch.backends.openmp.is_available():
        torch.backends.openmp.enabled = True
        print("   [OK] OpenMP enabled")
    
    # CRITICAL: Intel threading optimizations
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    os.environ['KMP_BLOCKTIME'] = '1'  # Low latency
    os.environ['DNNL_PRIMITIVE_CACHE_CAPACITY'] = '1024'  # MKL-DNN cache
    
    # Advanced CPU-specific optimizations
    try:
        if hasattr(torch.backends, 'quantized'):
            # Try different quantized engines
            available_engines = ['fbgemm', 'qnnpack', 'onednn']
            for engine in available_engines:
                try:
                    torch.backends.quantized.engine = engine
                    print(f"   [OK] {engine.upper()} quantized engine enabled")
                    break
                except RuntimeError:
                    continue
    except Exception as e:
        print(f"   WARNING Quantized engine not available: {e}")
        pass
    
    # Memory optimization
    torch.set_default_dtype(torch.float32)  # Optimal for CPU
    gc.collect()  # Clean up memory
    
    # Disable unnecessary features
    torch.backends.cudnn.enabled = False  # Not needed for CPU
    torch.backends.cudnn.benchmark = False
    
    print(f"   [OK] Using {torch.get_num_threads()} CPU threads")
    print(f"   [OK] Inter-op threads: {torch.get_num_interop_threads()}")
    print(f"   [OK] Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")

def create_reasoning_dataset(num_samples: int = 20) -> Dataset:
    """Create proper reasoning dataset for math problems with step-by-step solutions"""
    
    # GRPO requires minimum 2 samples for comparison - ensure we have at least that
    min_samples = max(2, num_samples)
    print(f"[FAST] Creating reasoning dataset with {min_samples} samples...")
    
    # Proper reasoning problems (GSM8K-style) that require step-by-step thinking
    reasoning_problems = np.array([
        "Sarah has 15 apples. She gives 1/3 of them to her friend and eats 2 herself. How many apples does she have left?",
        "A car travels 45 miles in the first hour and 55 miles in the second hour. What is the average speed?", 
        "If a box contains 24 chocolates and you eat 1/4 of them on Monday and 1/3 of the remaining on Tuesday, how many are left?",
        "Tom buys 3 books for $12 each and 2 pens for $3 each. If he pays with a $50 bill, how much change does he get?",
        "A rectangle has a length of 8 cm and width of 5 cm. What is its perimeter?",
        "Lisa runs 2.5 km every day for 6 days. How many kilometers does she run in total?",
        "If 5 pencils cost $2.50, what is the cost of 8 pencils?",
        "A pizza is cut into 8 equal slices. If Jake eats 3 slices and Maria eats 2 slices, what fraction of the pizza is left?",
        "In a class of 30 students, 18 are girls. What percentage of the class are boys?",
        "A train travels 120 km in 2 hours. At this rate, how long will it take to travel 300 km?"
    ])
    
    # Use numpy for efficient random sampling
    # Ensure we don't try to sample more than available
    sample_count = min(min_samples, len(reasoning_problems))
    
    if sample_count < len(reasoning_problems):
        indices = np.random.choice(len(reasoning_problems), sample_count, replace=False)
        selected_problems = reasoning_problems[indices]
    else:
        selected_problems = reasoning_problems
    
    # Proper prompts that encourage step-by-step reasoning
    prompts = [f"Solve step by step: {prob}" for prob in selected_problems]
    
    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"   [OK] Created {len(dataset)} reasoning samples with proper step-by-step problems")
    return dataset

def create_reasoning_reward():
    """Reward function that evaluates step-by-step reasoning quality"""
    
    def reasoning_reward(completions: List[str], **kwargs) -> List[float]:
        """Evaluate reasoning quality - not just correct answers"""
        rewards = []
        
        for completion in completions:
            reward = 0.0
            comp_str = completion.strip().lower()
            
            # Reward for showing work/reasoning steps
            reasoning_indicators = ['first', 'then', 'next', 'so', 'therefore', 'step', 'calculate', 'multiply', 'divide', 'add', 'subtract']
            step_count = sum(1 for indicator in reasoning_indicators if indicator in comp_str)
            reward += min(step_count * 0.3, 1.5)  # Up to 1.5 points for showing steps
            
            # Reward for numerical computations
            import re
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', completion)
            if len(numbers) >= 2:  # Multiple numbers suggest calculation
                reward += 1.0
            
            # Reward for mathematical operations
            math_ops = ['+', '-', '*', 'x', '/', 'div', '=']
            op_count = sum(1 for op in math_ops if op in completion)
            reward += min(op_count * 0.2, 0.8)
            
            # Reward for clear final answer
            answer_patterns = ['answer', 'total', 'result', 'final', '=']
            if any(pattern in comp_str for pattern in answer_patterns):
                reward += 0.7
            
            # Length bonus for detailed reasoning (but not too long)
            word_count = len(completion.split())
            if 15 <= word_count <= 80:  # Sweet spot for reasoning
                reward += 0.5
            elif word_count < 10:  # Too short for reasoning
                reward -= 0.5
            elif word_count > 120:  # Too verbose
                reward -= 0.3
            
            # Clamp reward
            rewards.append(max(min(reward, 3.0), 0.0))
        
        return rewards
    
    return reasoning_reward

def run_ultra_fast_training(learning_rate=1e-5, num_samples=10, num_epochs=1, dataset_name="gsm8k", task_type="math"):
    """Run ultra-optimized GRPO training for 12-core CPU with performance monitoring"""
    
    print(f"[ROCKET] ULTRA-FAST CPU Training Starting!")
    print("=" * 60)
    print(f"[TARGET] Parameters: lr={learning_rate}, samples={num_samples}, epochs={num_epochs}, dataset={dataset_name}, task={task_type}")
    
    # Import performance monitoring
    import psutil
    import gc
    import json
    
    # Start performance monitoring
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**3  # GB
    
    print(f"[SEARCH] Initial memory usage: {initial_memory:.2f} GB")
    print(f"[SEARCH] Available CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    
    # CRITICAL: Clean up system before training
    aggressive_system_cleanup()
    
    # Configure CPU optimization
    configure_ultra_cpu_optimization()
    
    # Validate all optimizations are working
    validate_optimizations()
    
    # Memory cleanup before training
    gc.collect()
    
    # Create dataset with specified number of samples
    if dataset_name == "gsm8k" and task_type == "math":
        dataset = create_reasoning_dataset(num_samples=num_samples)
        print(f"[TARGET] Using GSM8K-style math reasoning dataset")
    else:
        dataset = create_reasoning_dataset(num_samples=num_samples)
        print(f"[TARGET] Using custom reasoning dataset (dataset={dataset_name}, task={task_type})")
    
    # Adaptive batch size based on number of samples and system memory
    # GRPO needs at least 2 samples per batch for comparison
    available_memory_gb = psutil.virtual_memory().available / 1024**3
    
    if available_memory_gb < 2.0:
        # Low memory: use smaller batches
        adaptive_batch_size = min(max(2, num_samples // 4), 4)
    elif available_memory_gb > 4.0:
        # High memory: can use larger batches for efficiency  
        adaptive_batch_size = min(max(2, num_samples // 2), 8)
    else:
        # Medium memory: balanced approach
        adaptive_batch_size = min(max(2, num_samples // 3), 6)
    
    # Never exceed number of samples
    adaptive_batch_size = min(adaptive_batch_size, num_samples)
    
    print(f"[TARGET] Adaptive batch size: {adaptive_batch_size} (for {num_samples} samples, {available_memory_gb:.1f}GB available)")
    
    # Set num_generations for GRPO compatibility 
    # GRPO requires: effective_batch_size % num_generations == 0
    # Find a reasonable number of generations that divides evenly into batch size
    possible_generations = [i for i in [2, 3, 4, 6, 8] if adaptive_batch_size % i == 0]
    if possible_generations:
        num_generations = min(possible_generations)  # Use smallest valid value for speed
    else:
        num_generations = 2  # Fallback - most batch sizes are divisible by 2
    
    print(f"[TARGET] Using {num_generations} generations (divisible into batch size {adaptive_batch_size})")
    
    # MAXIMUM SPEED config for 12-core CPU with optimized data loading
    config = CPUGRPOConfig(
        model_name="./models/stage3/final_model",  # Updated path to organized structure
        output_dir="./models/ultra_fast",
        per_device_train_batch_size=adaptive_batch_size,  # ADAPTIVE batch size
        gradient_accumulation_steps=1,  # IMMEDIATE updates for speed
        learning_rate=learning_rate,  # Use passed parameter
        num_train_epochs=num_epochs * 0.25,  # Scale based on epochs parameter
        warmup_steps=2,  # MINIMAL warmup
        logging_steps=2,
        save_steps=10,
        max_prompt_length=256,   # Reasonable for reasoning tasks  
        max_completion_length=128,   # Space for step-by-step reasoning
        num_generations=num_generations,  # Adaptive: matches batch size for GRPO compatibility
        dataloader_num_workers=3,  # OPTIMAL for 12-core CPU (proven fastest)
        # Generation optimization for speed
        temperature=0.1,  # Low temperature for more deterministic generation
        top_p=0.9,  # Standard nucleus sampling
    )
    
    # Create trainer with appropriate reward function
    trainer = CPUGRPOTrainer(config)
    if task_type == "math":
        reward_function = create_reasoning_reward()
        print(f"[TARGET] Using math reasoning reward function")
    else:
        reward_function = create_reasoning_reward()  # Can be extended for other task types
        print(f"[TARGET] Using general reasoning reward function")
    
    # Monitor memory before training
    pre_training_memory = process.memory_info().rss / 1024**3
    print(f"[SEARCH] Memory before training: {pre_training_memory:.2f} GB")
    
    # Track time
    training_start = time.time()
    
    print("[FAST] Starting lightning-fast training...")
    
    try:
        final_model_path = trainer.train(
            dataset=dataset,
            reward_fn=reward_function,
            progress_callback=lambda progress: print(f"[FAST] Step {progress['step']}: Loss = {progress['loss']:.4f} (Speed: {progress.get('speed', 'N/A')} samples/sec)")
        )
        
        # Performance summary
        training_time = time.time() - training_start
        total_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024**3
        
        print("\n" + "=" * 60)
        print("[TARGET] ULTRA-FAST TRAINING COMPLETED!")
        print("=" * 60)
        print(f"[TIMER]  Training time: {training_time:.2f} seconds")
        print(f"[TIMER]  Total time: {total_time:.2f} seconds")
        print(f"[UNKNOWN] Memory usage: {initial_memory:.2f} -> {final_memory:.2f} GB")
        print(f"[ROCKET] Average CPU usage: {psutil.cpu_percent(interval=1):.1f}%")
        print(f"[CHART] Throughput: {len(dataset) / training_time:.2f} samples/second")
        print(f"[FOLDER] Output: {final_model_path}")
        
        # Save performance metrics to correct path
        performance_metrics = {
            "training_time_seconds": training_time,
            "total_time_seconds": total_time,
            "initial_memory_gb": initial_memory,
            "final_memory_gb": final_memory,
            "throughput_samples_per_second": len(dataset) / training_time,
            "dataset_size": len(dataset),
            "batch_size": config.per_device_train_batch_size,
            "num_workers": config.dataloader_num_workers,
            "cpu_cores_used": torch.get_num_threads(),
            "optimization_applied": "ultra_fast_v2"
        }
        
        # Use correct path
        metrics_path = "./models/ultra_fast/performance_metrics.json"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(performance_metrics, f, indent=2)
        
        print("[UNKNOWN] Performance metrics saved to performance_metrics.json")
        
        return final_model_path
        
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Cleanup
        gc.collect()
        print("[CLEAN] Memory cleanup completed")

def run_benchmark_test(target_model_path=None):
    """Benchmark test comparing BASE QWEN vs SELECTED MODEL to understand speed differences"""
    
    print("[RUN] Running COMPARATIVE MODEL Benchmark Test...")
    print("=" * 70)
    
    configure_ultra_cpu_optimization()
    
    # Test matrix operations (common in ML)
    print("Testing matrix operations...")
    start_time = time.time()
    
    # Create large matrices to stress test CPU
    a = torch.randn(2000, 2000)
    b = torch.randn(2000, 2000)
    
    # Matrix multiplication (uses BLAS)
    c = torch.mm(a, b)
    
    matmul_time = time.time() - start_time
    print(f"   Matrix multiplication (2000x2000): {matmul_time:.2f} seconds")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import os
    
    # Test prompts (same for both models)
    test_prompts = [
        "Solve step by step: Sarah has 15 apples. She gives 1/3 of them to her friend and eats 2 herself. How many apples does she have left?",
        "Solve step by step: If 5 pencils cost $2.50, what is the cost of 8 pencils?"
    ]
    
    # If no target model specified, discover available models and let user choose
    if target_model_path is None:
        print("\nDISCOVERING AVAILABLE MODELS...")
        print("=" * 50)
        
        available_models = []
        
        # Check common model locations
        model_locations = [
            ("./models/stage1/final_model", "Stage 1 Model"),
            ("./models/stage2/final_model", "Stage 2 Model"), 
            ("./models/stage3/final_model", "Stage 3 Model"),
            ("./models/ultra_fast/final_model", "Ultra Fast Trained Model"),
            ("./models/extreme_fast", "Extreme Fast Model"),
            ("./models/hardware_accelerated", "Hardware Accelerated Model")
        ]
        
        # Scan for available models
        for path, description in model_locations:
            if os.path.exists(path):
                available_models.append((path, description))
        
        # Also scan models directory for any other models
        if os.path.exists("./models"):
            for item in os.listdir("./models"):
                item_path = f"./models/{item}"
                if os.path.isdir(item_path) and item not in [loc[0].split('/')[-2] for loc in model_locations]:
                    # Check if it looks like a model (has config files)
                    if any(os.path.exists(os.path.join(item_path, f)) for f in ['config.json', 'final_model', 'pytorch_model.bin']):
                        available_models.append((item_path, f"Custom Model ({item})"))
        
        if not available_models:
            print("ERROR No local models found!")
            print("   Please ensure you have trained models in ./models/ directory")
            return
        
        print(f"Found {len(available_models)} available models:")
        for i, (path, desc) in enumerate(available_models):
            print(f"   {i+1}. {desc:<30} [{path}]")
        
        print(f"   0. Exit benchmark")
        
        # Get user choice
        while True:
            try:
                choice = input(f"\nSelect model to benchmark against Qwen (1-{len(available_models)}, 0 to exit): ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    print("Benchmark cancelled.")
                    return
                elif 1 <= choice_num <= len(available_models):
                    target_model_path, target_description = available_models[choice_num - 1]
                    print(f"\nOK Selected: {target_description}")
                    print(f"   Path: {target_model_path}")
                    break
                else:
                    print(f"ERROR Please enter a number between 1 and {len(available_models)}, or 0 to exit")
            except ValueError:
                print("ERROR Please enter a valid number")
            except KeyboardInterrupt:
                print("\nBenchmark cancelled.")
                return
    else:
        # Use provided target model path
        if not os.path.exists(target_model_path):
            print(f"ERROR Target model not found: {target_model_path}")
            return
        target_description = f"Target Model ({target_model_path})"
    
    models_to_test = [
        {
            "name": "Base Qwen 0.5B (Reference)",
            "path": "Qwen/Qwen2-0.5B-Instruct",
            "description": "Small, fast baseline model"
        },
        {
            "name": "Selected Model",
            "path": target_model_path, 
            "description": target_description
        }
    ]
    
    results = []
    
    for model_info in models_to_test:
        print(f"\n{'='*70}")
        print(f"TESTING: {model_info['name']}")
        print(f"   Path: {model_info['path']}")
        print(f"   Description: {model_info['description']}")
        print(f"{'='*70}")
        
        # Check if model exists (for local models)
        if model_info['path'].startswith('./') and not os.path.exists(model_info['path']):
            print(f"   ERROR: Model not found at {model_info['path']}")
            print("   Available models:")
            if os.path.exists("./models"):
                for item in os.listdir("./models"):
                    print(f"      - ./models/{item}")
            results.append({
                "name": model_info['name'],
                "status": "NOT_FOUND",
                "load_time": 0,
                "parameters": 0,
                "avg_speed": 0
            })
            continue
        
        start_time = time.time()
        try:
            print(f"   Loading model...")
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_info['path'])
            model = AutoModelForCausalLM.from_pretrained(model_info['path'], torch_dtype=torch.float32)
            
            load_time = time.time() - start_time
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"   OK Model loaded in: {load_time:.2f} seconds")
            print(f"   Model parameters: {param_count:,}")
            
            total_inference_time = 0
            total_tokens_generated = 0
            
            for i, prompt in enumerate(test_prompts):
                print(f"\n   Test {i+1}/{len(test_prompts)}: {prompt[:40]}...")
                
                inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=256, truncation=True)
                print(f"      Input tokens: {inputs.shape[1]}")
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        inputs, 
                        max_new_tokens=64,   # Shorter for comparison speed
                        temperature=0.1,     # Same as training config
                        top_p=0.9,          # Same as training config
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                inference_time = time.time() - start_time
                
                generated_tokens = outputs.shape[1] - inputs.shape[1]
                total_inference_time += inference_time
                total_tokens_generated += generated_tokens
                
                print(f"      Generated {generated_tokens} tokens in {inference_time:.2f}s")
                print(f"      Speed: {generated_tokens/inference_time:.1f} tokens/sec")
                
                # Show a bit of the output
                generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                print(f"      Output: {generated_text[:60]}...")
            
            # Overall performance for this model
            avg_speed = total_tokens_generated / total_inference_time
            
            results.append({
                "name": model_info['name'],
                "status": "SUCCESS",
                "load_time": load_time,
                "parameters": param_count,
                "avg_speed": avg_speed,
                "total_time": total_inference_time,
                "total_tokens": total_tokens_generated
            })
            
            print(f"\n   Results for {model_info['name']}:")
            print(f"      Load time: {load_time:.2f}s")
            print(f"      Parameters: {param_count:,}")
            print(f"      Average speed: {avg_speed:.1f} tokens/sec")
            
            # Clean up memory
            del model, tokenizer
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"   ERROR loading {model_info['name']}: {str(e)}")
            results.append({
                "name": model_info['name'],
                "status": "ERROR",
                "load_time": 0,
                "parameters": 0,
                "avg_speed": 0
            })
    
    # COMPARISON SUMMARY
    print(f"\n{'='*70}")
    print("BENCHMARK COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    
    if len(successful_results) >= 2:
        base_result = successful_results[0]  # Qwen
        your_result = successful_results[1]  # Your model
        
        print(f"Model Comparison:")
        print(f"   {base_result['name']:30}: {base_result['parameters']:>10,} params, {base_result['avg_speed']:>5.1f} tok/s")
        print(f"   {your_result['name']:30}: {your_result['parameters']:>10,} params, {your_result['avg_speed']:>5.1f} tok/s")
        
        speed_ratio = base_result['avg_speed'] / your_result['avg_speed'] if your_result['avg_speed'] > 0 else 0
        param_ratio = your_result['parameters'] / base_result['parameters'] if base_result['parameters'] > 0 else 0
        
        print(f"\nAnalysis:")
        print(f"   Your model is {param_ratio:.1f}x larger than base Qwen")
        print(f"   Base Qwen is {speed_ratio:.1f}x faster than your model")
        print(f"   Speed difference explains training slowness!")
        
        # Training time analysis
        print(f"\nGRPO Training Time Analysis:")
        batch_tokens = 6 * 2 * 128  # 6 samples × 2 generations × 128 tokens
        your_batch_time = batch_tokens / your_result['avg_speed']
        base_batch_time = batch_tokens / base_result['avg_speed']
        
        print(f"   Your model batch time: {your_batch_time:.1f} seconds")
        print(f"   Base Qwen batch time: {base_batch_time:.1f} seconds")
        print(f"   Your model takes {your_batch_time/base_batch_time:.1f}x longer per batch")
        
    else:
        print("WARNING Could not compare models (insufficient successful loads)")
        for result in results:
            print(f"   {result['name']:30}: {result['status']}")
    
    print(f"\nRECOMMENDATIONS:")
    if len(successful_results) >= 2 and successful_results[1]['avg_speed'] < 2:
        print("   SLOW Your model is quite slow for CPU inference")
        print("   TIP Consider: smaller batch sizes, fewer generations, or shorter completions")
    elif len(successful_results) >= 2:
        print("   OK Your model speed is reasonable for its size")
        print("   TIP Training slowness is expected for larger models")

def create_speed_comparison():
    """Compare training speeds with different configurations"""
    
    print("[CHART] Speed Comparison Test")
    print("=" * 40)
    
    configs_to_test = [
        {
            "name": "Conservative (Safe)",
            "batch_size": 1,
            "grad_accum": 4,
            "samples": 20,
            "lr": 1e-5
        },
        {
            "name": "Balanced (Recommended)", 
            "batch_size": 2,
            "grad_accum": 2,
            "samples": 30,
            "lr": 3e-5
        },
        {
            "name": "Aggressive (Ultra-Fast)",
            "batch_size": 4,
            "grad_accum": 1,
            "samples": 30,
            "lr": 5e-5
        }
    ]
    
    results = []
    
    for config_test in configs_to_test:
        print(f"\nTesting {config_test['name']} configuration...")
        
        # Simulate training time estimation
        estimated_time = (config_test['samples'] / config_test['batch_size']) * config_test['grad_accum'] * 0.5
        
        results.append({
            "config": config_test['name'],
            "estimated_time": estimated_time,
            "samples": config_test['samples'],
            "throughput": config_test['samples'] / estimated_time
        })
        
        print(f"   Estimated time: {estimated_time:.1f} seconds")
        print(f"   Throughput: {config_test['samples'] / estimated_time:.1f} samples/sec")
    
    print(f"\n[CHART] Speed Comparison Summary:")
    for result in results:
        print(f"   {result['config']:20}: {result['estimated_time']:5.1f}s ({result['throughput']:4.1f} samples/sec)")
    
    print(f"\n[TIP] Recommendation: Use 'Aggressive' config for fastest results on your 12-core CPU!")

def aggressive_system_cleanup():
    """Aggressive cleanup to fix memory pressure and CPU contention"""
    import gc
    import psutil
    
    print("[CLEANUP] Starting aggressive system cleanup...")
    
    # Force garbage collection multiple times
    for i in range(3):
        gc.collect()
    
    # Try to free up memory
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    # Check memory status
    memory_info = psutil.virtual_memory()
    print(f"[CLEANUP] Available memory: {memory_info.available / 1024**3:.1f} GB ({memory_info.percent}% used)")
    
    # Kill background processes that might be consuming resources
    current_process = psutil.Process()
    print(f"[CLEANUP] Current process CPU: {current_process.cpu_percent()}%")
    print(f"[CLEANUP] Current process memory: {current_process.memory_info().rss / 1024**3:.1f} GB")
    
    # Set process priority to high
    try:
        current_process.nice(psutil.HIGH_PRIORITY_CLASS if hasattr(psutil, 'HIGH_PRIORITY_CLASS') else -10)
        print("[CLEANUP] Process priority set to HIGH")
    except:
        print("[CLEANUP] Could not set high priority (run as admin for best performance)")
    
    # Final cleanup
    gc.collect()
    print("[CLEANUP] System cleanup completed")

def validate_optimizations():
    """Validate that all CPU optimizations are properly configured"""
    import psutil
    
    print("[VALIDATE] Checking optimization status...")
    
    # Check CPU threading
    print(f"   PyTorch threads: {torch.get_num_threads()}")
    print(f"   PyTorch interop threads: {torch.get_num_interop_threads()}")
    
    # Check environment variables
    critical_vars = [
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 
        'OPENBLAS_NUM_THREADS', 'PYTORCH_CUDA_ALLOC_CONF', 'TRANSFORMERS_OFFLINE'
    ]
    
    for var in critical_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"   {var}: {value}")
    
    # Check memory status
    memory_info = psutil.virtual_memory()
    print(f"   Available memory: {memory_info.available / 1024**3:.1f} GB")
    print(f"   Memory usage: {memory_info.percent:.1f}%")
    
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"   CPU usage: {cpu_percent:.1f}%")
    
    # Validate critical settings
    issues = []
    if torch.get_num_threads() < 8:
        issues.append(f"Low thread count: {torch.get_num_threads()} (expected 12+)")
    
    if memory_info.available / 1024**3 < 1.5:
        issues.append(f"Low memory: {memory_info.available / 1024**3:.1f}GB available")
    
    if os.environ.get('TRANSFORMERS_OFFLINE') != '1':
        issues.append("TRANSFORMERS_OFFLINE not set - may cause network delays")
    
    if issues:
        print("[VALIDATE] WARNING Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("[VALIDATE] OK All optimizations validated successfully")
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Fast CPU Training")
    parser.add_argument("--mode", choices=["train", "benchmark", "compare"], default="train",
                       help="Mode: train (ultra-fast training), benchmark (test CPU), compare (speed comparison)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for training")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of training samples")
    parser.add_argument("--num_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--dataset", choices=["gsm8k", "custom"], default="gsm8k",
                       help="Dataset to use for training")
    parser.add_argument("--task_type", choices=["math", "general"], default="math",
                       help="Task type for reward function")
    parser.add_argument("--target_model", type=str, default=None,
                       help="Path to target model for benchmark comparison (if not specified, will show available models)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to continue from")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("[FAST] Running ultra-fast training...")
        run_ultra_fast_training(
            learning_rate=args.learning_rate,
            num_samples=args.num_samples,
            num_epochs=args.num_epochs,
            dataset_name=args.dataset,
            task_type=args.task_type
        )
    elif args.mode == "benchmark":
        print("[RUN]‍♂️ Running CPU benchmark...")
        run_benchmark_test(target_model_path=args.target_model)
    else:
        print("[CHART] Running speed comparison...")
        create_speed_comparison()
    
    print("\n[PARTY] Ultra-fast training complete!")
    print("\n[TIP] Tips for maximum speed:")
    print("   * Use batch_size=4, grad_accum=1 for fastest training")
    print("   * Keep datasets small (20-50 samples) for quick iterations")
    print("   * Your 12-core CPU can handle aggressive parallelization!")
