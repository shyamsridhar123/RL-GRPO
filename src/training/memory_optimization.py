"""
Memory Optimization Module for Progressive Training
Implements aggressive memory optimization techniques for CPU-based training.
"""

import gc
import torch
import psutil
import tracemalloc
import warnings
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, TrainingArguments
)
import sys
import os

class MemoryProfiler:
    """Advanced memory profiling and optimization for training."""
    
    def __init__(self):
        self.tracemalloc_started = False
        self.baseline_memory = None
        
    def start_profiling(self):
        """Start memory profiling."""
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True
            self.baseline_memory = self.get_memory_stats()
            print(f"Memory profiling started. Baseline: {self.baseline_memory['memory_percent']:.1f}%")
    
    def stop_profiling(self):
        """Stop memory profiling."""
        if self.tracemalloc_started:
            tracemalloc.stop()
            self.tracemalloc_started = False
            print("Memory profiling stopped.")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics."""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        process = psutil.Process()
        
        stats = {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'swap_percent': swap.percent,
            'swap_used_gb': swap.used / (1024**3),
            'process_memory_gb': process.memory_info().rss / (1024**3),
            'process_memory_percent': process.memory_percent()
        }
        
        # Add tracemalloc info if available
        if self.tracemalloc_started:
            current, peak = tracemalloc.get_traced_memory()
            stats['tracemalloc_current_mb'] = current / (1024**2)
            stats['tracemalloc_peak_mb'] = peak / (1024**2)
        
        return stats
    
    def get_top_memory_consumers(self, limit: int = 10) -> list:
        """Get top memory consuming code locations."""
        if not self.tracemalloc_started:
            return []
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        consumers = []
        for index, stat in enumerate(top_stats[:limit]):
            consumers.append({
                'rank': index + 1,
                'size_mb': stat.size / (1024**2),
                'count': stat.count,
                'traceback': stat.traceback.format()
            })
        
        return consumers
    
    def check_memory_health(self) -> Dict[str, Any]:
        """Check memory health and provide recommendations."""
        stats = self.get_memory_stats()
        health = {
            'status': 'healthy',
            'warnings': [],
            'recommendations': [],
            'stats': stats
        }
        
        # Critical memory usage
        if stats['memory_percent'] > 95:
            health['status'] = 'critical'
            health['warnings'].append("CRITICAL: System memory usage > 95%")
            health['recommendations'].append("Consider model quantization or smaller batch size")
        elif stats['memory_percent'] > 90:
            health['status'] = 'warning'
            health['warnings'].append("WARNING: System memory usage > 90%")
            health['recommendations'].append("Monitor memory usage closely")
        
        # Swap usage
        if stats['swap_percent'] > 20:
            health['status'] = 'critical'
            health['warnings'].append(f"CRITICAL: High swap usage ({stats['swap_percent']:.1f}%)")
            health['recommendations'].append("Reduce memory footprint immediately")
        elif stats['swap_percent'] > 10:
            health['status'] = 'warning'
            health['warnings'].append(f"WARNING: Moderate swap usage ({stats['swap_percent']:.1f}%)")
        
        # Process memory
        if stats['process_memory_percent'] > 50:
            health['warnings'].append(f"High process memory usage ({stats['process_memory_percent']:.1f}%)")
            health['recommendations'].append("Consider gradient checkpointing")
        
        return health

class ModelOptimizer:
    """Optimizes model loading and memory usage."""
    
    @staticmethod
    def get_optimized_model_config(model_name: str, optimize_for: str = "memory") -> Dict[str, Any]:
        """Get optimized configuration for model loading."""
        config = {
            'torch_dtype': torch.float16,  # Use half precision
            'device_map': "cpu",
            'low_cpu_mem_usage': True,
            'trust_remote_code': True
        }
        
        if optimize_for == "memory":
            # Maximum memory optimization
            config.update({
                'torch_dtype': torch.float16,
                'use_cache': False  # Disable KV cache
            })
        elif optimize_for == "speed":
            # Balance memory and speed
            config.update({
                'torch_dtype': torch.bfloat16,
                'use_cache': True
            })
        
        return config
    
    @staticmethod
    def load_optimized_model(model_name: str, optimize_for: str = "memory") -> Tuple[Any, Any]:
        """Load model with memory optimizations."""
        print(f"Loading optimized model: {model_name} (optimize_for={optimize_for})")
        
        # Get optimized config
        config = ModelOptimizer.get_optimized_model_config(model_name, optimize_for)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimizations
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **config
            )
            
            # Apply additional optimizations after loading
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = False
            
            # Enable gradient checkpointing if available and optimizing for memory
            if optimize_for == "memory" and hasattr(model, 'gradient_checkpointing_enable'):
                try:
                    model.gradient_checkpointing_enable()
                    print("Gradient checkpointing enabled")
                except Exception as e:
                    print(f"Could not enable gradient checkpointing: {e}")
            
            # Move to CPU and optimize
            model = model.to('cpu')
            model.eval()
            
            print(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
            return model, tokenizer
            
        except Exception as e:
            print(f"Error loading model with optimizations: {e}")
            print("Falling back to basic model loading...")
            
            # Fallback to basic loading
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = False
            
            model = model.to('cpu')
            model.eval()
            
            print(f"Model loaded (fallback) with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
            return model, tokenizer
    
    @staticmethod
    def optimize_training_args(**kwargs) -> TrainingArguments:
        """Create memory-optimized training arguments."""
        default_args = {
            'output_dir': './tmp_trainer',
            'num_train_epochs': 1,
            'per_device_train_batch_size': 1,  # Minimal batch size
            'gradient_accumulation_steps': 1,
            'warmup_steps': 0,
            'logging_steps': 1,
            'save_steps': 1000,
            'eval_steps': 1000,
            'save_total_limit': 1,
            'remove_unused_columns': True,
            'dataloader_pin_memory': False,  # Disable pin memory for CPU
            'dataloader_num_workers': 0,     # No parallel workers
            'fp16': False,  # Disable FP16 on CPU
            'bf16': False,  # Disable BF16 on CPU
            'gradient_checkpointing': True,
            'optim': "adamw_torch",
            'learning_rate': 1e-5,
            'lr_scheduler_type': "constant",
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'report_to': [],  # Disable reporting
            'push_to_hub': False,
            'hub_token': None,
            'hub_model_id': None,
        }
        
        # Override with user arguments
        default_args.update(kwargs)
        
        return TrainingArguments(**default_args)

class MemoryManager:
    """Comprehensive memory management for progressive training."""
    
    def __init__(self):
        self.profiler = MemoryProfiler()
        self.checkpoints = []
        
    def aggressive_cleanup(self) -> Dict[str, float]:
        """Perform aggressive memory cleanup."""
        print("Performing aggressive memory cleanup...")
        
        # Get memory before cleanup
        before = self.profiler.get_memory_stats()
        
        # Multiple garbage collection passes
        for i in range(3):
            collected = gc.collect()
            if collected > 0:
                print(f"GC pass {i+1}: collected {collected} objects")
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection again
        gc.collect()
        
        # Clear internal caches
        if hasattr(torch, 'clear_autocast_cache'):
            torch.clear_autocast_cache()
        
        # Get memory after cleanup
        after = self.profiler.get_memory_stats()
        
        # Calculate freed memory
        freed_mb = (before['process_memory_gb'] - after['process_memory_gb']) * 1024
        freed_percent = before['memory_percent'] - after['memory_percent']
        
        cleanup_stats = {
            'memory_freed_mb': freed_mb,
            'memory_freed_percent': freed_percent,
            'before_memory_percent': before['memory_percent'],
            'after_memory_percent': after['memory_percent'],
            'before_process_gb': before['process_memory_gb'],
            'after_process_gb': after['process_memory_gb']
        }
        
        print(f"Memory cleanup completed:")
        print(f"  - Freed: {freed_mb:.1f} MB ({freed_percent:.1f}%)")
        print(f"  - System memory: {before['memory_percent']:.1f}% -> {after['memory_percent']:.1f}%")
        print(f"  - Process memory: {before['process_memory_gb']:.1f}GB -> {after['process_memory_gb']:.1f}GB")
        
        return cleanup_stats
    
    def create_memory_checkpoint(self, name: str) -> Dict[str, Any]:
        """Create a memory usage checkpoint."""
        checkpoint = {
            'name': name,
            'stats': self.profiler.get_memory_stats(),
            'timestamp': psutil.boot_time(),
            'top_consumers': self.profiler.get_top_memory_consumers(5)
        }
        
        self.checkpoints.append(checkpoint)
        print(f"Memory checkpoint '{name}': {checkpoint['stats']['memory_percent']:.1f}% used")
        
        return checkpoint
    
    def compare_checkpoints(self, checkpoint1: str, checkpoint2: str) -> Dict[str, float]:
        """Compare two memory checkpoints."""
        cp1 = next((cp for cp in self.checkpoints if cp['name'] == checkpoint1), None)
        cp2 = next((cp for cp in self.checkpoints if cp['name'] == checkpoint2), None)
        
        if not cp1 or not cp2:
            return {}
        
        comparison = {}
        for key in cp1['stats']:
            if key in cp2['stats']:
                comparison[f"{key}_diff"] = cp2['stats'][key] - cp1['stats'][key]
        
        return comparison
    
    def get_memory_recommendations(self) -> list:
        """Get memory optimization recommendations."""
        health = self.profiler.check_memory_health()
        recommendations = health['recommendations'].copy()
        
        # Add specific recommendations based on current state
        stats = health['stats']
        
        if stats['memory_percent'] > 90:
            recommendations.extend([
                "Use a smaller model (e.g., distilbert-base-uncased)",
                "Reduce batch size to 1",
                "Enable gradient checkpointing",
                "Use model quantization (int8/fp16)"
            ])
        
        if stats['swap_percent'] > 15:
            recommendations.extend([
                "Increase system RAM",
                "Close other applications",
                "Use model sharding techniques"
            ])
        
        if stats['process_memory_percent'] > 40:
            recommendations.extend([
                "Implement gradient accumulation",
                "Use memory-efficient optimizers",
                "Consider model parallelism"
            ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def monitor_memory_during_training(self, callback_fn=None):
        """Monitor memory usage during training with optional callback."""
        def memory_callback():
            health = self.profiler.check_memory_health()
            
            if health['status'] == 'critical':
                print("CRITICAL MEMORY SITUATION:")
                for warning in health['warnings']:
                    print(f"  - {warning}")
                
                if callback_fn:
                    callback_fn(health)
                
                # Force cleanup
                self.aggressive_cleanup()
        
        return memory_callback

def get_recommended_model_for_system() -> str:
    """Recommend appropriate model based on system memory, prioritizing GRPO-suitable models."""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    print(f"Available system memory: {available_gb:.1f} GB")
    
    # For GRPO fine-tuning, we need models that can do mathematical reasoning
    # Priority: Keep Qwen2-0.5B-Instruct if possible, as it's designed for instruction following
    
    if available_gb < 1.5:
        recommended = "microsoft/DialoGPT-small"  # Fallback only if critically low
        print(f"CRITICAL: Use lightweight model '{recommended}' due to severely limited memory")
        print(f"WARNING: This may not be suitable for GRPO mathematical reasoning training")
    elif available_gb < 3.0:
        recommended = "Qwen/Qwen2-0.5B-Instruct"  # Keep the target model
        print(f"RECOMMENDATION: Continue with '{recommended}' but with aggressive memory optimization")
        print(f"INFO: This model is suitable for GRPO fine-tuning on mathematical reasoning")
    else:
        recommended = "Qwen/Qwen2-0.5B-Instruct"  # Original choice - ideal for GRPO
        print(f"RECOMMENDATION: System can handle '{recommended}' with standard optimizations")
        print(f"INFO: Optimal choice for GRPO fine-tuning on GSM8K mathematical reasoning")
    
    return recommended

if __name__ == "__main__":
    # Test memory optimization tools
    print("Testing Memory Optimization Module")
    print("=" * 50)
    
    # Initialize memory manager
    manager = MemoryManager()
    manager.profiler.start_profiling()
    
    # Check initial memory health
    health = manager.profiler.check_memory_health()
    print(f"Initial memory health: {health['status']}")
    for warning in health['warnings']:
        print(f"  - {warning}")
    
    # Create checkpoint
    manager.create_memory_checkpoint("initial")
    
    # Get model recommendation
    recommended_model = get_recommended_model_for_system()
    
    # Test aggressive cleanup
    cleanup_stats = manager.aggressive_cleanup()
    
    # Final checkpoint
    manager.create_memory_checkpoint("after_cleanup")
    
    # Compare checkpoints
    comparison = manager.compare_checkpoints("initial", "after_cleanup")
    print("\nMemory checkpoint comparison:")
    for key, value in comparison.items():
        print(f"  {key}: {value:.2f}")
    
    # Get recommendations
    recommendations = manager.get_memory_recommendations()
    print(f"\nMemory optimization recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    manager.profiler.stop_profiling()
    print("\nMemory optimization testing completed.")
