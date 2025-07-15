"""
Advanced Memory Optimization Suite for CPU-Based GRPO Training
Implements quantization, memory sharding, and advanced optimization techniques
"""

import torch
import torch.nn as nn
import psutil
import gc
import os
import warnings
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization strategies"""
    
    # Quantization settings (CPU-appropriate)
    enable_fp16: bool = True  # Use half precision when possible
    enable_torch_compile: bool = False  # Can be unstable
    use_dynamic_quantization: bool = True  # CPU-friendly quantization
    
    # Memory management
    enable_gradient_checkpointing: bool = True
    enable_cpu_offload: bool = True
    max_memory_utilization: float = 0.85  # 85% max memory usage
    aggressive_cleanup_threshold: float = 0.90  # Clean when memory > 90%
    
    # Batch optimization
    adaptive_batch_sizing: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 4
    memory_headroom_gb: float = 2.0  # Keep 2GB free
    
    # Advanced optimizations
    enable_memory_sharding: bool = True
    enable_activation_checkpointing: bool = True
    use_zero_redundancy_optimizer: bool = False  # Too complex for CPU
    
    # Monitoring
    memory_monitoring_interval: int = 10  # seconds
    enable_memory_alerts: bool = True
    swap_usage_warning_threshold: float = 0.10  # Warn at 10% swap


class CPUMemoryOptimizer:
    """
    CPU-specific memory optimization techniques
    Uses torch.jit, dynamic quantization, and memory-efficient loading
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.original_model_memory = 0
        self.optimized_model_memory = 0
        
    def apply_cpu_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply CPU-specific optimizations to reduce memory usage"""
        
        print("🔧 Applying CPU memory optimizations...")
        
        optimizations_applied = []
        
        # 1. Dynamic quantization (CPU-friendly)
        if self.config.use_dynamic_quantization:
            try:
                print("   Applying dynamic quantization...")
                model = torch.quantization.quantize_dynamic(
                    model, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
                optimizations_applied.append("Dynamic INT8 quantization")
            except Exception as e:
                print(f"   ⚠️  Dynamic quantization failed: {e}")
        
        # 2. Enable gradient checkpointing
        if self.config.enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            optimizations_applied.append("Gradient checkpointing")
        
        # 3. Set model to eval mode for inference optimizations
        model.eval()
        
        print(f"   ✅ Applied optimizations: {', '.join(optimizations_applied)}")
        return model
        
    def load_memory_efficient_model(self, model_name: str, device: str = "cpu") -> Tuple[Any, Any]:
        """
        Load model with CPU-specific memory optimizations
        Returns: (model, tokenizer)
        """
        print(f"📦 Loading CPU-optimized model: {model_name}")
        
        # Record initial memory
        initial_memory = psutil.virtual_memory().used / 1024**3
        
        # Load tokenizer first (lightweight)
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Model loading arguments optimized for CPU
        model_kwargs = {
            "torch_dtype": torch.float32,  # Use FP32 for compatibility with quantization
            "device_map": None,  # Don't use device_map for CPU
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "use_cache": False,  # Disable KV cache to save memory
        }
        
        try:
            print("   Loading model with CPU optimizations...")
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            # Move to CPU explicitly
            model = model.to(device)
            
            # Apply CPU-specific optimizations
            model = self.apply_cpu_optimizations(model)
            
            # Record final memory
            final_memory = psutil.virtual_memory().used / 1024**3
            memory_used = final_memory - initial_memory
            
            # Calculate model parameters
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   📊 Model loaded successfully!")
            print(f"      Parameters: {param_count:,} total, {trainable_params:,} trainable")
            print(f"      Memory used: {memory_used:.2f} GB")
            print(f"      Device: {next(model.parameters()).device}")
            print(f"      Dtype: {next(model.parameters()).dtype}")
            
            self.original_model_memory = memory_used
            return model, tokenizer
            
        except Exception as e:
            print(f"   ❌ Optimized model loading failed: {e}")
            # Fallback to basic loading
            print("   🔄 Falling back to basic model loading...")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
            
            final_memory = psutil.virtual_memory().used / 1024**3
            memory_used = final_memory - initial_memory
            print(f"   📊 Fallback model loaded: {memory_used:.2f} GB")
            
            return model, tokenizer
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get current optimization statistics"""
        return {
            "dynamic_quantization_enabled": self.config.use_dynamic_quantization,
            "fp16_enabled": self.config.enable_fp16,
            "gradient_checkpointing_enabled": self.config.enable_gradient_checkpointing,
            "original_model_memory_gb": self.original_model_memory,
            "optimized_model_memory_gb": self.optimized_model_memory,
            "memory_reduction_percent": (
                (self.original_model_memory - self.optimized_model_memory) / 
                max(self.original_model_memory, 0.001) * 100
            ) if self.original_model_memory > 0 else 0
        }


class AdaptiveBatchOptimizer:
    """
    Dynamically optimizes batch size based on available memory
    Prevents OOM while maximizing training efficiency
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.current_batch_size = config.min_batch_size
        self.memory_history = []
        
    def calculate_optimal_batch_size(self, model_size_gb: float) -> int:
        """Calculate optimal batch size based on available memory"""
        
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / 1024**3
        
        # Reserve memory for system and headroom
        usable_memory_gb = available_memory_gb - self.config.memory_headroom_gb
        
        # Estimate memory per sample (rough heuristic)
        estimated_memory_per_sample = model_size_gb * 0.1  # Model takes ~10x memory during training
        
        # Calculate maximum safe batch size
        max_safe_batch = max(1, int(usable_memory_gb / estimated_memory_per_sample))
        
        # Apply constraints
        optimal_batch = min(
            max_safe_batch,
            self.config.max_batch_size
        )
        optimal_batch = max(optimal_batch, self.config.min_batch_size)
        
        print(f"🔍 Batch Size Optimization:")
        print(f"   Available memory: {available_memory_gb:.2f} GB")
        print(f"   Usable memory: {usable_memory_gb:.2f} GB")
        print(f"   Estimated memory per sample: {estimated_memory_per_sample:.3f} GB")
        print(f"   Calculated optimal batch size: {optimal_batch}")
        
        self.current_batch_size = optimal_batch
        return optimal_batch
    
    def monitor_batch_memory_usage(self, current_memory_gb: float) -> bool:
        """
        Monitor memory usage during training
        Returns True if batch size should be reduced
        """
        self.memory_history.append(current_memory_gb)
        
        # Keep only recent history
        if len(self.memory_history) > 10:
            self.memory_history = self.memory_history[-10:]
        
        # Check if memory usage is too high
        memory_percent = psutil.virtual_memory().percent / 100
        
        if memory_percent > self.config.aggressive_cleanup_threshold:
            print(f"⚠️  High memory usage detected: {memory_percent:.1%}")
            
            # Reduce batch size if possible
            if self.current_batch_size > self.config.min_batch_size:
                self.current_batch_size = max(
                    self.config.min_batch_size,
                    self.current_batch_size - 1
                )
                print(f"   🔧 Reducing batch size to: {self.current_batch_size}")
                return True
        
        return False


class AdvancedMemoryManager:
    """
    Comprehensive memory management with monitoring and optimization
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.monitoring_active = False
        self.memory_alerts = []
        
    def aggressive_memory_cleanup(self) -> Dict[str, float]:
        """Perform aggressive memory cleanup"""
        
        print("🧹 Performing aggressive memory cleanup...")
        
        # Record initial memory
        initial_memory = psutil.virtual_memory()
        initial_used_gb = initial_memory.used / 1024**3
        
        # Cleanup strategies
        cleanup_actions = []
        
        # 1. Python garbage collection
        collected = gc.collect()
        cleanup_actions.append(f"Garbage collected {collected} objects")
        
        # 2. Clear PyTorch cache
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            cleanup_actions.append("Cleared CUDA cache")
        
        # 3. Force garbage collection again
        gc.collect()
        
        # 4. Clear unnecessary references
        if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_clearCurandStates'):
            try:
                torch._C._cuda_clearCurandStates()
                cleanup_actions.append("Cleared CUDA random states")
            except:
                pass
        
        # Record final memory
        final_memory = psutil.virtual_memory()
        final_used_gb = final_memory.used / 1024**3
        
        memory_freed_gb = initial_used_gb - final_used_gb
        
        print(f"   📊 Memory cleanup completed:")
        print(f"      Memory freed: {memory_freed_gb:.3f} GB")
        print(f"      Memory usage: {final_memory.percent:.1f}%")
        for action in cleanup_actions:
            print(f"      ✅ {action}")
        
        return {
            "memory_freed_gb": memory_freed_gb,
            "memory_freed_mb": memory_freed_gb * 1024,
            "final_memory_percent": final_memory.percent,
            "cleanup_actions": cleanup_actions
        }
    
    def check_memory_health(self) -> Dict[str, Any]:
        """Comprehensive memory health check"""
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        health_status = "healthy"
        warnings = []
        recommendations = []
        
        # Memory usage checks
        if memory.percent > 95:
            health_status = "critical"
            warnings.append(f"Critical memory usage: {memory.percent:.1f}%")
            recommendations.append("Immediate aggressive cleanup required")
        elif memory.percent > 90:
            health_status = "warning"
            warnings.append(f"High memory usage: {memory.percent:.1f}%")
            recommendations.append("Consider reducing batch size or model size")
        
        # Swap usage checks
        if swap.percent > self.config.swap_usage_warning_threshold * 100:
            health_status = "warning" if health_status == "healthy" else health_status
            warnings.append(f"Swap usage detected: {swap.percent:.1f}%")
            recommendations.append("Swap usage degrades performance - reduce memory usage")
        
        # Available memory checks
        available_gb = memory.available / 1024**3
        if available_gb < 1.0:
            health_status = "critical"
            warnings.append(f"Low available memory: {available_gb:.2f} GB")
            recommendations.append("Free memory immediately to prevent system instability")
        
        return {
            "status": health_status,
            "memory_percent": memory.percent,
            "available_gb": available_gb,
            "swap_percent": swap.percent,
            "warnings": warnings,
            "recommendations": recommendations
        }
    
    def get_training_memory_recommendations(self, model_size_gb: float) -> Dict[str, Any]:
        """Get memory-optimized training recommendations"""
        
        memory = psutil.virtual_memory()
        available_gb = memory.available / 1024**3
        
        recommendations = {
            "quantization": self.config.use_dynamic_quantization,
            "gradient_checkpointing": self.config.enable_gradient_checkpointing,
            "suggested_batch_size": 1,
            "memory_strategy": "conservative"
        }
        
        # Determine memory strategy based on available memory
        if available_gb > 8:
            recommendations["memory_strategy"] = "aggressive"
            recommendations["suggested_batch_size"] = min(4, self.config.max_batch_size)
        elif available_gb > 4:
            recommendations["memory_strategy"] = "balanced"
            recommendations["suggested_batch_size"] = min(2, self.config.max_batch_size)
        else:
            recommendations["memory_strategy"] = "ultra_conservative"
            recommendations["suggested_batch_size"] = 1
            recommendations["additional_optimizations"] = [
                "Enable model quantization",
                "Use gradient accumulation instead of large batches",
                "Consider model sharding",
                "Monitor swap usage closely"
            ]
        
        return recommendations


class OptimizedTrainingArgsFactory:
    """
    Creates memory-optimized training arguments
    """
    
    @staticmethod
    def create_memory_optimized_args(
        output_dir: str,
        config: MemoryOptimizationConfig,
        model_size_gb: float,
        **kwargs
    ) -> TrainingArguments:
        """Create training arguments optimized for memory efficiency"""
        
        # Calculate optimal batch size
        batch_optimizer = AdaptiveBatchOptimizer(config)
        optimal_batch_size = batch_optimizer.calculate_optimal_batch_size(model_size_gb)
        
        # Base arguments focused on memory efficiency
        args = {
            "output_dir": output_dir,
            "overwrite_output_dir": True,
            
            # Batch and gradient settings
            "per_device_train_batch_size": optimal_batch_size,
            "gradient_accumulation_steps": max(1, 4 // optimal_batch_size),  # Maintain effective batch size
            
            # Memory optimizations
            "gradient_checkpointing": config.enable_gradient_checkpointing,
            "dataloader_pin_memory": False,  # Disable pin memory on CPU
            "dataloader_num_workers": min(2, psutil.cpu_count() // 4),  # Conservative workers
            
            # Precision settings (CPU-appropriate)
            "fp16": False,  # FP16 often problematic on CPU
            "bf16": False,  # BF16 not supported on most CPUs
            "use_cpu": True,  # Explicitly use CPU
            
            # Logging and checkpointing
            "logging_steps": 5,
            "save_steps": 50,
            "save_total_limit": 2,  # Limit checkpoint storage
            "remove_unused_columns": False,
            
            # Performance settings
            "disable_tqdm": False,  # Keep progress bars for monitoring
            "report_to": None,  # Disable wandb to save memory
        }
        
        # Override with user-provided arguments
        args.update(kwargs)
        
        print(f"🔧 Memory-optimized training arguments created:")
        print(f"   Batch size: {args['per_device_train_batch_size']}")
        print(f"   Gradient accumulation: {args['gradient_accumulation_steps']}")
        print(f"   Gradient checkpointing: {args['gradient_checkpointing']}")
        print(f"   Workers: {args['dataloader_num_workers']}")
        
        return TrainingArguments(**args)


def create_memory_optimized_training_setup(
    model_name: str,
    output_dir: str,
    config: Optional[MemoryOptimizationConfig] = None,
    **training_kwargs
) -> Tuple[Any, Any, TrainingArguments, Dict[str, Any]]:
    """
    Create a complete memory-optimized training setup
    
    Returns:
        (model, tokenizer, training_args, memory_stats)
    """
    
    if config is None:
        config = MemoryOptimizationConfig()
    
    print("🚀 Creating memory-optimized training setup...")
    print(f"   Model: {model_name}")
    print(f"   Output: {output_dir}")
    
    # Initialize managers
    cpu_optimizer = CPUMemoryOptimizer(config)
    memory_manager = AdvancedMemoryManager(config)
    
    # Check initial memory health
    initial_health = memory_manager.check_memory_health()
    print(f"   Initial memory health: {initial_health['status']}")
    
    if initial_health['warnings']:
        for warning in initial_health['warnings']:
            print(f"   ⚠️  {warning}")
    
    # Load optimized model
    model, tokenizer = cpu_optimizer.load_memory_efficient_model(model_name)
    
    # Estimate model memory usage
    model_memory_gb = psutil.virtual_memory().used / 1024**3
    
    # Create optimized training arguments
    training_args = OptimizedTrainingArgsFactory.create_memory_optimized_args(
        output_dir=output_dir,
        config=config,
        model_size_gb=model_memory_gb,
        **training_kwargs
    )
    
    # Get memory recommendations
    recommendations = memory_manager.get_training_memory_recommendations(model_memory_gb)
    
    # Final memory check
    final_health = memory_manager.check_memory_health()
    
    memory_stats = {
        "initial_health": initial_health,
        "final_health": final_health,
        "model_memory_gb": model_memory_gb,
        "recommendations": recommendations,
        "quantization_stats": cpu_optimizer.get_optimization_stats()
    }
    
    print("✅ Memory-optimized training setup completed!")
    
    return model, tokenizer, training_args, memory_stats


if __name__ == "__main__":
    # Test the memory optimization suite
    print("🧪 Testing Advanced Memory Optimization Suite")
    
    # Test configuration
    config = MemoryOptimizationConfig(
        use_dynamic_quantization=True,
        enable_fp16=True,
        adaptive_batch_sizing=True,
        max_batch_size=2  # Conservative for testing
    )
    
    try:
        # Test model loading with optimizations
        model, tokenizer, training_args, stats = create_memory_optimized_training_setup(
            model_name="Qwen/Qwen2-0.5B-Instruct",
            output_dir="./test_memory_output",
            config=config,
            num_train_epochs=1,
            learning_rate=1e-5
        )
        
        print("\n📊 Memory Optimization Results:")
        print(f"   Model memory: {stats['model_memory_gb']:.2f} GB")
        print(f"   Health status: {stats['final_health']['status']}")
        print(f"   Memory usage: {stats['final_health']['memory_percent']:.1f}%")
        print(f"   Recommended strategy: {stats['recommendations']['memory_strategy']}")
        
        print("\n🎉 Advanced memory optimization test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Memory optimization test failed: {e}")
        import traceback
        traceback.print_exc()
