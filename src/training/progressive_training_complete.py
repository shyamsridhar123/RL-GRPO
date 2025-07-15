#!/usr/bin/env python3
"""
Progressive GRPO Training with Ultra-Optimized Integration
Implements curriculum learning with ultra-optimized components
"""

import os
import sys
import json
import time
import torch
import psutil
import gc
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .grpo_trainer import CPUGRPOTrainer, CPUGRPOConfig
from .lightning_fisher import create_optimal_fisher_calculator
from .advanced_memory_optimization import (
    MemoryOptimizationConfig,
    create_memory_optimized_training_setup,
    AdvancedMemoryManager
)


@dataclass
class ProgressiveTrainingConfig:
    """Configuration for progressive GRPO training with ultra-optimizations"""
    
    # Base model settings
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    base_output_dir: str = "./models"
    
    # Progressive curriculum settings
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        {
            "name": "Stage 1: Basic Math",
            "samples": 10,  # Small for testing
            "epochs": 1.0,
            "lr": 1e-5,
            "output_subdir": "stage1"
        },
        {
            "name": "Stage 2: Intermediate Math", 
            "samples": 20,
            "epochs": 1.0,
            "lr": 5e-6,
            "output_subdir": "stage2"
        },
        {
            "name": "Stage 3: Advanced Math",
            "samples": 30,
            "epochs": 1.0,
            "lr": 2e-6,
            "output_subdir": "stage3"
        }
    ])
    
    # Ultra-optimization settings
    enable_ultra_optimizations: bool = True
    enable_lightning_fisher: bool = True
    enable_memory_optimization: bool = True
    enable_quantization: bool = True
    
    # Testing settings
    test_prompts: List[str] = field(default_factory=lambda: [
        "What is 15 + 27?",
        "Solve: 2x + 10 = 30",
        "If a box contains 24 apples and you eat 1/3 of them, how many are left?"
    ])
    
    # Resource limits
    max_memory_usage_percent: float = 85.0
    target_training_time_per_stage: float = 30.0  # seconds


class ProgressiveTrainer:
    """
    Progressive GRPO trainer with ultra-optimized components
    
    Features:
    - Curriculum learning across multiple stages
    - Ultra-optimized Fisher Information and memory management
    - Automatic memory monitoring and adjustment
    - Progressive difficulty scaling
    """
    
    def __init__(self, config: ProgressiveTrainingConfig):
        self.config = config
        self.results = []
        self.current_stage = 0
        self.memory_manager = None
        
        # Initialize memory manager if enabled
        if config.enable_memory_optimization:
            memory_config = MemoryOptimizationConfig(
                enable_quantization=config.enable_quantization,
                enable_gradient_checkpointing=True,
                enable_mixed_precision=True
            )
            self.memory_manager = AdvancedMemoryManager(memory_config)
        
        # Create base output directory
        os.makedirs(config.base_output_dir, exist_ok=True)
    
    def run_progressive_training(self) -> Dict:
        """Run the complete progressive training curriculum"""
        print("🎓 Starting Progressive GRPO Training with Ultra-Optimizations")
        print("=" * 70)
        
        # System info
        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / 1024**3
        print(f"💾 Available memory: {available_gb:.1f}GB")
        print(f"⚡ Ultra-optimizations: {self.config.enable_ultra_optimizations}")
        print(f"🔥 Lightning Fisher: {self.config.enable_lightning_fisher}")
        print(f"📊 Memory optimization: {self.config.enable_memory_optimization}")
        
        previous_model = self.config.model_name
        overall_start_time = time.time()
        
        try:
            for i, stage in enumerate(self.config.curriculum_stages):
                self.current_stage = i
                print(f"\n🚀 {stage['name']} ({i+1}/{len(self.config.curriculum_stages)})")
                print(f"📊 Samples: {stage['samples']}, Epochs: {stage['epochs']}, LR: {stage['lr']}")
                print(f"🔄 Starting from: {previous_model}")
                print("-" * 50)
                
                # Train this stage
                stage_result = self._train_stage(stage, previous_model)
                
                if stage_result['success']:
                    self.results.append(stage_result)
                    previous_model = stage_result['model_path']
                    print(f"✅ {stage['name']} completed in {stage_result['training_time']:.2f}s!")
                else:
                    print(f"❌ {stage['name']} failed: {stage_result['error']}")
                    break
                
                # Memory cleanup between stages
                if self.memory_manager:
                    self.memory_manager.cleanup()
                gc.collect()
                
                # Check memory usage
                memory_usage = psutil.virtual_memory().percent
                if memory_usage > self.config.max_memory_usage_percent:
                    print(f"⚠️ High memory usage: {memory_usage:.1f}%. Pausing for cleanup...")
                    time.sleep(2)
                    gc.collect()
        
        except Exception as e:
            print(f"❌ Progressive training failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Compile final results
        overall_time = time.time() - overall_start_time
        final_results = self._compile_final_results(overall_time, previous_model)
        
        # Save results
        self._save_results(final_results)
        
        return final_results
    
    def _train_stage(self, stage: Dict, previous_model: str) -> Dict:
        """Train a single stage with ultra-optimizations"""
        stage_start_time = time.time()
        
        try:
            # Setup stage output directory
            stage_output_dir = os.path.join(
                self.config.base_output_dir, 
                stage['output_subdir']
            )
            os.makedirs(stage_output_dir, exist_ok=True)
            
            # Configure GRPO trainer
            grpo_config = CPUGRPOConfig()
            grpo_config.model_name = previous_model
            grpo_config.learning_rate = stage['lr']
            grpo_config.num_train_epochs = stage['epochs']
            grpo_config.output_dir = stage_output_dir
            
            # Apply ultra-optimizations if enabled
            if self.config.enable_ultra_optimizations:
                grpo_config = self._apply_ultra_optimizations(grpo_config)
            
            # Initialize trainer
            trainer = CPUGRPOTrainer(grpo_config)
            
            # Apply memory optimizations
            if self.memory_manager:
                trainer = self.memory_manager.optimize_trainer(trainer)
            
            # Prepare dataset
            dataset = trainer.prepare_dataset("gsm8k", num_samples=stage['samples'])
            
            # Create reward function
            reward_fn = trainer.create_reward_function("math")
            
            # Apply Lightning Fisher if enabled
            if self.config.enable_lightning_fisher:
                # Get available memory for Fisher calculation
                available_memory = psutil.virtual_memory().available / 1024**3
                fisher_calculator = create_optimal_fisher_calculator(
                    trainer.model,
                    available_memory_gb=available_memory,
                    dataloader=trainer.eval_dataloader if hasattr(trainer, 'eval_dataloader') else None
                )
                trainer.fisher_calculator = fisher_calculator
            
            # Train the model
            model_path = trainer.train(dataset, reward_fn)
            
            # Test the trained model
            test_results = self._test_stage_model(trainer, stage)
            
            training_time = time.time() - stage_start_time
            
            return {
                'success': True,
                'stage_name': stage['name'],
                'stage_config': stage,
                'training_time': training_time,
                'model_path': model_path,
                'test_results': test_results,
                'memory_usage': psutil.virtual_memory().percent
            }
            
        except Exception as e:
            return {
                'success': False,
                'stage_name': stage['name'],
                'error': str(e),
                'training_time': time.time() - stage_start_time
            }
    
    def _apply_ultra_optimizations(self, config: CPUGRPOConfig) -> CPUGRPOConfig:
        """Apply ultra-optimizations to GRPO config"""
        # Enable all CPU optimizations
        config.dataloader_num_workers = min(4, os.cpu_count())
        config.gradient_accumulation_steps = 1
        config.eval_accumulation_steps = 1
        config.dataloader_pin_memory = True
        config.bf16 = False  # Use FP16 for CPU
        config.fp16 = True
        
        # Memory optimizations
        config.max_grad_norm = 1.0
        config.gradient_checkpointing = True
        config.remove_unused_columns = True
        
        return config
    
    def _test_stage_model(self, trainer: CPUGRPOTrainer, stage: Dict) -> List[Dict]:
        """Test the trained model on validation prompts"""
        test_results = []
        
        for prompt in self.config.test_prompts:
            try:
                response = trainer.generate_response(prompt, max_new_tokens=100)
                test_results.append({
                    'prompt': prompt,
                    'response': response,
                    'response_length': len(response),
                    'success': True
                })
                print(f"✅ Test: {prompt}")
                print(f"   Response: {response[:80]}...")
            except Exception as e:
                test_results.append({
                    'prompt': prompt,
                    'error': str(e),
                    'success': False
                })
                print(f"❌ Test failed: {prompt} -> {str(e)}")
        
        return test_results
    
    def _compile_final_results(self, total_time: float, final_model: str) -> Dict:
        """Compile final results summary"""
        successful_stages = [r for r in self.results if r['success']]
        total_training_time = sum(r['training_time'] for r in successful_stages)
        
        return {
            'progressive_training_completed': datetime.now().isoformat(),
            'total_time': total_time,
            'total_training_time': total_training_time,
            'stages_completed': len(successful_stages),
            'total_stages': len(self.config.curriculum_stages),
            'final_model_path': final_model,
            'ultra_optimizations_used': self.config.enable_ultra_optimizations,
            'lightning_fisher_used': self.config.enable_lightning_fisher,
            'memory_optimization_used': self.config.enable_memory_optimization,
            'stages': self.results,
            'summary': {
                'avg_training_time_per_stage': total_training_time / len(successful_stages) if successful_stages else 0,
                'successful_stages': len(successful_stages),
                'failed_stages': len(self.results) - len(successful_stages),
                'total_samples_trained': sum(r['stage_config']['samples'] for r in successful_stages)
            }
        }
    
    def _save_results(self, results: Dict):
        """Save training results to file"""
        results_file = os.path.join(
            self.config.base_output_dir, 
            'progressive_training_results.json'
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Progressive Training Summary:")
        print(f"📊 Completed {results['stages_completed']}/{results['total_stages']} stages")
        print(f"⏱️ Total time: {results['total_time']:.2f}s")
        print(f"🏃 Training time: {results['total_training_time']:.2f}s")
        print(f"💾 Final model: {results['final_model_path']}")
        print(f"📄 Results saved to: {results_file}")
        
        if results['stages_completed'] == results['total_stages']:
            print("✅ All stages completed successfully!")
        else:
            print(f"⚠️ {results['total_stages'] - results['stages_completed']} stages failed")


def run_progressive_training_test(num_samples_per_stage: int = 10) -> Dict:
    """
    Run a smaller test version of progressive training
    
    Args:
        num_samples_per_stage: Number of samples per stage (keep small for testing)
    
    Returns:
        Dict with training results
    """
    print(f"🧪 Running Progressive Training Test (samples per stage: {num_samples_per_stage})")
    
    # Create test configuration
    test_config = ProgressiveTrainingConfig(
        model_name="Qwen/Qwen2-0.5B-Instruct",
        base_output_dir="./models",
        curriculum_stages=[
            {
                "name": "Test Stage 1: Basic Math",
                "samples": num_samples_per_stage,
                "epochs": 1.0,
                "lr": 1e-5,
                "output_subdir": "test_stage1"
            },
            {
                "name": "Test Stage 2: Intermediate Math",
                "samples": num_samples_per_stage,
                "epochs": 1.0,
                "lr": 5e-6,
                "output_subdir": "test_stage2"
            }
        ],
        enable_ultra_optimizations=True,
        enable_lightning_fisher=True,
        enable_memory_optimization=True,
        max_memory_usage_percent=85.0,
        target_training_time_per_stage=30.0
    )
    
    # Run progressive training
    trainer = ProgressiveTrainer(test_config)
    results = trainer.run_progressive_training()
    
    return results


if __name__ == "__main__":
    # Run test with small samples
    results = run_progressive_training_test(num_samples_per_stage=5)
    
    if results['stages_completed'] > 0:
        print("\n🎉 Progressive training test completed successfully!")
    else:
        print("\n❌ Progressive training test failed!")
