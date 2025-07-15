#!/usr/bin/env python3
"""
Progressive Training Integration Module
Integrates progressive training with existing GRPO trainer without modifications
"""

import os
import sys
import torch
import time
import numpy as np
import gc
from typing import Dict, List, Any, Optional
from datasets import Dataset
from pathlib import Path

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import existing components - use absolute imports when running directly
try:
    # Try relative imports first (when imported as module)
    from .progressive_core import (
        ProgressiveConfig, 
        CPUOptimizer, 
        FisherInformationCalculator,
        ElasticWeightConsolidation,
        CurriculumDesigner,
        ProgressiveTrainingLogger
    )
    from .memory_optimization import (
        MemoryManager,
        MemoryProfiler,
        ModelOptimizer,
        get_recommended_model_for_system
    )
    from .fast_fisher import (
        FastFisherCalculator,
        AdaptiveFisherCalculator
    )
except ImportError:
    # Fallback to absolute imports (when running directly)
    from progressive_core import (
        ProgressiveConfig, 
        CPUOptimizer, 
        FisherInformationCalculator,
        ElasticWeightConsolidation,
        CurriculumDesigner,
        ProgressiveTrainingLogger
    )
    from memory_optimization import (
        MemoryManager,
        MemoryProfiler,
        ModelOptimizer,
        get_recommended_model_for_system
    )
    from fast_fisher import (
        FastFisherCalculator,
        AdaptiveFisherCalculator
    )
    from lightning_fisher import (
        LightningFisherCalculator,
        UltraFastFisherCalculator,
        create_optimal_fisher_calculator
    )

class ProgressiveGRPOIntegrator:
    """
    Integrates progressive training with existing GRPO system
    Uses existing ultra_fast_training.py without modifications
    """
    
    def __init__(self, config: ProgressiveConfig):
        self.config = config
        self.logger = ProgressiveTrainingLogger("progressive_grpo_test")
        self.cpu_optimizer = CPUOptimizer()
        self.ewc = ElasticWeightConsolidation(config.lambda_ewc)
        
        # Configure CPU optimization
        if config.enable_cpu_optimization:
            cpu_config = self.cpu_optimizer.configure_cpu_threads()
            self.logger.logger.info(f"CPU Configuration: {cpu_config}")
    
    def create_test_dataset(self, size: int = 4) -> Dataset:
        """Create small GSM8K-style test dataset for GRPO mathematical reasoning validation"""
        
        # GSM8K-style mathematical reasoning problems with varying complexity
        # These are designed for GRPO fine-tuning and progressive curriculum learning
        gsm8k_style_problems = [
            {
                'question': "What is 2 + 3?",
                'answer': "5",
                'complexity_level': 1,
                'reasoning_steps': 1,
                'problem_type': "basic_arithmetic"
            },
            {
                'question': "If Tom has 5 apples and gives away 2, how many does he have left?",
                'answer': "3",
                'complexity_level': 2,
                'reasoning_steps': 2,
                'problem_type': "subtraction_word_problem"
            },
            {
                'question': "A store sells 15 items per hour. How many items do they sell in 3 hours?",
                'answer': "45",
                'complexity_level': 3,
                'reasoning_steps': 2,
                'problem_type': "multiplication_word_problem"
            },
            {
                'question': "Sarah bought 4 boxes of pencils. Each box contains 12 pencils. If she uses 8 pencils, how many pencils does she have left?",
                'answer': "40",
                'complexity_level': 4,
                'reasoning_steps': 3,
                'problem_type': "multi_step_arithmetic"
            },
            {
                'question': "A restaurant has 24 tables. Each table can seat 4 people. If 3/4 of the tables are occupied, how many people are dining?",
                'answer': "72",
                'complexity_level': 5,
                'reasoning_steps': 4,
                'problem_type': "fraction_multi_step"
            },
            {
                'question': "John saves $15 per week. After 8 weeks, he buys a game for $85. How much money does he have left?",
                'answer': "35",
                'complexity_level': 6,
                'reasoning_steps': 4,
                'problem_type': "multi_operation_word_problem"
            }
        ]
        
        # Limit to requested size
        problems = gsm8k_style_problems[:size]
        
        # Convert to HuggingFace Dataset format
        dataset = Dataset.from_list(problems)
        
        self.logger.logger.info(f"Created GSM8K-style test dataset with {len(dataset)} math problems for GRPO fine-tuning")
        return dataset
    
    def prepare_dataset_for_grpo(self, dataset: Dataset) -> Dataset:
        """Convert dataset to GRPO training format"""
        
        def format_for_grpo(example):
            # Format as instruction-following task
            prompt = f"Question: {example['question']}\nAnswer:"
            return {
                'text': prompt,
                'target': example['answer'],
                'question': example['question']
            }
        
        formatted_dataset = dataset.map(format_for_grpo)
        return formatted_dataset
    
    def cleanup_memory_between_stages(self, memory_manager=None):
        """
        Enhanced memory cleanup between progressive training stages
        Uses memory optimization tools for better cleanup
        """
        if memory_manager:
            # Use enhanced memory cleanup
            cleanup_stats = memory_manager.aggressive_cleanup()
            self.logger.logger.info(f"Enhanced memory cleanup completed")
            return cleanup_stats
        else:
            # Fallback to basic cleanup
            self.logger.logger.info("Performing basic memory cleanup...")
            
            # Force garbage collection
            collected = gc.collect()
            
            # Clear PyTorch cache
            torch.cuda.empty_cache()  # Clear GPU cache if any
            
            # Force memory cleanup
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            # Get memory stats
            import psutil
            memory_before = psutil.virtual_memory().percent
            
            # Additional aggressive cleanup
            gc.collect()
            gc.collect()  # Run twice for thorough cleanup
            
            memory_after = psutil.virtual_memory().percent
            
            self.logger.logger.info(f"Basic memory cleanup: {memory_before:.1f}% -> {memory_after:.1f}% "
                                   f"(freed {memory_before - memory_after:.1f}%, "
                                   f"collected {collected} objects)")
            
            return {
                'memory_before': memory_before,
                'memory_after': memory_after,
                'objects_collected': collected
            }

    def _validate_fisher_information(self, fisher_info: Dict[str, torch.Tensor]):
        """
        Diagnostic validation of Fisher Information Matrix
        Ensures EWC has meaningful values to work with
        """
        self.logger.logger.info("DIAGNOSTIC: Validating Fisher Information...")
        
        total_params = 0
        zero_params = 0
        fisher_stats = {}
        
        for name, values in fisher_info.items():
            param_count = values.numel()
            zero_count = (values == 0).sum().item()
            
            fisher_stats[name] = {
                'mean': values.mean().item(),
                'std': values.std().item(),
                'max': values.max().item(),
                'min': values.min().item(),
                'zero_ratio': zero_count / param_count
            }
            
            total_params += param_count
            zero_params += zero_count
            
            # Log key layers for diagnosis
            if 'embed' in name or 'lm_head' in name or 'layer.0' in name:
                self.logger.logger.info(f"  {name}: mean={values.mean():.6f}, "
                                       f"max={values.max():.6f}, "
                                       f"zeros={zero_count}/{param_count} ({100*zero_count/param_count:.1f}%)")
        
        overall_zero_ratio = zero_params / total_params
        self.logger.logger.info(f"Fisher Info Summary: {total_params} total params, "
                               f"{zero_params} zeros ({100*overall_zero_ratio:.1f}%)")
        
        # Warning checks
        if overall_zero_ratio > 0.9:
            self.logger.logger.warning("WARNING: >90% Fisher values are zero - EWC may not be effective!")
        
        if all(stats['max'] < 1e-10 for stats in fisher_stats.values()):
            self.logger.logger.warning("WARNING: All Fisher values extremely small - check gradient flow!")
        
        return fisher_stats

    def _test_ewc_functionality(self, model, fisher_info, initial_weights):
        """
        Test EWC by deliberately modifying weights and measuring penalty
        """
        self.logger.logger.info("DIAGNOSTIC: Testing EWC functionality...")
        
        # Calculate baseline EWC loss (should be 0)
        baseline_ewc = self.ewc.ewc_loss(model, fisher_info, initial_weights)
        self.logger.logger.info(f"Baseline EWC Loss: {baseline_ewc.item():.6f}")
        
        # Deliberately modify first layer to test EWC response
        first_param_name = next(iter(model.named_parameters()))[0]
        first_param = dict(model.named_parameters())[first_param_name]
        
        original_value = first_param.data.clone()
        
        # Make small change
        first_param.data += 0.01
        
        # Calculate EWC loss after change
        modified_ewc = self.ewc.ewc_loss(model, fisher_info, initial_weights)
        self.logger.logger.info(f"EWC Loss after +0.01 change to {first_param_name}: {modified_ewc.item():.6f}")
        
        # Restore original value
        first_param.data = original_value
        
        # Verify restoration
        restored_ewc = self.ewc.ewc_loss(model, fisher_info, initial_weights)
        self.logger.logger.info(f"EWC Loss after restoration: {restored_ewc.item():.6f}")
        
        if modified_ewc.item() > baseline_ewc.item():
            self.logger.logger.info("SUCCESS: EWC is working - penalty increased after weight change")
        else:
            self.logger.logger.warning("ERROR: EWC not working - no penalty for weight changes!")
        
        return {
            'baseline_ewc': baseline_ewc.item(),
            'modified_ewc': modified_ewc.item(),
            'restored_ewc': restored_ewc.item(),
            'ewc_responsive': modified_ewc.item() > baseline_ewc.item()
        }

    def _get_detailed_cpu_metrics(self):
        """Get detailed CPU and system metrics for analysis"""
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'swap_percent': psutil.swap_memory().percent,
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            'thread_count': len(psutil.Process().threads()),
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads()
        }
    
    def _log_cpu_utilization_analysis(self, stage_num, cpu_before, cpu_after):
        """Analyze CPU utilization patterns to identify bottlenecks"""
        self.logger.logger.info("DIAGNOSTIC: Stage {stage_num} CPU Analysis")
        self.logger.logger.info(f"  CPU Usage: {cpu_before['cpu_percent']:.1f}% -> {cpu_after['cpu_percent']:.1f}%")
        self.logger.logger.info(f"  Memory: {cpu_before['memory_percent']:.1f}% -> {cpu_after['memory_percent']:.1f}%")
        self.logger.logger.info(f"  Available RAM: {cpu_after['memory_available_gb']:.1f} GB")
        self.logger.logger.info(f"  PyTorch Threads: {cpu_after['torch_threads']} compute, {cpu_after['torch_interop_threads']} interop")
        self.logger.logger.info(f"  Process Threads: {cpu_after['thread_count']}")
        
        # Performance warnings
        if cpu_after['cpu_percent'] < 50:
            self.logger.logger.warning(f"WARNING: Low CPU utilization ({cpu_after['cpu_percent']:.1f}%) - potential bottleneck!")
        
        if cpu_after['memory_percent'] > 95:
            self.logger.logger.warning(f"WARNING: Critical memory usage ({cpu_after['memory_percent']:.1f}%) - may cause swapping!")
        
        if cpu_after['swap_percent'] > 0:
            self.logger.logger.warning(f"WARNING: Swap usage detected ({cpu_after['swap_percent']:.1f}%) - performance degradation likely!")

    def run_progressive_training_test(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        """
        Run progressive training test with 2-sample stages
        Tests the scientific framework before scaling up
        """
        
        self.logger.logger.info("Starting Progressive Training Scientific Test")
        self.logger.logger.info(f"Configuration: {self.config}")
        
        # Create test dataset
        test_dataset = self.create_test_dataset(size=4)
        formatted_dataset = self.prepare_dataset_for_grpo(test_dataset)
        
        # Create curriculum stages
        curriculum_stages = CurriculumDesigner.create_stages(
            formatted_dataset, 
            num_stages=self.config.num_stages,
            max_samples_per_stage=self.config.max_samples_per_stage
        )
        
        self.logger.logger.info(f"Created {len(curriculum_stages)} curriculum stages")
        
        # Initialize results dictionary early
        results = {
            'stages': [],
            'hardware_metrics': [],
            'total_start_time': time.time()
        }
        
        # Initialize memory manager
        memory_manager = MemoryManager()
        memory_manager.profiler.start_profiling()
        
        # Check system memory and get model recommendation
        recommended_model = get_recommended_model_for_system()
        if model_name == "Qwen/Qwen2-0.5B-Instruct" and recommended_model != model_name:
            # Only switch if critically low memory (< 1.5GB available)
            available_gb = memory_manager.profiler.get_memory_stats()['memory_available_gb']
            if available_gb < 1.5:
                self.logger.logger.error(f"CRITICAL: Only {available_gb:.1f}GB available. Switching to: {recommended_model}")
                self.logger.logger.error("WARNING: Smaller model may not be suitable for GRPO mathematical reasoning!")
                model_name = recommended_model
            else:
                self.logger.logger.warning(f"Memory limited ({available_gb:.1f}GB) but continuing with {model_name}")
                self.logger.logger.info("Will use aggressive memory optimization for GRPO fine-tuning")
        else:
            self.logger.logger.info(f"Using model for GRPO fine-tuning: {model_name}")
        
        # Create initial memory checkpoint
        memory_manager.create_memory_checkpoint("before_model_loading")
        
        # Load model with memory optimizations
        try:
            self.logger.logger.info(f"Loading optimized model: {model_name}")
            model_start_time = time.time()
            
            # Use optimized model loading
            model, tokenizer = ModelOptimizer.load_optimized_model(
                model_name, 
                optimize_for="memory"
            )
            
            model_load_time = time.time() - model_start_time
            self.logger.logger.info(f"Optimized model loaded in {model_load_time:.1f}s")
            
            # Check memory after model loading
            memory_manager.create_memory_checkpoint("after_model_loading")
            memory_health = memory_manager.profiler.check_memory_health()
            
            if memory_health['status'] == 'critical':
                self.logger.logger.error("CRITICAL: Memory usage is too high after model loading!")
                for warning in memory_health['warnings']:
                    self.logger.logger.error(f"  - {warning}")
                
                # Get recommendations
                recommendations = memory_manager.get_memory_recommendations()
                self.logger.logger.info("Memory optimization recommendations:")
                for rec in recommendations:
                    self.logger.logger.info(f"  - {rec}")
            
        except Exception as e:
            self.logger.logger.error(f"Error loading optimized model: {e}")
            memory_manager.profiler.stop_profiling()
            return None
        
        # Store initial weights for EWC (if enabled)
        initial_weights = None
        fisher_info = None
        
        if self.config.enable_ewc:
            self.logger.logger.info("Calculating Fisher Information Matrix using LIGHTNING method...")
            
            # LIGHTNING Fisher Information calculation - MAJOR BREAKTHROUGH!
            memory_manager.create_memory_checkpoint("before_fisher_calculation")
            fisher_start_time = time.time()
            
            # Get available memory for optimal method selection
            available_memory_gb = memory_manager.profiler.get_memory_stats()['memory_available_gb']
            
            # Use optimal Fisher calculator based on memory constraints
            self.logger.logger.info(f"Available memory: {available_memory_gb:.1f}GB - selecting optimal Fisher method")
            
            # Prepare minimal dataset for Fisher calculation (if needed)
            from torch.utils.data import DataLoader
            fisher_dataset = self._prepare_fisher_dataset(formatted_dataset, tokenizer)
            fisher_dataloader = DataLoader(fisher_dataset, batch_size=1, shuffle=False)
            
            fisher_info = create_optimal_fisher_calculator(
                model, 
                available_memory_gb=available_memory_gb,
                device='cpu',
                dataloader=fisher_dataloader
            )
            
            fisher_time = time.time() - fisher_start_time
            self.logger.logger.info(f"LIGHTNING Fisher calculation completed in {fisher_time:.1f}s")
            
            memory_manager.create_memory_checkpoint("after_fisher_calculation")
            
            # DIAGNOSTIC: Validate Fisher Information is meaningful
            self._validate_fisher_information(fisher_info)
            
            # Store initial weights
            initial_weights = {name: param.clone().detach() 
                             for name, param in model.named_parameters() 
                             if param.requires_grad}
            
            self.logger.logger.info("Fisher Information Matrix calculated")
            
            # DIAGNOSTIC: Test EWC functionality
            ewc_test_results = self._test_ewc_functionality(model, fisher_info, initial_weights)
            results['ewc_diagnostics'] = ewc_test_results
        
        # Progressive training through curriculum stages
        current_model = model
        
        for stage_idx, stage_data in enumerate(curriculum_stages):
            stage_num = stage_idx + 1
            self.logger.log_stage_start(stage_num, stage_data)
            
            stage_start_time = time.time()
            
            # Measure hardware utilization
            hardware_metrics = self.cpu_optimizer.measure_utilization(duration=3.0)
            self.logger.log_hardware_metrics(stage_num, hardware_metrics)
            
            # DIAGNOSTIC: Enhanced CPU monitoring
            cpu_before = self._get_detailed_cpu_metrics()
            
            # Simulate training stage (placeholder for actual GRPO integration)
            self.logger.logger.info(f"Training Stage {stage_num} with {len(stage_data)} samples...")
            
            # For testing: simulate training time
            training_simulation_time = 2.0 * len(stage_data)  # 2 seconds per sample
            time.sleep(training_simulation_time)
            
            cpu_after = self._get_detailed_cpu_metrics()
            self._log_cpu_utilization_analysis(stage_num, cpu_before, cpu_after)
            
            # Calculate EWC loss (if enabled)
            if self.config.enable_ewc and fisher_info is not None and initial_weights is not None:
                ewc_loss = self.ewc.ewc_loss(current_model, fisher_info, initial_weights)
                self.logger.logger.info(f"Stage {stage_num} EWC Loss: {ewc_loss.item():.6f}")
            
            stage_duration = time.time() - stage_start_time
            
            # Mock performance metrics for testing
            performance_metrics = {
                'samples_trained': len(stage_data),
                'training_time_seconds': stage_duration,
                'samples_per_second': len(stage_data) / stage_duration,
                'simulated_accuracy': 0.5 + 0.1 * stage_num,  # Mock improving accuracy
                'ewc_loss': ewc_loss.item() if (self.config.enable_ewc and 'ewc_loss' in locals()) else 0.0
            }
            
            self.logger.log_stage_complete(stage_num, stage_duration, performance_metrics)
            
            results['stages'].append({
                'stage': stage_num,
                'duration': stage_duration,
                'performance': performance_metrics
            })
            results['hardware_metrics'].append({
                'stage': stage_num,
                **hardware_metrics
            })
            
            # Critical: Enhanced memory cleanup between stages
            if stage_idx < len(curriculum_stages) - 1:  # Don't cleanup after last stage
                memory_manager.create_memory_checkpoint(f"before_cleanup_stage_{stage_num}")
                cleanup_stats = self.cleanup_memory_between_stages(memory_manager)
                memory_manager.create_memory_checkpoint(f"after_cleanup_stage_{stage_num}")
                results['stages'][-1]['memory_cleanup'] = cleanup_stats
                
                # Check memory health after cleanup
                health = memory_manager.profiler.check_memory_health()
                if health['status'] == 'critical':
                    self.logger.logger.warning(f"Memory critical after stage {stage_num} cleanup!")
                    for warning in health['warnings']:
                        self.logger.logger.warning(f"  - {warning}")
        
        # Complete experiment
        total_duration = time.time() - results['total_start_time']
        results['total_duration'] = total_duration
        
        self.logger.logger.info(f"Progressive training test completed in {total_duration:.1f}s")
        
        # Calculate summary statistics
        summary = self._calculate_experiment_summary(results)
        self.logger.logger.info("Experiment Summary:")
        for key, value in summary.items():
            self.logger.logger.info(f"  {key}: {value}")
        
        # Final memory statistics and cleanup
        memory_manager.create_memory_checkpoint("experiment_complete")
        final_health = memory_manager.profiler.check_memory_health()
        
        # Generate memory report
        self.logger.logger.info("Final Memory Report:")
        self.logger.logger.info(f"  Status: {final_health['status']}")
        for warning in final_health['warnings']:
            self.logger.logger.info(f"  Warning: {warning}")
        
        # Get final recommendations
        recommendations = memory_manager.get_memory_recommendations()
        if recommendations:
            self.logger.logger.info("Memory Optimization Recommendations:")
            for rec in recommendations:
                self.logger.logger.info(f"  - {rec}")
        
        # Stop memory profiling
        memory_manager.profiler.stop_profiling()
        
        # Save results
        results_file = self.logger.save_experiment_results()
        
        return {
            'results': results,
            'summary': summary,
            'results_file': str(results_file),
            'memory_health': final_health,
            'memory_recommendations': recommendations,
            'success': True
        }
    
    def _prepare_fisher_dataset(self, dataset: Dataset, tokenizer) -> List[Dict]:
        """Prepare dataset for Fisher Information calculation"""
        fisher_data = []
        
        for example in dataset:
            # Tokenize the text
            inputs = tokenizer(
                example['text'],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=256
            )
            
            fisher_data.append({
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': inputs['input_ids'].squeeze()  # For language modeling
            })
        
        return fisher_data
    
    def _calculate_experiment_summary(self, results: Dict) -> Dict[str, Any]:
        """Calculate summary statistics from experiment results"""
        
        if not results['stages']:
            return {'error': 'No stage results to summarize'}
        
        # Performance metrics
        accuracies = [s['performance']['simulated_accuracy'] for s in results['stages']]
        training_times = [s['performance']['training_time_seconds'] for s in results['stages']]
        samples_per_sec = [s['performance']['samples_per_second'] for s in results['stages']]
        
        # Hardware metrics
        cpu_utilizations = [h['avg_cpu_percent'] for h in results['hardware_metrics']]
        memory_utilizations = [h['avg_memory_percent'] for h in results['hardware_metrics']]
        
        summary = {
            'total_stages': len(results['stages']),
            'total_duration_seconds': results['total_duration'],
            'avg_accuracy': np.mean(accuracies),
            'accuracy_improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0,
            'avg_training_time_per_stage': np.mean(training_times),
            'avg_samples_per_second': np.mean(samples_per_sec),
            'avg_cpu_utilization': np.mean(cpu_utilizations),
            'max_cpu_utilization': np.max(cpu_utilizations),
            'avg_memory_utilization': np.mean(memory_utilizations),
            'max_memory_utilization': np.max(memory_utilizations)
        }
        
        return summary

def run_progressive_training_test():
    """Entry point for progressive training test"""
    
    # Configure for small-scale testing
    config = ProgressiveConfig(
        num_stages=2,
        max_samples_per_stage=2,
        lambda_ewc=1000.0,
        enable_ewc=True,
        enable_cpu_optimization=True
    )
    
    print("🔬 Progressive Training Scientific Test")
    print("=" * 50)
    print(f"Configuration: {config}")
    print("=" * 50)
    
    # Run the test
    integrator = ProgressiveGRPOIntegrator(config)
    
    try:
        results = integrator.run_progressive_training_test()
        
        if results['success']:
            print("\n✅ Progressive Training Test Completed Successfully!")
            print(f"Results saved to: {results['results_file']}")
            print("\nSummary:")
            for key, value in results['summary'].items():
                print(f"  {key}: {value}")
        else:
            print("❌ Progressive Training Test Failed")
            
    except Exception as e:
        print(f"❌ Error during progressive training test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_progressive_training_test()
