#!/usr/bin/env python3
"""
Progressive Training Core Module
Scientific implementation without modifying existing working code
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datasets import Dataset
from dataclasses import dataclass
import logging
import json
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

@dataclass
class ProgressiveConfig:
    """Configuration for progressive training experiments"""
    num_stages: int = 2
    lambda_ewc: float = 1000.0
    learning_rate: float = 2e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 2
    max_samples_per_stage: int = 2  # Start small for testing
    enable_ewc: bool = True
    enable_cpu_optimization: bool = True
    log_level: str = "INFO"

class CPUOptimizer:
    """Hardware acceleration utilities"""
    
    @staticmethod
    def configure_cpu_threads():
        """Optimize CPU thread configuration"""
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        
        # Reserve 2 cores for system, use rest for computation
        compute_cores = max(1, logical_cores - 2)
        interop_cores = max(1, physical_cores // 4)
        
        torch.set_num_threads(compute_cores)
        torch.set_num_interop_threads(interop_cores)
        
        # Set environment variables
        os.environ['OMP_NUM_THREADS'] = str(compute_cores)
        os.environ['MKL_NUM_THREADS'] = str(compute_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(compute_cores)
        
        return {
            'logical_cores': logical_cores,
            'physical_cores': physical_cores,
            'compute_cores': compute_cores,
            'interop_cores': interop_cores
        }
    
    @staticmethod
    def measure_utilization(duration: float = 5.0) -> Dict[str, float]:
        """Measure CPU and memory utilization"""
        cpu_readings = []
        memory_readings = []
        
        start_time = time.time()
        while time.time() - start_time < duration:
            cpu_readings.append(psutil.cpu_percent(interval=0.1))
            memory_readings.append(psutil.virtual_memory().percent)
        
        return {
            'avg_cpu_percent': np.mean(cpu_readings),
            'max_cpu_percent': np.max(cpu_readings),
            'avg_memory_percent': np.mean(memory_readings),
            'max_memory_percent': np.max(memory_readings),
            'measurement_duration': duration
        }

class FisherInformationCalculator:
    """Calculate Fisher Information Matrix for EWC"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
    def calculate_fisher_information(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Calculate Fisher Information Matrix for each parameter
        
        Theory: Fisher Information quantifies parameter importance
        Higher values = more critical for maintaining performance
        """
        fisher_info = {}
        
        # Initialize Fisher Information to zeros
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param, device=self.device)
        
        self.model.eval()
        total_samples = 0
        
        print(f"Calculating Fisher Information from {len(dataloader)} batches...")
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
            
            # Clear gradients
            self.model.zero_grad()
            
            try:
                # Forward pass
                outputs = self.model(**batch)
                
                # Handle different output types
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    # Create a loss from logits if no loss available
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    loss = torch.mean(logits)
                
                # Backward pass to get gradients
                loss.backward()
                
                # Accumulate squared gradients (Fisher Information approximation)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher_info[name] += param.grad.data ** 2
                
                total_samples += 1
                
            except Exception as e:
                print(f"Warning: Error processing batch {batch_idx}: {e}")
                continue
        
        # Normalize by number of samples
        if total_samples > 0:
            for name in fisher_info:
                fisher_info[name] /= total_samples
        
        print(f"Fisher Information calculated from {total_samples} samples")
        return fisher_info

class ElasticWeightConsolidation:
    """Prevent catastrophic forgetting using EWC regularization"""
    
    def __init__(self, lambda_ewc: float = 1000.0):
        self.lambda_ewc = lambda_ewc
        
    def ewc_loss(self, model: nn.Module, 
                 fisher_info: Dict[str, torch.Tensor],
                 optimal_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate EWC regularization loss
        
        Theory: Penalize changes to important parameters
        Loss = λ * Σ F_i * (θ_i - θ*_i)²
        """
        ewc_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if (name in fisher_info and 
                name in optimal_weights and 
                param.requires_grad):
                
                # Fisher-weighted L2 penalty on parameter changes
                importance = fisher_info[name]
                weight_change = (param - optimal_weights[name]) ** 2
                ewc_loss += (importance * weight_change).sum()
        
        return self.lambda_ewc * ewc_loss

class CurriculumDesigner:
    """Design curriculum stages based on problem complexity"""
    
    @staticmethod
    def calculate_complexity(text: str) -> float:
        """Calculate problem complexity score"""
        # Token-based complexity
        tokens = text.split()
        token_complexity = len(tokens)
        
        # Arithmetic operation complexity
        arithmetic_ops = (text.count('+') + text.count('-') + 
                         text.count('*') + text.count('/') + 
                         text.count('%'))
        
        # Number complexity (count of numbers)
        import re
        numbers = re.findall(r'\b\d+\b', text)
        number_complexity = len(numbers)
        
        # Nested reasoning (parentheses depth)
        max_depth = 0
        current_depth = 0
        for char in text:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        # Combined complexity score
        complexity = (token_complexity + 
                     2 * arithmetic_ops + 
                     1.5 * number_complexity + 
                     3 * max_depth)
        
        return complexity
    
    @staticmethod
    def create_stages(dataset: Dataset, num_stages: int = 2, 
                     max_samples_per_stage: int = 2) -> List[Dataset]:
        """Create curriculum stages with limited samples for testing"""
        
        # Calculate complexities
        complexities = []
        for example in dataset:
            question = example.get('question', example.get('text', ''))
            complexity = CurriculumDesigner.calculate_complexity(question)
            complexities.append(complexity)
        
        # Add complexity scores to dataset
        dataset_with_complexity = dataset.add_column('complexity', complexities)
        
        # Sort by complexity
        sorted_dataset = dataset_with_complexity.sort('complexity')
        
        # Create stages with limited samples
        stage_datasets = []
        total_samples = min(len(dataset), num_stages * max_samples_per_stage)
        samples_per_stage = max_samples_per_stage
        
        for stage in range(num_stages):
            start_idx = stage * samples_per_stage
            end_idx = min(start_idx + samples_per_stage, total_samples)
            
            if start_idx >= total_samples:
                break
                
            stage_data = sorted_dataset.select(range(start_idx, end_idx))
            stage_datasets.append(stage_data)
            
            avg_complexity = np.mean([complexities[i] for i in range(start_idx, end_idx)])
            print(f"Stage {stage + 1}: {len(stage_data)} samples, "
                  f"avg complexity: {avg_complexity:.1f}")
        
        return stage_datasets

class ProgressiveTrainingLogger:
    """Scientific logging for progressive training experiments"""
    
    def __init__(self, experiment_name: str, log_dir: str = "experiments/logs"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / f"{experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.experiment_data = {
            'experiment_name': experiment_name,
            'start_time': time.time(),
            'stages': [],
            'hardware_metrics': [],
            'configuration': {}
        }
    
    def log_stage_start(self, stage_num: int, stage_data: Dataset):
        """Log the start of a training stage"""
        self.logger.info(f"Starting Stage {stage_num}")
        self.logger.info(f"Stage {stage_num} dataset size: {len(stage_data)}")
        
        stage_info = {
            'stage': stage_num,
            'start_time': time.time(),
            'dataset_size': len(stage_data)
        }
        self.experiment_data['stages'].append(stage_info)
    
    def log_stage_complete(self, stage_num: int, duration: float, 
                          performance_metrics: Dict[str, Any]):
        """Log stage completion with metrics"""
        self.logger.info(f"Stage {stage_num} completed in {duration:.1f}s")
        for metric, value in performance_metrics.items():
            self.logger.info(f"Stage {stage_num} {metric}: {value}")
        
        # Update stage info
        for stage_info in self.experiment_data['stages']:
            if stage_info['stage'] == stage_num:
                stage_info['duration'] = duration
                stage_info['performance'] = performance_metrics
                break
    
    def log_hardware_metrics(self, stage_num: int, metrics: Dict[str, float]):
        """Log hardware utilization metrics"""
        self.logger.info(f"Stage {stage_num} CPU utilization: {metrics['avg_cpu_percent']:.1f}%")
        self.logger.info(f"Stage {stage_num} Memory usage: {metrics['avg_memory_percent']:.1f}%")
        
        hardware_entry = {
            'stage': stage_num,
            'timestamp': time.time(),
            **metrics
        }
        self.experiment_data['hardware_metrics'].append(hardware_entry)
    
    def save_experiment_results(self):
        """Save comprehensive experiment results"""
        self.experiment_data['end_time'] = time.time()
        self.experiment_data['total_duration'] = (
            self.experiment_data['end_time'] - self.experiment_data['start_time']
        )
        
        results_file = self.log_dir / f"{self.experiment_name}_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.experiment_data, f, indent=2, default=str)
        
        self.logger.info(f"Experiment results saved to {results_file}")
        return results_file
