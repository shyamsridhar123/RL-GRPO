"""
Fast Fisher Information Calculator for CPU Training
Optimized implementation for memory-constrained environments
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import time
import gc
import numpy as np
from collections import defaultdict

class FastFisherCalculator:
    """
    Optimized Fisher Information calculation for CPU training
    
    Key optimizations:
    1. Diagonal approximation (ignore off-diagonal terms)
    2. Subsampling strategy (don't use all data)
    3. Layer-wise processing (reduce memory)
    4. Gradient accumulation with periodic cleanup
    """
    
    def __init__(self, model: nn.Module, device='cpu'):
        self.model = model
        self.device = device
        self.layer_groups = self._group_parameters()
        
    def _group_parameters(self) -> Dict[str, List[str]]:
        """Group parameters by layer type for efficient processing"""
        groups = {
            'embeddings': [],
            'attention': [],
            'feedforward': [],
            'output': [],
            'other': []
        }
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'embed' in name.lower():
                groups['embeddings'].append(name)
            elif any(attn in name.lower() for attn in ['attn', 'attention', 'self_attn']):
                groups['attention'].append(name)
            elif any(ff in name.lower() for ff in ['mlp', 'ffn', 'feed_forward']):
                groups['feedforward'].append(name)
            elif any(out in name.lower() for out in ['lm_head', 'output', 'classifier']):
                groups['output'].append(name)
            else:
                groups['other'].append(name)
        
        return groups
    
    def calculate_fisher_diagonal_fast(self, 
                                     dataloader: DataLoader,
                                     max_samples: int = 50,
                                     subsample_ratio: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Fast diagonal Fisher Information approximation
        
        Args:
            dataloader: Training data
            max_samples: Maximum samples to use (computational budget)
            subsample_ratio: Fraction of parameters to track (memory budget)
        """
        print(f"🚀 FAST Fisher Information calculation starting...")
        print(f"   Max samples: {max_samples}, Subsample ratio: {subsample_ratio:.1%}")
        
        start_time = time.time()
        
        # Initialize Fisher Information storage
        fisher_info = {}
        param_importance = self._calculate_parameter_importance()
        
        # Select subset of parameters to track
        selected_params = self._select_important_parameters(param_importance, subsample_ratio)
        
        for name in selected_params:
            param = dict(self.model.named_parameters())[name]
            fisher_info[name] = torch.zeros_like(param, device=self.device)
        
        print(f"   Tracking {len(selected_params)}/{len(list(self.model.parameters()))} parameters")
        
        # Process samples in batches
        self.model.eval()
        samples_processed = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if samples_processed >= max_samples:
                break
                
            # Memory management
            if batch_idx % 10 == 0 and batch_idx > 0:
                gc.collect()
            
            try:
                # Process batch efficiently
                fisher_batch = self._process_batch_fast(batch, selected_params)
                
                # Accumulate Fisher Information
                for name, grad_sq in fisher_batch.items():
                    fisher_info[name] += grad_sq
                
                samples_processed += 1
                
                if samples_processed % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"   Processed {samples_processed}/{max_samples} samples ({elapsed:.1f}s)")
                
            except Exception as e:
                print(f"   Warning: Skipping batch {batch_idx}: {e}")
                continue
        
        # Normalize by number of samples
        if samples_processed > 0:
            for name in fisher_info:
                fisher_info[name] /= samples_processed
        
        # Fill in untracked parameters with small values
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in fisher_info:
                fisher_info[name] = torch.full_like(param, 1e-8, device=self.device)
        
        elapsed = time.time() - start_time
        print(f"✅ Fast Fisher Information completed in {elapsed:.1f}s ({samples_processed} samples)")
        
        return fisher_info
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Estimate parameter importance for subsampling strategy"""
        importance = {}
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Base importance on parameter properties
            base_score = 1.0
            
            # Layer type importance
            if 'embed' in name.lower():
                base_score *= 2.0  # Embeddings are very important
            elif 'attention' in name.lower() or 'attn' in name.lower():
                base_score *= 1.5  # Attention weights important
            elif 'lm_head' in name.lower() or 'output' in name.lower():
                base_score *= 1.8  # Output layer important
            elif 'bias' in name.lower():
                base_score *= 0.5  # Biases less critical
            
            # Size penalty (don't over-sample huge matrices)
            param_size = param.numel()
            if param_size > 1e6:  # Very large parameters
                base_score *= 0.7
            elif param_size < 1000:  # Small parameters
                base_score *= 1.2
            
            importance[name] = base_score
        
        return importance
    
    def _select_important_parameters(self, 
                                   importance: Dict[str, float], 
                                   subsample_ratio: float) -> List[str]:
        """Select most important parameters to track"""
        # Sort by importance
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Select top fraction
        num_to_select = max(1, int(len(sorted_params) * subsample_ratio))
        selected = [name for name, score in sorted_params[:num_to_select]]
        
        # Always include critical layers
        critical_keywords = ['embed_tokens', 'lm_head', 'attention.0', 'layer.0']
        for name, param in self.model.named_parameters():
            if any(keyword in name for keyword in critical_keywords):
                if name not in selected:
                    selected.append(name)
        
        return selected
    
    def _process_batch_fast(self, batch: Dict, selected_params: List[str]) -> Dict[str, torch.Tensor]:
        """Process single batch with minimal memory footprint"""
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
        
        # Clear gradients
        self.model.zero_grad()
        
        # Forward pass
        outputs = self.model(**batch)
        
        # Calculate loss
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        else:
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            # Use cross-entropy loss approximation
            if 'labels' in batch:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    batch['labels'].view(-1),
                    ignore_index=-100
                )
            else:
                loss = torch.mean(logits)
        
        # Backward pass
        loss.backward()
        
        # Extract gradients for selected parameters only
        fisher_batch = {}
        for name in selected_params:
            param = dict(self.model.named_parameters())[name]
            if param.grad is not None:
                # Store squared gradient (diagonal Fisher approximation)
                fisher_batch[name] = param.grad.data ** 2
            else:
                fisher_batch[name] = torch.zeros_like(param, device=self.device)
        
        return fisher_batch

class AdaptiveFisherCalculator(FastFisherCalculator):
    """
    Adaptive Fisher calculation that adjusts based on available memory
    """
    
    def __init__(self, model: nn.Module, device='cpu', memory_budget_gb: float = 1.0):
        super().__init__(model, device)
        self.memory_budget_gb = memory_budget_gb
        
    def calculate_adaptive_fisher(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Automatically adjust Fisher calculation based on memory constraints
        """
        import psutil
        
        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        print(f"🧠 Adaptive Fisher calculation:")
        print(f"   Available memory: {available_gb:.1f} GB")
        print(f"   Memory budget: {self.memory_budget_gb:.1f} GB")
        
        # Adjust parameters based on memory
        if available_gb < 2.0:
            # Very limited memory - aggressive subsampling
            max_samples = 20
            subsample_ratio = 0.05
            print("   Mode: ULTRA_FAST (very limited memory)")
        elif available_gb < 4.0:
            # Limited memory - moderate subsampling
            max_samples = 50
            subsample_ratio = 0.1
            print("   Mode: FAST (limited memory)")
        else:
            # Good memory - standard subsampling
            max_samples = 100
            subsample_ratio = 0.2
            print("   Mode: STANDARD (good memory)")
        
        return self.calculate_fisher_diagonal_fast(
            dataloader, 
            max_samples=max_samples,
            subsample_ratio=subsample_ratio
        )

def benchmark_fisher_methods(model, dataloader, num_runs=3):
    """Benchmark different Fisher calculation methods"""
    print("🏁 Benchmarking Fisher Information Methods")
    print("=" * 50)
    
    results = {}
    
    # Method 1: Original (slow but accurate)
    print("\n1. Testing Original Fisher Calculator...")
    from progressive_core import FisherInformationCalculator
    
    original_calc = FisherInformationCalculator(model)
    start_time = time.time()
    fisher_original = original_calc.calculate_fisher_information(dataloader)
    original_time = time.time() - start_time
    
    results['original'] = {
        'time': original_time,
        'memory_peak': None,  # Would need memory profiling
        'accuracy': 'baseline'
    }
    
    # Method 2: Fast approximation
    print("\n2. Testing Fast Fisher Calculator...")
    fast_calc = FastFisherCalculator(model)
    start_time = time.time()
    fisher_fast = fast_calc.calculate_fisher_diagonal_fast(dataloader, max_samples=50)
    fast_time = time.time() - start_time
    
    results['fast'] = {
        'time': fast_time,
        'speedup': original_time / fast_time,
        'accuracy': 'approximation'
    }
    
    # Method 3: Adaptive
    print("\n3. Testing Adaptive Fisher Calculator...")
    adaptive_calc = AdaptiveFisherCalculator(model, memory_budget_gb=1.0)
    start_time = time.time()
    fisher_adaptive = adaptive_calc.calculate_adaptive_fisher(dataloader)
    adaptive_time = time.time() - start_time
    
    results['adaptive'] = {
        'time': adaptive_time,
        'speedup': original_time / adaptive_time,
        'accuracy': 'memory_adaptive'
    }
    
    # Print results
    print("\n📊 BENCHMARK RESULTS:")
    print("-" * 50)
    for method, stats in results.items():
        print(f"{method.upper():12} | Time: {stats['time']:6.1f}s", end="")
        if 'speedup' in stats:
            print(f" | Speedup: {stats['speedup']:4.1f}x", end="")
        print(f" | Accuracy: {stats['accuracy']}")
    
    return results

if __name__ == "__main__":
    print("Fast Fisher Information Calculator")
    print("Optimized for CPU training with memory constraints")
