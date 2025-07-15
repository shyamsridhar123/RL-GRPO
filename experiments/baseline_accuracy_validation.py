"""
BASELINE Accuracy Validation Suite for Ultra-Optimized GRPO Training
Ensures our speed optimizations don't sacrifice GSM8K reasoning quality

Critical Mission: Maintain GSM8K accuracy while achieving speed gains
NOTE: This is a BASELINE experiment - we'll return after completing progressive training

TODO: Complete progressive training implementation first!
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
import time
import numpy as np
from typing import Dict, List, Tuple
import json
import gc
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.lightning_fisher import create_optimal_fisher_calculator
from src.training.advanced_memory_optimization import CPUMemoryOptimizer

class AccuracyValidationSuite:
    """
    Comprehensive validation of optimization impact on reasoning quality
    
    Tests:
    1. Reasoning benchmark comparison (before/after optimization)
    2. Fisher Information quality assessment
    3. EWC effectiveness validation
    4. Quantization impact analysis
    5. Training convergence comparison
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.baseline_model = None
        self.optimized_model = None
        
    def load_models(self):
        """Load baseline and optimized versions"""
        print("📊 Loading models for accuracy comparison...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load baseline model (no optimizations)
        print("   Loading baseline model...")
        self.baseline_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Full precision
            device_map="cpu"
        )
        
        # Load optimized model (with our optimizations)
        print("   Loading optimized model...")
        self.optimized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # Lower precision
            device_map="cpu"
        )
        
        # Apply memory optimizations to optimized model
        optimizer = CPUMemoryOptimizer()
        self.optimized_model = optimizer.optimize_model_for_cpu(self.optimized_model)
        
        print("✅ Models loaded successfully")
    
    def create_gsm8k_baseline_benchmarks(self) -> List[Dict]:
        """
        Create GSM8K-style math reasoning test cases 
        (Our established benchmark for GRPO training)
        
        These are simplified GSM8K problems to test:
        - Multi-step mathematical reasoning
        - Word problem solving
        - Arithmetic accuracy
        - Logical step progression
        """
        benchmarks = [
            {
                "category": "gsm8k_arithmetic",
                "prompt": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much does she make every day at the farmers' market?",
                "expected_steps": ["16 - 3 - 4 = 9", "9 × $2 = $18"],
                "correct_answer": "$18"
            },
            {
                "category": "gsm8k_multi_step",
                "prompt": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts are needed for 3 robes?",
                "expected_steps": ["2 blue + 1 white = 3 bolts per robe", "3 robes × 3 bolts = 9 bolts"],
                "correct_answer": "9 bolts"
            },
            {
                "category": "gsm8k_word_problem",
                "prompt": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
                "expected_steps": ["Cost: $80,000 + $50,000 = $130,000", "Value increase: 150% of $80,000 = $120,000", "New value: $80,000 + $120,000 = $200,000", "Profit: $200,000 - $130,000 = $70,000"],
                "correct_answer": "$70,000"
            },
            {
                "category": "gsm8k_percentage",
                "prompt": "There are 15 trees in the grove. Grove workers will plant trees today. After they are done there will be 21 trees. How many trees did they plant today?",
                "expected_steps": ["21 - 15 = 6"],
                "correct_answer": "6 trees"
            }
        ]
        
        return benchmarks
    
    def evaluate_gsm8k_reasoning_quality(self, model: nn.Module, benchmarks: List[Dict]) -> Dict:
        """
        Evaluate GSM8K reasoning quality on benchmark tasks
        
        Returns metrics:
        - Mathematical reasoning coherence
        - Step-by-step accuracy
        - Answer correctness (key for GSM8K)
        - Response quality
        """
        print("� Evaluating GSM8K reasoning quality...")
        
        results = {
            "coherence_scores": [],
            "step_accuracy": [],
            "answer_correctness": [],
            "response_lengths": []
        }
        
        model.eval()
        with torch.no_grad():
            for benchmark in benchmarks:
                # Generate response
                inputs = self.tokenizer(
                    benchmark["prompt"],
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.1,  # Low temperature for consistent reasoning
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                # Analyze response quality
                coherence = self._analyze_gsm8k_coherence(response, benchmark)
                step_acc = self._analyze_step_accuracy(response, benchmark)
                answer_correct = self._check_answer_correctness(response, benchmark)
                
                results["coherence_scores"].append(coherence)
                results["step_accuracy"].append(step_acc)
                results["answer_correctness"].append(answer_correct)
                results["response_lengths"].append(len(response))
        
        # Calculate summary metrics
        summary = {
            "avg_coherence": np.mean(results["coherence_scores"]),
            "avg_step_accuracy": np.mean(results["step_accuracy"]),
            "answer_accuracy": np.mean(results["answer_correctness"]),
            "avg_response_length": np.mean(results["response_lengths"])
        }
        
        return summary
    
    def _analyze_gsm8k_coherence(self, response: str, benchmark: Dict) -> float:
        """Analyze mathematical reasoning coherence for GSM8K problems"""
        coherence_score = 0.0
        
        # Check for structured mathematical reasoning
        if "step" in response.lower() or "first" in response.lower() or "then" in response.lower():
            coherence_score += 0.2
        
        # Check for mathematical operations (critical for GSM8K)
        math_ops = ["+", "-", "*", "/", "=", "×", "÷", "$", "%"]
        if any(op in response for op in math_ops):
            coherence_score += 0.4
        
        # Check for numerical calculations
        import re
        numbers = re.findall(r'\d+', response)
        if len(numbers) >= 2:  # At least 2 numbers (showing calculation)
            coherence_score += 0.2
        
        # Check for logical connectors
        if any(connector in response.lower() for connector in ["therefore", "so", "total", "remainder"]):
            coherence_score += 0.2
        
        return min(coherence_score, 1.0)
    
    def _analyze_step_accuracy(self, response: str, benchmark: Dict) -> float:
        """Check if response contains expected reasoning steps"""
        expected_steps = benchmark.get("expected_steps", [])
        if not expected_steps:
            return 0.5  # No specific steps to check
        
        steps_found = 0
        for step in expected_steps:
            # Flexible matching for key concepts
            step_concepts = step.lower().replace("*", "×").replace("/", "÷")
            if any(concept in response.lower() for concept in step_concepts.split()):
                steps_found += 1
        
        return steps_found / len(expected_steps)
    
    def _check_answer_correctness(self, response: str, benchmark: Dict) -> float:
        """Check if final answer is correct"""
        correct_answer = benchmark["correct_answer"].lower()
        response_lower = response.lower()
        
        # Extract key answer components
        answer_parts = correct_answer.split()
        matches = sum(1 for part in answer_parts if part in response_lower)
        
        return matches / len(answer_parts)
    
    def validate_fisher_quality(self) -> Dict:
        """
        Validate Fisher Information approximation quality
        
        Compares Lightning Fisher vs true Fisher (when computationally feasible)
        """
        print("⚡ Validating Fisher Information quality...")
        
        # Create a smaller test model for true Fisher comparison
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.linear1 = nn.Linear(32, 16)
                self.linear2 = nn.Linear(16, 10)
            
            def forward(self, input_ids, **kwargs):
                x = self.embed(input_ids)
                x = torch.mean(x, dim=1)
                x = torch.relu(self.linear1(x))
                return self.linear2(x)
        
        test_model = TestModel()
        
        # Generate test data
        test_data = [{"input_ids": torch.randint(0, 100, (1, 10))} for _ in range(5)]
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_data, batch_size=1)
        
        # Calculate Lightning Fisher
        lightning_fisher = create_optimal_fisher_calculator(
            test_model, available_memory_gb=2.0, dataloader=test_loader
        )
        
        # Calculate approximate "true" Fisher using gradients (for comparison)
        true_fisher = self._calculate_approximate_true_fisher(test_model, test_loader)
        
        # Compare Fisher approximations
        comparison = self._compare_fisher_matrices(lightning_fisher, true_fisher)
        
        return comparison
    
    def _calculate_approximate_true_fisher(self, model: nn.Module, dataloader: DataLoader) -> Dict:
        """Calculate Fisher using gradient-based method for comparison"""
        fisher_info = {}
        model.train()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param)
        
        for batch in dataloader:
            model.zero_grad()
            
            # Forward pass
            outputs = model(**batch)
            loss = torch.mean(outputs)  # Simple loss for Fisher calculation
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients (Fisher approximation)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
        
        # Normalize by number of samples
        num_samples = len(dataloader)
        for name in fisher_info:
            fisher_info[name] /= num_samples
        
        return fisher_info
    
    def _compare_fisher_matrices(self, lightning_fisher: Dict, true_fisher: Dict) -> Dict:
        """Compare Lightning Fisher vs true Fisher"""
        comparisons = {}
        
        for name in lightning_fisher:
            if name in true_fisher:
                lightning_vals = lightning_fisher[name].flatten()
                true_vals = true_fisher[name].flatten()
                
                # Calculate correlation
                correlation = torch.corrcoef(torch.stack([lightning_vals, true_vals]))[0, 1]
                
                # Calculate relative magnitude
                lightning_mean = torch.mean(lightning_vals)
                true_mean = torch.mean(true_vals)
                magnitude_ratio = lightning_mean / (true_mean + 1e-8)
                
                comparisons[name] = {
                    "correlation": correlation.item(),
                    "magnitude_ratio": magnitude_ratio.item()
                }
        
        return comparisons
    
    def run_baseline_gsm8k_validation(self) -> Dict:
        """
        Run BASELINE GSM8K accuracy validation 
        
        IMPORTANT: This is a baseline experiment!
        We need to complete progressive training first before full validation.
        
        Returns baseline report on optimization impact on GSM8K performance
        """
        print("🎯 BASELINE GSM8K ACCURACY VALIDATION")
        print("=" * 60)
        print("⚠️  NOTE: This is a BASELINE experiment")
        print("⚠️  TODO: Complete progressive training implementation first!")
        print("=" * 60)
        
        # Load models
        self.load_models()
        
        # Create GSM8K benchmarks
        benchmarks = self.create_gsm8k_baseline_benchmarks()
        
        # Evaluate baseline model
        print("📊 Evaluating baseline model on GSM8K problems...")
        baseline_results = self.evaluate_gsm8k_reasoning_quality(self.baseline_model, benchmarks)
        
        # Evaluate optimized model
        print("📊 Evaluating optimized model on GSM8K problems...")
        optimized_results = self.evaluate_gsm8k_reasoning_quality(self.optimized_model, benchmarks)
        
        # Validate Fisher quality
        fisher_validation = self.validate_fisher_quality()
        
        # Calculate accuracy impact
        accuracy_impact = self._calculate_accuracy_impact(baseline_results, optimized_results)
        
        # Compile baseline report
        baseline_report = {
            "baseline_performance": baseline_results,
            "optimized_performance": optimized_results,
            "accuracy_impact": accuracy_impact,
            "fisher_validation": fisher_validation,
            "recommendation": self._generate_baseline_recommendation(accuracy_impact),
            "next_steps": [
                "Complete progressive training implementation",
                "Run full GSM8K validation after progressive training",
                "Integrate accuracy validation into production pipeline"
            ]
        }
        
        self._print_baseline_validation_report(baseline_report)
        
        return baseline_report
    
    def _calculate_accuracy_impact(self, baseline: Dict, optimized: Dict) -> Dict:
        """Calculate the impact of optimizations on accuracy"""
        impact = {}
        
        for metric in baseline:
            baseline_val = baseline[metric]
            optimized_val = optimized[metric]
            
            # Calculate percentage change
            change = ((optimized_val - baseline_val) / baseline_val) * 100
            impact[metric] = {
                "baseline": baseline_val,
                "optimized": optimized_val,
                "change_percent": change,
                "acceptable": abs(change) < 10  # <10% degradation is acceptable
            }
        
        return impact
    
    def _generate_baseline_recommendation(self, accuracy_impact: Dict) -> str:
        """Generate baseline recommendation with next steps"""
        degradations = [impact["change_percent"] for impact in accuracy_impact.values() 
                      if impact["change_percent"] < 0]
        
        if not degradations:
            return "✅ BASELINE SAFE: No accuracy degradation detected. Ready for progressive training integration."
        
        max_degradation = abs(min(degradations))
        
        if max_degradation < 5:
            return "✅ BASELINE SAFE: Minor impact (<5%). Proceed with progressive training."
        elif max_degradation < 10:
            return "⚠️ BASELINE CAUTION: Moderate impact (5-10%). Monitor during progressive training."
        else:
            return "❌ BASELINE RISK: Significant degradation (>10%). Review optimizations before progressive training."
    
    def _print_baseline_validation_report(self, report: Dict):
        """Print baseline validation report with next steps"""
        print("\n" + "=" * 60)
        print("🎯 BASELINE GSM8K ACCURACY VALIDATION REPORT")
        print("=" * 60)
        
        print("\n📊 GSM8K PERFORMANCE COMPARISON:")
        for metric, data in report["accuracy_impact"].items():
            status = "✅" if data["acceptable"] else "❌"
            print(f"   {status} {metric}:")
            print(f"      Baseline: {data['baseline']:.3f}")
            print(f"      Optimized: {data['optimized']:.3f}")
            print(f"      Change: {data['change_percent']:+.1f}%")
        
        print(f"\n🎯 BASELINE RECOMMENDATION:")
        print(f"   {report['recommendation']}")
        
        print(f"\n⚡ FISHER INFORMATION QUALITY:")
        avg_correlation = np.mean([data["correlation"] for data in report["fisher_validation"].values()])
        print(f"   Average correlation with true Fisher: {avg_correlation:.3f}")
        
        print(f"\n📋 NEXT STEPS:")
        for i, step in enumerate(report["next_steps"], 1):
            print(f"   {i}. {step}")
        
        print(f"\n⚠️  IMPORTANT: This is a BASELINE validation.")
        print(f"   Complete progressive training before full production validation!")


if __name__ == "__main__":
    print("🎯 Starting BASELINE GSM8K Accuracy Validation")
    print("Testing optimization impact on GSM8K reasoning quality...")
    print("⚠️  NOTE: This is a baseline experiment!")
    print("⚠️  TODO: Complete progressive training first!")
    
    validator = AccuracyValidationSuite()
    report = validator.run_baseline_gsm8k_validation()
    
    print("\n🎉 Baseline validation completed!")
    print("Next: Complete progressive training implementation.")
    print("Then: Run full GSM8K validation in production pipeline.")
