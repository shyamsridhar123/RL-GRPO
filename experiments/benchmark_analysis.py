#!/usr/bin/env python3
"""
GRPO Model Benchmark Analysis
=============================

Comprehensive benchmark comparing all trained model variants against the base Qwen model
on GSM8K mathematical reasoning tasks.

Measurements:
- Inference speed (tokens/second, response time)
- Accuracy on GSM8K math problems
- Memory utilization
- CPU utilization during inference
- Response quality analysis

Author: GRPO CPU Implementation Project
Date: July 8, 2025
"""

import os
import sys
import json
import time
import psutil
import torch
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class BenchmarkAnalyzer:
    """Scientific benchmark analysis for GRPO model variants."""
    
    def __init__(self, output_dir: str = "experiments/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_file = self.output_dir / f"benchmark_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Model configurations
        self.models = {
            "base_qwen": {
                "path": "Qwen/Qwen2-0.5B-Instruct",
                "description": "Base Qwen2-0.5B-Instruct model"
            },
            "ultra_fast": {
                "path": "./models/ultra_fast/final_model",
                "description": "Ultra Fast GRPO trained model"
            },
            "extreme_fast": {
                "path": "./models/extreme_fast/final_model", 
                "description": "Extreme Fast GRPO trained model"
            },
            "hardware_accelerated": {
                "path": "./models/hardware_accelerated/final_model",
                "description": "Hardware Accelerated GRPO trained model"
            }
        }
        
        # Test configuration
        self.test_size = 50  # Number of GSM8K problems to test
        self.max_new_tokens = 256
        self.temperature = 0.1
        self.device = "cpu"
        
        self.results = {}
        
    def extract_number_from_answer(self, text: str) -> float:
        """Extract the final numerical answer from model response."""
        # Look for patterns like "The answer is X" or just numbers
        patterns = [
            r"(?:the answer is|answer:|final answer:?)\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*$",  # Number at end
            r"=\s*(\d+(?:\.\d+)?)",  # After equals sign
        ]
        
        text = text.lower().strip()
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Fallback: find last number in text
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                pass
                
        return None
    
    def evaluate_math_accuracy(self, model_answer: str, correct_answer: str) -> bool:
        """Evaluate if the model's answer matches the correct answer."""
        model_num = self.extract_number_from_answer(model_answer)
        
        # Clean up correct answer
        correct_num = self.extract_number_from_answer(correct_answer)
        
        if model_num is None or correct_num is None:
            return False
            
        # Allow for small floating point differences
        return abs(model_num - correct_num) < 0.01
    
    def measure_system_resources(self) -> Dict[str, float]:
        """Measure current system resource utilization."""
        process = psutil.Process()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_used_gb": process.memory_info().rss / (1024**3),
            "memory_percent": process.memory_percent(),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3)
        }
    
    def benchmark_model(self, model_name: str, model_config: Dict[str, str]) -> Dict[str, Any]:
        """Benchmark a single model on GSM8K tasks."""
        self.logger.info(f"Starting benchmark for {model_name}: {model_config['description']}")
        
        results = {
            "model_name": model_name,
            "model_path": model_config["path"],
            "description": model_config["description"],
            "test_problems": [],
            "summary_metrics": {},
            "system_metrics": {},
            "errors": []
        }
        
        try:
            # Load model and tokenizer
            self.logger.info(f"Loading model from {model_config['path']}")
            start_load_time = time.time()
            
            tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
            model = AutoModelForCausalLM.from_pretrained(
                model_config["path"],
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )
            
            load_time = time.time() - start_load_time
            self.logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Get initial system metrics
            initial_resources = self.measure_system_resources()
            
            # Load GSM8K dataset
            self.logger.info("Loading GSM8K dataset")
            dataset = load_dataset("gsm8k", "main", split="test")
            test_problems = dataset.select(range(min(self.test_size, len(dataset))))
            
            # Benchmark each problem
            correct_answers = 0
            total_inference_time = 0
            total_tokens_generated = 0
            
            for i, problem in enumerate(test_problems):
                self.logger.info(f"Processing problem {i+1}/{len(test_problems)}")
                
                # Prepare input
                question = problem["question"]
                correct_answer = problem["answer"]
                
                prompt = f"Solve this math problem step by step:\n\n{question}\n\nSolution:"
                
                # Measure inference
                start_inference = time.time()
                pre_memory = self.measure_system_resources()
                
                inputs = tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                inference_time = time.time() - start_inference
                post_memory = self.measure_system_resources()
                
                # Decode response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = response[len(prompt):].strip()
                
                # Calculate tokens generated
                tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
                tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
                
                # Evaluate accuracy
                is_correct = self.evaluate_math_accuracy(generated_text, correct_answer)
                if is_correct:
                    correct_answers += 1
                
                # Record problem result
                problem_result = {
                    "problem_id": i,
                    "question": question,
                    "correct_answer": correct_answer,
                    "model_response": generated_text,
                    "is_correct": is_correct,
                    "inference_time_seconds": inference_time,
                    "tokens_generated": tokens_generated,
                    "tokens_per_second": tokens_per_second,
                    "memory_before_mb": pre_memory["memory_used_gb"] * 1024,
                    "memory_after_mb": post_memory["memory_used_gb"] * 1024,
                    "cpu_percent": post_memory["cpu_percent"]
                }
                
                results["test_problems"].append(problem_result)
                total_inference_time += inference_time
                total_tokens_generated += tokens_generated
                
                # Log progress
                if (i + 1) % 10 == 0:
                    current_accuracy = correct_answers / (i + 1)
                    avg_time = total_inference_time / (i + 1)
                    self.logger.info(f"Progress: {i+1}/{len(test_problems)}, "
                                   f"Accuracy: {current_accuracy:.3f}, "
                                   f"Avg time: {avg_time:.2f}s")
            
            # Calculate summary metrics
            final_resources = self.measure_system_resources()
            
            results["summary_metrics"] = {
                "accuracy": correct_answers / len(test_problems),
                "total_problems": len(test_problems),
                "correct_answers": correct_answers,
                "avg_inference_time_seconds": total_inference_time / len(test_problems),
                "avg_tokens_per_second": total_tokens_generated / total_inference_time if total_inference_time > 0 else 0,
                "total_inference_time_seconds": total_inference_time,
                "total_tokens_generated": total_tokens_generated,
                "model_load_time_seconds": load_time
            }
            
            results["system_metrics"] = {
                "initial_memory_gb": initial_resources["memory_used_gb"],
                "final_memory_gb": final_resources["memory_used_gb"],
                "peak_cpu_percent": max(p["cpu_percent"] for p in results["test_problems"]),
                "avg_cpu_percent": np.mean([p["cpu_percent"] for p in results["test_problems"]]),
                "memory_increase_gb": final_resources["memory_used_gb"] - initial_resources["memory_used_gb"]
            }
            
            self.logger.info(f"Benchmark completed for {model_name}")
            self.logger.info(f"Accuracy: {results['summary_metrics']['accuracy']:.3f}")
            self.logger.info(f"Avg inference time: {results['summary_metrics']['avg_inference_time_seconds']:.2f}s")
            self.logger.info(f"Avg tokens/second: {results['summary_metrics']['avg_tokens_per_second']:.2f}")
            
        except Exception as e:
            error_msg = f"Error benchmarking {model_name}: {str(e)}"
            self.logger.error(error_msg)
            results["errors"].append(error_msg)
            
        finally:
            # Clean up memory
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        return results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run benchmark on all model variants."""
        self.logger.info("Starting comprehensive model benchmark analysis")
        
        benchmark_session = {
            "session_id": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "start_time": datetime.now().isoformat(),
            "test_configuration": {
                "dataset": "GSM8K",
                "test_size": self.test_size,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "device": self.device
            },
            "system_info": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                "platform": sys.platform
            },
            "model_results": {},
            "comparative_analysis": {}
        }
        
        # Benchmark each model
        for model_name, model_config in self.models.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"BENCHMARKING: {model_name}")
            self.logger.info(f"{'='*60}")
            
            # Check if model exists
            model_path = Path(model_config["path"])
            if not model_path.exists() and not model_config["path"].startswith("Qwen/"):
                self.logger.warning(f"Model path does not exist: {model_path}")
                continue
                
            result = self.benchmark_model(model_name, model_config)
            benchmark_session["model_results"][model_name] = result
        
        # Comparative analysis
        benchmark_session["comparative_analysis"] = self.analyze_comparative_performance(
            benchmark_session["model_results"]
        )
        
        benchmark_session["end_time"] = datetime.now().isoformat()
        
        # Save results
        results_file = self.output_dir / f"benchmark_results_{benchmark_session['session_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_session, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark results saved to: {results_file}")
        
        # Generate summary report
        self.generate_summary_report(benchmark_session)
        
        return benchmark_session
    
    def analyze_comparative_performance(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze comparative performance across models."""
        analysis = {
            "accuracy_ranking": [],
            "speed_ranking": [],
            "efficiency_ranking": [],
            "detailed_comparison": {}
        }
        
        # Filter out models with errors
        valid_results = {k: v for k, v in model_results.items() 
                        if "summary_metrics" in v and len(v.get("errors", [])) == 0}
        
        if len(valid_results) < 2:
            return analysis
        
        # Accuracy ranking
        accuracy_data = [(name, result["summary_metrics"]["accuracy"]) 
                        for name, result in valid_results.items()]
        analysis["accuracy_ranking"] = sorted(accuracy_data, key=lambda x: x[1], reverse=True)
        
        # Speed ranking (tokens per second)
        speed_data = [(name, result["summary_metrics"]["avg_tokens_per_second"]) 
                     for name, result in valid_results.items()]
        analysis["speed_ranking"] = sorted(speed_data, key=lambda x: x[1], reverse=True)
        
        # Efficiency ranking (accuracy / inference_time)
        efficiency_data = []
        for name, result in valid_results.items():
            accuracy = result["summary_metrics"]["accuracy"]
            avg_time = result["summary_metrics"]["avg_inference_time_seconds"]
            efficiency = accuracy / avg_time if avg_time > 0 else 0
            efficiency_data.append((name, efficiency))
        analysis["efficiency_ranking"] = sorted(efficiency_data, key=lambda x: x[1], reverse=True)
        
        # Detailed comparisons
        for name, result in valid_results.items():
            metrics = result["summary_metrics"]
            analysis["detailed_comparison"][name] = {
                "accuracy": metrics["accuracy"],
                "avg_inference_time_seconds": metrics["avg_inference_time_seconds"],
                "tokens_per_second": metrics["avg_tokens_per_second"],
                "model_load_time_seconds": metrics["model_load_time_seconds"],
                "memory_usage_gb": result["system_metrics"]["final_memory_gb"],
                "efficiency_score": metrics["accuracy"] / metrics["avg_inference_time_seconds"]
            }
        
        return analysis
    
    def generate_summary_report(self, benchmark_session: Dict[str, Any]) -> None:
        """Generate a human-readable summary report."""
        report_file = self.output_dir / f"benchmark_summary_{benchmark_session['session_id']}.md"
        
        with open(report_file, 'w') as f:
            f.write("# GRPO Model Benchmark Analysis Report\n\n")
            f.write(f"**Analysis Date:** {benchmark_session['start_time'][:10]}\n")
            f.write(f"**Session ID:** {benchmark_session['session_id']}\n")
            f.write(f"**Test Dataset:** GSM8K Mathematical Reasoning\n")
            f.write(f"**Problems Tested:** {benchmark_session['test_configuration']['test_size']}\n\n")
            
            # Performance summary table
            f.write("## Performance Summary\n\n")
            f.write("| Model | Accuracy | Avg Time (s) | Tokens/sec | Load Time (s) | Memory (GB) |\n")
            f.write("|-------|----------|--------------|------------|---------------|-------------|\n")
            
            valid_results = {k: v for k, v in benchmark_session["model_results"].items() 
                           if "summary_metrics" in v and len(v.get("errors", [])) == 0}
            
            for name, result in valid_results.items():
                metrics = result["summary_metrics"]
                sys_metrics = result["system_metrics"]
                f.write(f"| {name} | {metrics['accuracy']:.3f} | "
                       f"{metrics['avg_inference_time_seconds']:.2f} | "
                       f"{metrics['avg_tokens_per_second']:.2f} | "
                       f"{metrics['model_load_time_seconds']:.2f} | "
                       f"{sys_metrics['final_memory_gb']:.2f} |\n")
            
            # Rankings
            analysis = benchmark_session["comparative_analysis"]
            
            f.write("\n## Rankings\n\n")
            f.write("### Accuracy Ranking\n")
            for i, (name, accuracy) in enumerate(analysis["accuracy_ranking"], 1):
                f.write(f"{i}. **{name}**: {accuracy:.3f}\n")
            
            f.write("\n### Speed Ranking (Tokens/Second)\n")
            for i, (name, speed) in enumerate(analysis["speed_ranking"], 1):
                f.write(f"{i}. **{name}**: {speed:.2f} tokens/sec\n")
            
            f.write("\n### Efficiency Ranking (Accuracy/Time)\n")
            for i, (name, efficiency) in enumerate(analysis["efficiency_ranking"], 1):
                f.write(f"{i}. **{name}**: {efficiency:.3f} accuracy/second\n")
            
            # Error summary
            f.write("\n## Errors and Issues\n\n")
            error_found = False
            for name, result in benchmark_session["model_results"].items():
                if result.get("errors"):
                    error_found = True
                    f.write(f"**{name}**: {'; '.join(result['errors'])}\n")
            
            if not error_found:
                f.write("No errors encountered during benchmarking.\n")
            
            f.write("\n## Data Sources\n\n")
            f.write(f"- Detailed results: `benchmark_results_{benchmark_session['session_id']}.json`\n")
            f.write(f"- Log file: `benchmark_analysis_{benchmark_session['session_id']}.log`\n")
            f.write("- Test dataset: GSM8K (grade school math problems)\n")
            f.write("- Evaluation metric: Exact numerical answer matching\n")
        
        self.logger.info(f"Summary report saved to: {report_file}")

def main():
    """Run the comprehensive benchmark analysis."""
    print("GRPO Model Benchmark Analysis")
    print("=" * 50)
    
    # Create benchmark analyzer
    analyzer = BenchmarkAnalyzer()
    
    # Run comprehensive benchmark
    results = analyzer.run_comprehensive_benchmark()
    
    print("\nBenchmark analysis completed!")
    print(f"Results saved in: {analyzer.output_dir}")
    
    return results

if __name__ == "__main__":
    results = main()
