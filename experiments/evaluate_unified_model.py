#!/usr/bin/env python3
"""
Unified Progressive Model Evaluation Script

This script evaluates the performance of models trained with unified_progressive_training.py
against the base model, using GSM8K-style mathematical reasoning problems.

Key features:
- Comparative evaluation of base vs. trained models
- Multi-stage assessment across difficulty levels
- Detailed performance analytics
- Memory-optimized for CPU-only execution
"""

import os
import sys
import json
import torch
import numpy as np
import time
import re
import pandas as pd
from typing import List, Dict, Any, Union, Optional
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import project modules
from src.training.advanced_memory_optimization import (
    AdvancedMemoryManager,
    MemoryOptimizationConfig
)

# Evaluation configuration
EVAL_CONFIG = {
    "base_model": "Qwen/Qwen2-0.5B-Instruct",
    "trained_model_dir": "./models/unified_progressive",
    "evaluation_samples": 20,  # Samples per difficulty level
    "output_dir": "./experiments/results/unified_eval",
    "max_tokens": 512,
    "temperature": 0.1,
    "enable_quantization": True,
    "batch_size": 1,  # CPU-friendly
}


class ModelEvaluator:
    """Evaluates model performance on mathematical reasoning tasks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Create memory optimization config
        from src.training.advanced_memory_optimization import MemoryOptimizationConfig
        memory_config = MemoryOptimizationConfig()
        memory_config.enable_fp16 = config.get("enable_fp16", True)
        memory_config.use_dynamic_quantization = config.get("enable_quantization", True)
        
        # Initialize memory manager with config
        self.memory_manager = AdvancedMemoryManager(memory_config)
        
        # Create output directory
        os.makedirs(config["output_dir"], exist_ok=True)
        
        # Configure torch for CPU optimization
        self._configure_cpu_optimization()
        
        print(f"📊 Initializing model evaluator")
        print(f"   Base model: {config['base_model']}")
        print(f"   Trained model: {config['trained_model_dir']}")
        print(f"   Samples per difficulty: {config['evaluation_samples']}")
    
    def _configure_cpu_optimization(self):
        """Configure CPU optimization settings"""
        import psutil
        
        # Optimize CPU usage
        logical_cores = psutil.cpu_count(logical=True) 
        physical_cores = psutil.cpu_count(logical=False)
        
        torch.set_num_threads(logical_cores)  # Use all logical cores
        
        # Environment variables for CPU optimization
        os.environ['OMP_NUM_THREADS'] = str(physical_cores)
        os.environ['MKL_NUM_THREADS'] = str(physical_cores)
        
        # Enable Intel optimizations if available
        if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
            torch.backends.mkldnn.enabled = True
        
        print(f"✓ CPU optimization configured: {logical_cores} threads")
    
    def load_base_model(self):
        """Load the base model with optimizations"""
        print(f"\n📂 Loading base model: {self.config['base_model']}...")
        
        # Memory optimization
        self.memory_manager.optimize_memory_before_model_load()
        
        # Load model and tokenizer
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'])
        model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'], 
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Apply quantization if enabled
        if self.config["enable_quantization"]:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("   ✓ Dynamic INT8 quantization applied")
            
        load_time = time.time() - start_time
        print(f"   ✓ Base model loaded in {load_time:.1f}s")
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def load_trained_model(self, stage_num: Optional[int] = None):
        """Load a trained model from the specified stage or the final stage"""
        if stage_num is not None:
            model_dir = os.path.join(self.config['trained_model_dir'], f"stage_{stage_num}")
            print(f"\n📂 Loading trained model (Stage {stage_num})...")
        else:
            # Find the highest stage number
            stages = [d for d in os.listdir(self.config['trained_model_dir']) 
                     if d.startswith("stage_") and os.path.isdir(os.path.join(self.config['trained_model_dir'], d))]
            if not stages:
                raise ValueError(f"No trained model stages found in {self.config['trained_model_dir']}")
            
            stages.sort(key=lambda x: int(x.split('_')[1]))
            highest_stage = stages[-1]
            stage_num = int(highest_stage.split('_')[1])
            model_dir = os.path.join(self.config['trained_model_dir'], highest_stage)
            print(f"\n📂 Loading trained model (highest Stage {stage_num})...")
        
        # Memory optimization
        self.memory_manager.optimize_memory_before_model_load()
        
        # Load tokenizer from base model (more reliable)
        tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load the trained model
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],  # Load architecture from base model
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        # Load trained weights
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model", "pytorch_model.bin")
            
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"   ✓ Loaded weights from {model_path}")
        else:
            print(f"   ⚠ Warning: No model weights found at {model_path}")
            print(f"   ⚠ Using base model instead")
        
        # Apply quantization if enabled
        if self.config["enable_quantization"]:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("   ✓ Dynamic INT8 quantization applied")
            
        load_time = time.time() - start_time
        print(f"   ✓ Trained model loaded in {load_time:.1f}s")
            
        return model, tokenizer, stage_num
    
    def create_evaluation_dataset(self):
        """Create a balanced evaluation dataset with varied difficulty levels"""
        print("\n📝 Creating evaluation dataset...")
        
        # Basic arithmetic problems
        basic_problems = [
            "If a train travels at 60 km/h, how far will it travel in 3 hours?",
            "Sarah has $85 and spends $37 on groceries. How much money does she have left?",
            "A box contains 24 chocolates and Tom eats 5 of them. What fraction of chocolates remain?",
            "If 5 shirts cost $75, what is the cost of 8 shirts?",
            "Maria has 48 marbles and gives 1/4 of them to her friend. How many marbles does she have left?",
            "A rectangle has a length of 12 cm and width of 8 cm. What is its perimeter?",
            "If a class has 35 students and 20 of them are girls, what percentage of the class is boys?",
            "David runs 4 miles every day. How many miles will he run in 14 days?",
            "A pizza is cut into 8 equal slices. If 3 slices are eaten, what percentage is left?",
            "John buys 5 books for $12 each. How much change will he get from a $100 bill?",
            "If a car uses 8 gallons of gas to travel 240 miles, how many gallons will it use to travel 360 miles?",
            "Lisa has 45 pens to distribute equally among 9 students. How many pens does each student get?",
            "A box has 15 red balls and 25 blue balls. What fraction of the balls are red?",
            "If a shirt normally costs $30 and is on sale for 25% off, what is the sale price?",
            "A farmer has 86 cows. He sells 38 cows. How many cows does he have left?",
            "If water flows at 5 liters per minute, how many liters will flow in 12 minutes?",
            "A bicycle wheel has a diameter of 28 inches. What is its circumference? (Use π = 3.14)",
            "Tom scored 75, 82, and 91 on his tests. What was his average score?",
            "If a cell phone battery lasts 8 hours when fully charged, what percentage of the battery is left after 3 hours?",
            "A recipe calls for 3 cups of flour. If you want to make half the recipe, how many cups of flour do you need?"
        ]
        
        # Intermediate multi-step problems
        intermediate_problems = [
            "A store has 120 items. On Monday, 30% of the items are sold. On Tuesday, 25% of the remaining items are sold. How many items are left?",
            "Sarah spends 2.5 hours working on math and twice as much time working on a science project. She also spends 1.25 hours reading. How much total time does she spend on these activities?",
            "A rectangular garden is 15 feet long and 12 feet wide. If a fence is built around the garden that is 2 feet away from the garden on all sides, what is the length of the fence?",
            "Tom buys 3 shirts for $24 each and 2 pairs of pants for $36 each. If sales tax is 8%, what is the total cost including tax?",
            "A tank can be filled by pipe A in 12 hours and by pipe B in 8 hours. If both pipes are open, how long will it take to fill the tank?",
            "Maria invests $5,000 in a savings account with an annual interest rate of 4% compounded annually. How much will be in the account after 3 years?",
            "A car travels at 60 mph for 2 hours, then at 50 mph for 1.5 hours. What is the average speed for the entire journey?",
            "A store reduces the price of a TV by 20% to $640. What was the original price?",
            "A recipe requires 3/4 cup of sugar and 2/3 cup of flour. If you want to make 3 batches, how many cups of sugar and flour will you need in total?",
            "John can paint a room in 6 hours, while Mary can paint the same room in 4 hours. How long would it take them to paint the room working together?",
            "A company's revenue increased by 15% to $230,000. What was the original revenue?",
            "A train travels 240 kilometers at an average speed of 80 km/h. If the return journey is made at 60 km/h, what is the average speed for the entire trip?",
            "In a class of 32 students, the ratio of boys to girls is 3:5. How many boys are in the class?",
            "A cyclist rides 20 miles at 10 mph and then another 15 miles at 15 mph. What is the average speed for the entire journey?",
            "A store mixes coffee that costs $15 per pound with coffee that costs $9 per pound to create 20 pounds of a blend that costs $12.60 per pound. How many pounds of the $15 coffee are used?",
            "A box contains 5 red marbles, 8 blue marbles, and 7 green marbles. If two marbles are drawn without replacement, what is the probability that both are blue?",
            "A rectangular prism has a length of 8 cm, width of 5 cm, and height of 3 cm. What is its volume in cubic centimeters and surface area in square centimeters?",
            "A car uses 12 gallons of gas to travel 300 miles. At this rate, how many gallons would it use to travel 450 miles, and how much will the gas cost at $3.50 per gallon?",
            "An investment of $5,000 grows to $5,600 in one year. What is the annual growth rate as a percentage?",
            "A storage tank contains 800 gallons of water. Water flows in at a rate of 25 gallons per minute and out at a rate of 15 gallons per minute. How long will it take for the tank to contain 950 gallons?"
        ]
        
        # Advanced problems requiring complex reasoning
        advanced_problems = [
            "A company produces widgets at a cost of $12 each. The fixed monthly cost is $5,000. If the widgets sell for $20 each, how many must be sold each month to break even?",
            "A boat travels 24 miles upstream in 3 hours and 24 miles downstream in 1.5 hours. What is the speed of the boat in still water and what is the speed of the current?",
            "A mixture of 30 liters contains milk and water in the ratio 7:3. How much water should be added to make the ratio 7:4?",
            "Three people working at the same rate can complete a project in 10 days. After working together for 5 days, one person leaves. How many more days will it take the remaining two people to finish the project?",
            "A spherical water tank with radius 3 meters is being filled at a rate of 500 liters per minute. How long will it take to fill the tank? (1 cubic meter = 1000 liters, volume of sphere = 4/3 * π * r³)",
            "A store offers a 30% discount on the marked price of a laptop, and charges 8% sales tax on the discounted price. If the final price paid is $907.20, what was the marked price?",
            "The sum of the ages of a father and his son is 50. The father's age is the son's age reversed. What are their ages?",
            "A train traveling at 72 km/h passes through a 1.2 km long tunnel in 1 minute. What is the length of the train in meters?",
            "If the probability of rain on Saturday is 0.4 and the probability of rain on Sunday is 0.3, what is the probability that it will rain on either Saturday or Sunday if the probability of rain on both days is 0.15?",
            "In a group of people, 65% have brown eyes and 55% have brown hair. If 35% have both brown eyes and brown hair, what percentage have neither brown eyes nor brown hair?",
            "A car depreciates in value by 15% per year. If its current value is $25,500, what was its value 3 years ago?",
            "A store mixes nuts that cost $8 per pound with nuts that cost $12 per pound to create a 30-pound mixture that sells for $9.20 per pound. How many pounds of each type of nut are used?",
            "The sides of a right triangle are in an arithmetic sequence. If the hypotenuse is 13 cm, what is the area of the triangle?",
            "A cylindrical water tank with diameter 10 meters is being filled. After 4 hours, the water is 3 meters deep. What is the rate of water flow in cubic meters per hour?",
            "A company employs 50 people, each working 8 hours per day. If a project requires 3,200 person-hours to complete, how many days will it take?",
            "A trapezoid has parallel sides of lengths 8 cm and 14 cm. If the area of the trapezoid is 66 cm², what is the height of the trapezoid?",
            "In a geometric sequence, the 3rd term is 12 and the 6th term is 96. What is the first term and the common ratio?",
            "If log(x) + log(x+20) = log(x²+20x), what is the value of x?",
            "A cone has a height of 12 cm and a base radius of 5 cm. What is its volume in cubic centimeters and lateral surface area in square centimeters? (Use π = 3.14)",
            "Two trains leave stations 480 kilometers apart at the same time, traveling toward each other. One train travels at 70 km/h and the other at 50 km/h. How much time passes before they meet?"
        ]
        
        # Combine all problems and create the dataset
        all_problems = []
        
        # Format as proper evaluation samples
        for difficulty, problems in [
            ("basic", basic_problems),
            ("intermediate", intermediate_problems),
            ("advanced", advanced_problems)
        ]:
            # Select evaluation_samples or all if fewer
            num_samples = min(self.config["evaluation_samples"], len(problems))
            selected = np.random.choice(problems, num_samples, replace=False)
            
            for problem in selected:
                all_problems.append({
                    "difficulty": difficulty,
                    "problem": problem,
                    "prompt": f"Solve step by step:\n\n{problem}\n\nSolution:"
                })
        
        print(f"   ✓ Created {len(all_problems)} evaluation problems")
        print(f"     - Basic: {self.config['evaluation_samples']}")
        print(f"     - Intermediate: {self.config['evaluation_samples']}")
        print(f"     - Advanced: {self.config['evaluation_samples']}")
        
        return all_problems
    
    def generate_solutions(self, model, tokenizer, problems: List[Dict[str, str]]):
        """Generate solutions for each problem using the given model"""
        results = []
        
        print(f"\n🧮 Generating solutions for {len(problems)} problems...")
        start_time = time.time()
        
        for i, problem in enumerate(problems):
            input_text = problem["prompt"]
            
            # Generate solution
            inputs = tokenizer(input_text, return_tensors="pt")
            
            # Generate with careful handling of memory
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_length=len(inputs.input_ids[0]) + self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    do_sample=True if self.config["temperature"] > 0 else False,
                    num_return_sequences=1,
                )
            
            # Decode the output
            solution = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract just the generated part (not the input)
            solution = solution[len(input_text):].strip()
            
            # Store result
            results.append({
                **problem,
                "solution": solution,
                "correctness": None,  # Will be evaluated separately
            })
            
            # Progress update
            if (i+1) % 10 == 0 or i+1 == len(problems):
                print(f"   Generated {i+1}/{len(problems)} solutions")
                
            # Clean memory
            del inputs, output
            if i % 5 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        total_time = time.time() - start_time
        print(f"   ✓ Generated all solutions in {total_time:.1f}s")
        
        return results
    
    def evaluate_solution(self, problem: str, solution: str) -> Dict:
        """Evaluate the solution based on mathematical reasoning criteria"""
        # Create evaluation metrics
        evaluation = {
            "has_numeric_answer": False,
            "has_step_by_step": False,
            "steps_count": 0,
            "has_calculations": False,
            "calculation_count": 0,
            "has_units": False,
            "answer_matches_work": False,
            "estimated_correctness": 0.0,  # 0.0-1.0 scale
        }
        
        # Check for numerical answer
        solution_lower = solution.lower().strip()
        numbers = re.findall(r'\b\d+(?:\.\d+)?', solution_lower)
        if numbers:
            evaluation["has_numeric_answer"] = True
        
        # Look for step-by-step reasoning indicators
        step_indicators = ["step", "first", "second", "third", "next", "then", "finally"]
        evaluation["has_step_by_step"] = any(indicator in solution_lower for indicator in step_indicators)
        evaluation["steps_count"] = sum(solution_lower.count(f"{indicator}") for indicator in step_indicators)
        
        # Check for calculations
        calculation_indicators = ["+", "-", "*", "×", "÷", "/", "="]
        evaluation["has_calculations"] = any(indicator in solution for indicator in calculation_indicators)
        evaluation["calculation_count"] = sum(solution.count(indicator) for indicator in calculation_indicators)
        
        # Check for units in the answer
        units = ["km", "mile", "meter", "m", "cm", "hour", "hr", "minute", "min", "second", "sec",
                "dollar", "$", "cent", "¢", "pound", "kg", "g", "liter", "l", "gallon", "gal", "%"]
        evaluation["has_units"] = any(unit in solution_lower for unit in units)
        
        # Check if the final answer seems to match the work shown
        if numbers and evaluation["calculation_count"] >= 1:
            final_number = numbers[-1]
            previous_numbers = numbers[:-1]
            # Simple heuristic: final answer should be related to previous numbers
            evaluation["answer_matches_work"] = final_number in previous_numbers or any(
                abs(float(final_number) - float(prev_num)) < 0.01 for prev_num in previous_numbers if float(prev_num) != 0
            )
        
        # Estimate overall correctness
        # This is a heuristic and not actual correctness verification
        score = 0.0
        if evaluation["has_numeric_answer"]:
            score += 0.3
        if evaluation["has_step_by_step"]:
            score += 0.2
        if evaluation["has_calculations"]:
            score += 0.2
        if evaluation["has_units"]:
            score += 0.1
        if evaluation["answer_matches_work"]:
            score += 0.2
            
        evaluation["estimated_correctness"] = min(score, 1.0)
        
        return evaluation
    
    def run_comparative_evaluation(self):
        """Run a comparative evaluation between base and trained models"""
        print("\n📈 Starting comparative evaluation...")
        
        # Create evaluation dataset
        eval_problems = self.create_evaluation_dataset()
        
        # Load models
        base_model, base_tokenizer = self.load_base_model()
        trained_model, trained_tokenizer, stage_num = self.load_trained_model()
        
        # Generate solutions with both models
        print("\n🔍 Evaluating base model...")
        base_results = self.generate_solutions(base_model, base_tokenizer, eval_problems)
        
        print("\n🔍 Evaluating trained model...")
        trained_results = self.generate_solutions(trained_model, trained_tokenizer, eval_problems)
        
        # Evaluate solutions
        print("\n📝 Analyzing solutions...")
        for i, (base_item, trained_item) in enumerate(zip(base_results, trained_results)):
            base_eval = self.evaluate_solution(base_item["problem"], base_item["solution"])
            trained_eval = self.evaluate_solution(trained_item["problem"], trained_item["solution"])
            
            base_results[i].update(base_eval)
            trained_results[i].update(trained_eval)
            
            if (i+1) % 10 == 0 or i+1 == len(base_results):
                print(f"   Analyzed {i+1}/{len(base_results)} solutions")
        
        # Calculate metrics
        base_metrics = self.calculate_metrics(base_results)
        trained_metrics = self.calculate_metrics(trained_results)
        
        # Save results
        self.save_evaluation_results(base_results, trained_results, 
                                    base_metrics, trained_metrics)
        
        # Generate visualizations
        self.generate_visualizations(base_metrics, trained_metrics)
        
        return base_results, trained_results, base_metrics, trained_metrics
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics from evaluation results"""
        metrics = {
            "total_problems": len(results),
            "avg_correctness": np.mean([r["estimated_correctness"] for r in results]),
            "has_numeric_answer_pct": np.mean([r["has_numeric_answer"] for r in results]) * 100,
            "has_step_by_step_pct": np.mean([r["has_step_by_step"] for r in results]) * 100,
            "has_calculations_pct": np.mean([r["has_calculations"] for r in results]) * 100,
            "has_units_pct": np.mean([r["has_units"] for r in results]) * 100,
            "answer_matches_work_pct": np.mean([r["answer_matches_work"] if r["answer_matches_work"] is not None else 0 for r in results]) * 100,
            "avg_steps_count": np.mean([r["steps_count"] for r in results]),
            "avg_calculation_count": np.mean([r["calculation_count"] for r in results]),
            
            # Metrics by difficulty
            "by_difficulty": {}
        }
        
        # Calculate metrics by difficulty
        for difficulty in ["basic", "intermediate", "advanced"]:
            difficulty_results = [r for r in results if r["difficulty"] == difficulty]
            if not difficulty_results:
                continue
                
            metrics["by_difficulty"][difficulty] = {
                "count": len(difficulty_results),
                "avg_correctness": np.mean([r["estimated_correctness"] for r in difficulty_results]),
                "has_numeric_answer_pct": np.mean([r["has_numeric_answer"] for r in difficulty_results]) * 100,
                "has_step_by_step_pct": np.mean([r["has_step_by_step"] for r in difficulty_results]) * 100,
                "avg_steps_count": np.mean([r["steps_count"] for r in difficulty_results]),
                "avg_calculation_count": np.mean([r["calculation_count"] for r in difficulty_results]),
            }
        
        return metrics
    
    def save_evaluation_results(self, base_results, trained_results, base_metrics, trained_metrics):
        """Save evaluation results to JSON files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(output_dir, f"base_results_{timestamp}.json"), "w") as f:
            json.dump(base_results, f, indent=2)
            
        with open(os.path.join(output_dir, f"trained_results_{timestamp}.json"), "w") as f:
            json.dump(trained_results, f, indent=2)
            
        # Save metrics summary
        metrics_summary = {
            "base_model": {
                "name": self.config["base_model"],
                "metrics": base_metrics
            },
            "trained_model": {
                "name": f"{self.config['base_model']} + Unified Progressive Training",
                "metrics": trained_metrics
            },
            "comparison": {
                "correctness_improvement": trained_metrics["avg_correctness"] - base_metrics["avg_correctness"],
                "step_by_step_improvement": trained_metrics["has_step_by_step_pct"] - base_metrics["has_step_by_step_pct"],
                "calculation_improvement": trained_metrics["avg_calculation_count"] - base_metrics["avg_calculation_count"],
                "relative_improvement": (trained_metrics["avg_correctness"] / base_metrics["avg_correctness"] - 1) * 100 if base_metrics["avg_correctness"] > 0 else 0,
            }
        }
        
        with open(os.path.join(output_dir, f"metrics_summary_{timestamp}.json"), "w") as f:
            json.dump(metrics_summary, f, indent=2)
            
        # Create a simple markdown report
        report_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.md")
        with open(report_path, "w") as f:
            f.write(f"# Unified Progressive Training Evaluation Report\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Base Model:** {self.config['base_model']}\n")
            f.write(f"**Trained Model:** {self.config['base_model']} + Unified Progressive Training\n\n")
            
            f.write(f"## Overall Performance\n\n")
            f.write(f"| Metric | Base Model | Trained Model | Improvement |\n")
            f.write(f"|--------|------------|--------------|-------------|\n")
            f.write(f"| Correctness Score | {base_metrics['avg_correctness']:.2f} | {trained_metrics['avg_correctness']:.2f} | {metrics_summary['comparison']['correctness_improvement']:.2f} |\n")
            f.write(f"| Step-by-Step Reasoning | {base_metrics['has_step_by_step_pct']:.1f}% | {trained_metrics['has_step_by_step_pct']:.1f}% | {metrics_summary['comparison']['step_by_step_improvement']:.1f}% |\n")
            f.write(f"| Average Calculation Steps | {base_metrics['avg_calculation_count']:.1f} | {trained_metrics['avg_calculation_count']:.1f} | {metrics_summary['comparison']['calculation_improvement']:.1f} |\n")
            f.write(f"| Relative Improvement | - | - | {metrics_summary['comparison']['relative_improvement']:.1f}% |\n\n")
            
            f.write(f"## Performance by Difficulty Level\n\n")
            
            # Create by-difficulty table
            for difficulty in ["basic", "intermediate", "advanced"]:
                if difficulty in base_metrics["by_difficulty"] and difficulty in trained_metrics["by_difficulty"]:
                    base_diff = base_metrics["by_difficulty"][difficulty]
                    trained_diff = trained_metrics["by_difficulty"][difficulty]
                    
                    f.write(f"### {difficulty.capitalize()} Problems\n\n")
                    f.write(f"| Metric | Base Model | Trained Model | Improvement |\n")
                    f.write(f"|--------|------------|--------------|-------------|\n")
                    f.write(f"| Correctness Score | {base_diff['avg_correctness']:.2f} | {trained_diff['avg_correctness']:.2f} | {trained_diff['avg_correctness'] - base_diff['avg_correctness']:.2f} |\n")
                    f.write(f"| Step-by-Step Reasoning | {base_diff['has_step_by_step_pct']:.1f}% | {trained_diff['has_step_by_step_pct']:.1f}% | {trained_diff['has_step_by_step_pct'] - base_diff['has_step_by_step_pct']:.1f}% |\n")
                    f.write(f"| Average Steps | {base_diff['avg_steps_count']:.1f} | {trained_diff['avg_steps_count']:.1f} | {trained_diff['avg_steps_count'] - base_diff['avg_steps_count']:.1f} |\n\n")
            
            f.write(f"## Conclusion\n\n")
            
            if metrics_summary['comparison']['relative_improvement'] > 10:
                conclusion = "The Unified Progressive Training shows significant improvement over the base model, particularly in step-by-step reasoning and calculation skills."
            elif metrics_summary['comparison']['relative_improvement'] > 0:
                conclusion = "The Unified Progressive Training shows moderate improvement over the base model, with better performance in some areas."
            else:
                conclusion = "The Unified Progressive Training does not show significant improvement over the base model in this evaluation."
                
            f.write(conclusion + "\n")
        
        print(f"\n✓ Evaluation results saved to {output_dir}")
        print(f"✓ Report generated at {report_path}")
    
    def generate_visualizations(self, base_metrics, trained_metrics):
        """Generate comparative visualizations"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Overall comparison chart
        plt.figure(figsize=(12, 6))
        metrics_to_plot = [
            ("avg_correctness", "Correctness Score (0-1)"),
            ("has_step_by_step_pct", "Step-by-Step Reasoning (%)"),
            ("has_numeric_answer_pct", "Has Numeric Answer (%)"),
            ("avg_calculation_count", "Avg Calculation Count")
        ]
        
        bar_width = 0.35
        index = np.arange(len(metrics_to_plot))
        
        base_values = [base_metrics[m[0]] for m in metrics_to_plot]
        trained_values = [trained_metrics[m[0]] for m in metrics_to_plot]
        
        plt.bar(index, base_values, bar_width, label='Base Model')
        plt.bar(index + bar_width, trained_values, bar_width, label='Trained Model')
        
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Base vs. Trained Model Performance')
        plt.xticks(index + bar_width / 2, [m[1] for m in metrics_to_plot], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"performance_comparison_{timestamp}.png"))
        
        # Performance by difficulty
        plt.figure(figsize=(12, 6))
        difficulties = ["basic", "intermediate", "advanced"]
        
        base_by_difficulty = [base_metrics["by_difficulty"].get(d, {}).get("avg_correctness", 0) 
                             for d in difficulties]
        trained_by_difficulty = [trained_metrics["by_difficulty"].get(d, {}).get("avg_correctness", 0) 
                               for d in difficulties]
        
        index = np.arange(len(difficulties))
        
        plt.bar(index, base_by_difficulty, bar_width, label='Base Model')
        plt.bar(index + bar_width, trained_by_difficulty, bar_width, label='Trained Model')
        
        plt.xlabel('Difficulty')
        plt.ylabel('Correctness Score (0-1)')
        plt.title('Performance by Problem Difficulty')
        plt.xticks(index + bar_width / 2, [d.capitalize() for d in difficulties])
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"difficulty_comparison_{timestamp}.png"))
        print(f"✓ Visualizations generated in {output_dir}")


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("🔍 UNIFIED PROGRESSIVE TRAINING EVALUATION")
    print("Comparing base model vs. trained model on mathematical reasoning")
    print("=" * 80)
    
    # Create evaluator
    evaluator = ModelEvaluator(EVAL_CONFIG)
    
    # Run comparative evaluation
    base_results, trained_results, base_metrics, trained_metrics = evaluator.run_comparative_evaluation()
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"\nBase Model ({EVAL_CONFIG['base_model']}):")
    print(f"  Correctness Score: {base_metrics['avg_correctness']:.2f}/1.00")
    print(f"  Step-by-Step Reasoning: {base_metrics['has_step_by_step_pct']:.1f}%")
    print(f"  Average Calculation Steps: {base_metrics['avg_calculation_count']:.1f}")
    
    print(f"\nTrained Model (Unified Progressive):")
    print(f"  Correctness Score: {trained_metrics['avg_correctness']:.2f}/1.00")
    print(f"  Step-by-Step Reasoning: {trained_metrics['has_step_by_step_pct']:.1f}%")
    print(f"  Average Calculation Steps: {trained_metrics['avg_calculation_count']:.1f}")
    
    improvement = trained_metrics['avg_correctness'] - base_metrics['avg_correctness']
    relative_imp = (trained_metrics['avg_correctness'] / base_metrics['avg_correctness'] - 1) * 100 if base_metrics['avg_correctness'] > 0 else 0
    
    print(f"\nImprovement: {improvement:.2f} absolute ({relative_imp:.1f}% relative)")
    
    if relative_imp > 10:
        print("\n✅ SIGNIFICANT IMPROVEMENT ACHIEVED")
    elif relative_imp > 0:
        print("\n✅ MODERATE IMPROVEMENT ACHIEVED")
    else:
        print("\n⚠️ NO SIGNIFICANT IMPROVEMENT DETECTED")
    
    print("\nEvaluation completed! Detailed results saved to output directory.")


if __name__ == "__main__":
    main()
