#!/usr/bin/env python3
"""
Stage 3 Model Benchmark Analysis
Tests the progressive training strategy final model against base Qwen on GSM8K
"""

import json
import time
import torch
import psutil
import random
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer with timing and memory tracking"""
    start_time = time.time()
    start_memory = psutil.virtual_memory().used / (1024**3)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        load_time = time.time() - start_time
        end_memory = psutil.virtual_memory().used / (1024**3)
        memory_used = end_memory - start_memory
        
        return model, tokenizer, load_time, memory_used
        
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None, None, 0, 0

def generate_response(model, tokenizer, prompt, max_length=512):
    """Generate response with timing"""
    start_time = time.time()
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = time.time() - start_time
    
    # Calculate tokens generated
    input_length = len(inputs['input_ids'][0])
    output_length = len(outputs[0])
    tokens_generated = output_length - input_length
    
    tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
    
    return response, generation_time, tokens_per_second

def extract_numerical_answer(text):
    """Extract numerical answer from response"""
    import re
    
    # Look for patterns like "The answer is X" or just numbers
    patterns = [
        r"(?:the answer is|answer:|equals?|=)\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:is the answer|$)",
        r"(\d+(?:\.\d+)?)"
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                return float(matches[-1])  # Return the last number found
            except ValueError:
                continue
    
    return None

def evaluate_gsm8k_problem(model, tokenizer, problem, model_name):
    """Evaluate a single GSM8K problem"""
    prompt = f"Question: {problem['question']}\nPlease solve this step by step and provide the numerical answer.\nAnswer:"
    
    try:
        response, gen_time, tokens_per_sec = generate_response(model, tokenizer, prompt)
        predicted_answer = extract_numerical_answer(response)
        
        # Extract ground truth answer
        actual_answer_text = problem['answer'].split('####')[-1].strip()
        try:
            actual_answer = float(actual_answer_text)
        except ValueError:
            actual_answer = None
            
        is_correct = (predicted_answer == actual_answer) if (predicted_answer is not None and actual_answer is not None) else False
        
        return {
            'model': model_name,
            'question': problem['question'],
            'predicted_answer': predicted_answer,
            'actual_answer': actual_answer,
            'correct': is_correct,
            'generation_time': gen_time,
            'tokens_per_second': tokens_per_sec,
            'response': response
        }
        
    except Exception as e:
        print(f"Error evaluating problem with {model_name}: {e}")
        return None

def main():
    print("=== Stage 3 Progressive Training Model Benchmark ===")
    print(f"Start time: {datetime.now()}")
    
    # Model configurations
    models_to_test = [
        {
            'name': 'base_qwen',
            'path': 'Qwen/Qwen2-0.5B-Instruct',
            'description': 'Base Qwen2-0.5B-Instruct model'
        },
        {
            'name': 'stage3_final',
            'path': './models/stage3/final_model',
            'description': 'Stage 3 progressive training final model'
        }
    ]
    
    # Load GSM8K dataset
    print("\nLoading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    # Use subset for testing (same as previous benchmark)
    num_problems = 50
    problems = random.sample(list(dataset), num_problems)
    random.seed(42)  # For reproducibility
    
    print(f"Testing {num_problems} problems from GSM8K")
    
    # Results storage
    all_results = []
    model_summaries = {}
    
    for model_config in models_to_test:
        model_name = model_config['name']
        model_path = model_config['path']
        
        print(f"\n--- Testing {model_name} ---")
        print(f"Model path: {model_path}")
        
        # Load model
        model, tokenizer, load_time, memory_used = load_model_and_tokenizer(model_path)
        
        if model is None:
            print(f"Failed to load {model_name}, skipping...")
            continue
            
        print(f"Model loaded in {load_time:.2f} seconds, Memory used: {memory_used:.2f} GB")
        
        # Test problems
        model_results = []
        total_time = 0
        total_tokens_per_sec = 0
        correct_count = 0
        
        for i, problem in enumerate(problems):
            print(f"Problem {i+1}/{num_problems}", end=" ")
            
            result = evaluate_gsm8k_problem(model, tokenizer, problem, model_name)
            if result:
                model_results.append(result)
                total_time += result['generation_time']
                total_tokens_per_sec += result['tokens_per_second']
                if result['correct']:
                    correct_count += 1
                    print("✓")
                else:
                    print("✗")
            else:
                print("E")
        
        # Calculate summary statistics
        if model_results:
            avg_time = total_time / len(model_results)
            avg_tokens_per_sec = total_tokens_per_sec / len(model_results)
            accuracy = correct_count / len(model_results)
            
            model_summaries[model_name] = {
                'accuracy': accuracy,
                'avg_generation_time': avg_time,
                'avg_tokens_per_second': avg_tokens_per_sec,
                'load_time': load_time,
                'memory_used': memory_used,
                'problems_tested': len(model_results),
                'correct_answers': correct_count,
                'description': model_config['description']
            }
            
            print(f"Results: {correct_count}/{len(model_results)} correct ({accuracy:.1%})")
            print(f"Avg time: {avg_time:.2f}s, Avg tokens/sec: {avg_tokens_per_sec:.2f}")
        
        all_results.extend(model_results)
        
        # Clear memory
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"experiments/results/stage3_benchmark_results_{timestamp}.json"
    summary_file = f"experiments/results/stage3_benchmark_summary_{timestamp}.md"
    
    # Ensure results directory exists
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    
    # Save detailed JSON results
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'models_tested': list(model_summaries.keys()),
            'problems_tested': num_problems,
            'detailed_results': all_results,
            'model_summaries': model_summaries
        }, f, indent=2)
    
    # Generate summary report
    with open(summary_file, 'w') as f:
        f.write("# Stage 3 Progressive Training Model Benchmark Report\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Session ID:** {timestamp}\n")
        f.write(f"**Test Dataset:** GSM8K Mathematical Reasoning\n")
        f.write(f"**Problems Tested:** {num_problems}\n")
        f.write(f"**Training Context:** Stage 3 model used progressive training strategy\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Model | Accuracy | Avg Time (s) | Tokens/sec | Load Time (s) | Memory (GB) |\n")
        f.write("|-------|----------|--------------|------------|---------------|--------------|\n")
        
        for model_name, summary in model_summaries.items():
            f.write(f"| {model_name} | {summary['accuracy']:.3f} | {summary['avg_generation_time']:.2f} | "
                   f"{summary['avg_tokens_per_second']:.2f} | {summary['load_time']:.2f} | {summary['memory_used']:.2f} |\n")
        
        f.write("\n## Model Descriptions\n\n")
        for model_name, summary in model_summaries.items():
            f.write(f"**{model_name}:** {summary['description']}\n")
        
        # Rankings
        f.write("\n## Rankings\n\n")
        
        # Accuracy ranking
        f.write("### Accuracy Ranking\n")
        sorted_by_accuracy = sorted(model_summaries.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        for i, (model_name, summary) in enumerate(sorted_by_accuracy, 1):
            f.write(f"{i}. **{model_name}**: {summary['accuracy']:.3f}\n")
        
        # Speed ranking
        f.write("\n### Speed Ranking (Tokens/Second)\n")
        sorted_by_speed = sorted(model_summaries.items(), key=lambda x: x[1]['avg_tokens_per_second'], reverse=True)
        for i, (model_name, summary) in enumerate(sorted_by_speed, 1):
            f.write(f"{i}. **{model_name}**: {summary['avg_tokens_per_second']:.2f} tokens/sec\n")
        
        # Efficiency ranking
        f.write("\n### Efficiency Ranking (Accuracy/Time)\n")
        sorted_by_efficiency = sorted(model_summaries.items(), 
                                    key=lambda x: x[1]['accuracy'] / x[1]['avg_generation_time'] if x[1]['avg_generation_time'] > 0 else 0, 
                                    reverse=True)
        for i, (model_name, summary) in enumerate(sorted_by_efficiency, 1):
            efficiency = summary['accuracy'] / summary['avg_generation_time'] if summary['avg_generation_time'] > 0 else 0
            f.write(f"{i}. **{model_name}**: {efficiency:.3f} accuracy/second\n")
        
        f.write("\n## Progressive Training Analysis\n\n")
        f.write("### Training Strategy Context\n")
        f.write("The stage3_final model was trained using a progressive training strategy:\n")
        f.write("- **Stage 1:** Basic reasoning task adaptation\n")
        f.write("- **Stage 2:** Intermediate complexity reasoning\n") 
        f.write("- **Stage 3:** Advanced mathematical reasoning focus\n\n")
        
        f.write("### Expected Improvements\n")
        f.write("Progressive training should provide:\n")
        f.write("- Better mathematical reasoning capability\n")
        f.write("- More stable training convergence\n")
        f.write("- Reduced catastrophic forgetting\n")
        f.write("- Improved generalization to unseen problems\n\n")
        
        f.write("## Data Sources\n\n")
        f.write(f"- Detailed results: `stage3_benchmark_results_{timestamp}.json`\n")
        f.write(f"- Stage 3 training logs: `models/stage3/training.log`\n")
        f.write(f"- Training summary: `models/stage3/training_summary.json`\n")
        f.write("- Test dataset: GSM8K (grade school math problems)\n")
        f.write("- Evaluation metric: Exact numerical answer matching\n\n")
    
    print(f"\n=== Benchmark Complete ===")
    print(f"Results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()
