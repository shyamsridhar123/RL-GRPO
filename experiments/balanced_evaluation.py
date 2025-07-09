#!/usr/bin/env python3
"""
Enhanced Evaluation Script
Tests models on a balanced set of mathematical problems to measure accuracy improvements
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def create_balanced_test_problems() -> List[Dict[str, Any]]:
    """Create a balanced set of test problems covering all mathematical operations"""
    
    problems = [
        # Addition (4 problems)
        {
            "question": "What is 127 + 284?",
            "expected": "411",
            "category": "addition",
            "difficulty": "basic"
        },
        {
            "question": "Sarah has 156 stickers. Her friend gives her 89 more stickers. How many stickers does Sarah have now?",
            "expected": "245",
            "category": "addition",
            "difficulty": "word_problem"
        },
        {
            "question": "A store sold 342 items on Monday and 178 items on Tuesday. How many items did they sell in total?",
            "expected": "520",
            "category": "addition",
            "difficulty": "word_problem"
        },
        {
            "question": "What is 45 + 67 + 23?",
            "expected": "135",
            "category": "addition",
            "difficulty": "multi_number"
        },
        
        # Subtraction (4 problems)
        {
            "question": "What is 456 - 178?",
            "expected": "278",
            "category": "subtraction",
            "difficulty": "basic"
        },
        {
            "question": "Tom had 234 marbles. He gave away 97 marbles to his friends. How many marbles does Tom have left?",
            "expected": "137",
            "category": "subtraction",
            "difficulty": "word_problem"
        },
        {
            "question": "A library had 567 books. After lending some books, they have 289 books remaining. How many books were lent?",
            "expected": "278",
            "category": "subtraction",
            "difficulty": "word_problem"
        },
        {
            "question": "What is 1000 - 347?",
            "expected": "653",
            "category": "subtraction",
            "difficulty": "basic"
        },
        
        # Multiplication (4 problems)
        {
            "question": "What is 23 × 17?",
            "expected": "391",
            "category": "multiplication",
            "difficulty": "basic"
        },
        {
            "question": "A box contains 24 pencils. If there are 15 boxes, how many pencils are there in total?",
            "expected": "360",
            "category": "multiplication",
            "difficulty": "word_problem"
        },
        {
            "question": "Each classroom has 28 students. If there are 12 classrooms, how many students are there in total?",
            "expected": "336",
            "category": "multiplication",
            "difficulty": "word_problem"
        },
        {
            "question": "What is 45 × 8?",
            "expected": "360",
            "category": "multiplication",
            "difficulty": "basic"
        },
        
        # Division (4 problems)
        {
            "question": "What is 456 ÷ 12?",
            "expected": "38",
            "category": "division",
            "difficulty": "basic"
        },
        {
            "question": "A pizza is cut into 8 equal slices. If there are 64 slices total, how many pizzas are there?",
            "expected": "8",
            "category": "division",
            "difficulty": "word_problem"
        },
        {
            "question": "425 students are divided equally into 17 groups. How many students are in each group?",
            "expected": "25",
            "category": "division",
            "difficulty": "word_problem"
        },
        {
            "question": "What is 720 ÷ 15?",
            "expected": "48",
            "category": "division",
            "difficulty": "basic"
        },
        
        # Mixed operations (4 problems)
        {
            "question": "John buys 3 packs of pens. Each pack has 12 pens. He then gives away 7 pens. How many pens does he have left?",
            "expected": "29",
            "category": "mixed_operations",
            "difficulty": "multi_step"
        },
        {
            "question": "A store has 240 apples. They sell 156 apples and then receive a new shipment of 89 apples. How many apples do they have now?",
            "expected": "173",
            "category": "mixed_operations",
            "difficulty": "multi_step"
        },
        {
            "question": "Maria saves $25 per week. After 8 weeks, she spends $120 on a gift. How much money does she have left?",
            "expected": "80",
            "category": "mixed_operations",
            "difficulty": "multi_step"
        },
        {
            "question": "A factory produces 150 toys per day. After 5 days, they ship out 600 toys. How many toys remain in the factory?",
            "expected": "150",
            "category": "mixed_operations",
            "difficulty": "multi_step"
        }
    ]
    
    return problems

def extract_numerical_answer(response: str) -> str:
    """Extract numerical answer from model response"""
    import re
    
    # Look for patterns like "answer is 123", "= 123", "123.", etc.
    patterns = [
        r'(?:answer is|answer:|equals?|=)\s*(\d+)',
        r'(\d+)\s*(?:is the answer|is correct)',
        r'(?:total|result|sum)\s*(?:is|=)\s*(\d+)',
        r'(?:^|\s)(\d+)(?:\s*$|\s*\.)',  # Number at end or followed by period
        r'(\d+)(?:\s+(?:toys|apples|students|books|pencils|marbles|stickers|items|pizzas|pens|dollars?))?\.?\s*$'
    ]
    
    response_lower = response.lower().strip()
    
    for pattern in patterns:
        matches = re.findall(pattern, response_lower, re.IGNORECASE)
        if matches:
            return matches[-1]  # Return the last match
    
    # Fallback: find all numbers and return the last one
    numbers = re.findall(r'\b\d+\b', response)
    if numbers:
        return numbers[-1]
    
    return "NO_ANSWER_FOUND"

def evaluate_model_balanced(model_path: str, problems: List[Dict]) -> Dict[str, Any]:
    """Evaluate a model on balanced test problems"""
    
    print(f"📊 Evaluating model: {model_path}")
    print("-" * 50)
    
    # Load model and tokenizer
    try:
        if model_path.startswith("./") or "/" not in model_path:
            # Local model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Base model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
    except Exception as e:
        print(f"❌ Error loading model {model_path}: {e}")
        return {"error": str(e)}
    
    results = {
        "model_path": model_path,
        "total_problems": len(problems),
        "correct": 0,
        "accuracy": 0.0,
        "category_results": {},
        "responses": [],
        "avg_time": 0.0
    }
    
    total_time = 0.0
    category_stats = {}
    
    for i, problem in enumerate(problems):
        print(f"Problem {i+1}/{len(problems)}: {problem['category']}")
        
        # Prepare prompt
        prompt = f"Solve this math problem step by step:\n\n{problem['question']}\n\nSolution:"
        
        # Generate response
        start_time = time.time()
        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
        except Exception as e:
            response = f"ERROR: {e}"
        
        generation_time = time.time() - start_time
        total_time += generation_time
        
        # Extract answer
        extracted = extract_numerical_answer(response)
        expected = problem['expected']
        is_correct = extracted == expected
        
        if is_correct:
            results["correct"] += 1
        
        # Track category performance
        category = problem['category']
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}
        
        category_stats[category]["total"] += 1
        if is_correct:
            category_stats[category]["correct"] += 1
        
        # Store detailed result
        results["responses"].append({
            "question": problem['question'],
            "expected": expected,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "extracted": extracted,
            "correct": is_correct,
            "category": category,
            "difficulty": problem['difficulty'],
            "time": generation_time
        })
        
        status = "✅" if is_correct else "❌"
        print(f"  {status} Expected: {expected}, Got: {extracted}")
    
    # Calculate final metrics
    results["accuracy"] = (results["correct"] / results["total_problems"]) * 100
    results["avg_time"] = total_time / len(problems)
    
    # Calculate category accuracies
    for category, stats in category_stats.items():
        accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        results["category_results"][category] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": accuracy
        }
    
    return results

def compare_models():
    """Compare multiple models on balanced evaluation"""
    
    print("🔬 Balanced Model Comparison")
    print("=" * 60)
    
    # Models to compare
    models = {
        "Base Model": "Qwen/Qwen2-0.5B-Instruct",
        "Stage 1 (Original)": "./grpo_stage1/final_model",
        "Stage 2 (Original)": "./grpo_stage2/final_model",
        "Stage 3 (Original)": "./grpo_stage3/final_model",
    }
    
    # Add balanced models if they exist
    if os.path.exists("./grpo_balanced/final_model"):
        models["Balanced Single"] = "./grpo_balanced/final_model"
    
    for i in range(1, 4):
        balanced_path = f"./grpo_balanced_stage{i}/final_model"
        if os.path.exists(balanced_path):
            models[f"Balanced Stage {i}"] = balanced_path
    
    # Create test problems
    problems = create_balanced_test_problems()
    print(f"📋 Testing on {len(problems)} balanced problems:")
    
    category_counts = {}
    for problem in problems:
        cat = problem['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for category, count in category_counts.items():
        print(f"   {category:15}: {count} problems")
    
    print()
    
    # Evaluate each model
    all_results = {}
    
    for name, path in models.items():
        if os.path.exists(path) or path.startswith("Qwen/"):
            print(f"\n🔍 Evaluating {name}...")
            results = evaluate_model_balanced(path, problems)
            
            if "error" not in results:
                all_results[name] = results
                print(f"   Overall Accuracy: {results['accuracy']:.1f}%")
                print(f"   Average Time: {results['avg_time']:.2f}s")
        else:
            print(f"⚠️  Model not found: {path}")
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("📊 BALANCED EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"{'Model':<20} {'Accuracy':<10} {'Add':<6} {'Sub':<6} {'Mul':<6} {'Div':<6} {'Mix':<6} {'Time':<8}")
    print("-" * 80)
    
    for name, results in all_results.items():
        cats = results['category_results']
        add_acc = cats.get('addition', {}).get('accuracy', 0)
        sub_acc = cats.get('subtraction', {}).get('accuracy', 0)
        mul_acc = cats.get('multiplication', {}).get('accuracy', 0)
        div_acc = cats.get('division', {}).get('accuracy', 0)
        mix_acc = cats.get('mixed_operations', {}).get('accuracy', 0)
        
        print(f"{name:<20} {results['accuracy']:>6.1f}%   {add_acc:>4.0f}% {sub_acc:>4.0f}% {mul_acc:>4.0f}% {div_acc:>4.0f}% {mix_acc:>4.0f}% {results['avg_time']:>6.2f}s")
    
    # Save detailed results
    output_file = "balanced_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return all_results

if __name__ == "__main__":
    compare_models()
    
    print("\n✅ Balanced evaluation complete!")
    print("\n💡 Key insights:")
    print("1. Check if balanced training improved division accuracy")
    print("2. Look for more consistent performance across all operations")
    print("3. Compare with previous unbalanced results")
    print("4. Identify remaining areas for improvement")
