#!/usr/bin/env python3
"""
Progressive GRPO Training Script
Implements curriculum learning with increasing difficulty
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.grpo_trainer import CPUGRPOTrainer, CPUGRPOConfig


def progressive_training():
    """Implement progressive curriculum learning for GRPO"""
    
    print("🎓 Starting Progressive GRPO Training")
    print("=" * 60)
    
    # Define curriculum stages
    curriculum = [
        {
            "name": "Stage 1: Basic Math",
            "samples": 20,
            "epochs": 2.0,
            "lr": 1e-5,
            "output": "./grpo_stage1"
        },
        {
            "name": "Stage 2: Intermediate Math", 
            "samples": 40,
            "epochs": 2.0,
            "lr": 5e-6,  # Lower LR for fine-tuning
            "output": "./grpo_stage2"
        },
        {
            "name": "Stage 3: Advanced Math",
            "samples": 80,
            "epochs": 1.0,
            "lr": 2e-6,  # Even lower LR
            "output": "./grpo_stage3"
        }
    ]
    
    previous_model = "Qwen/Qwen2-0.5B-Instruct"  # Start with base model
    results = []
    
    for i, stage in enumerate(curriculum):
        print(f"\n🚀 {stage['name']}")
        print(f"📊 Samples: {stage['samples']}, Epochs: {stage['epochs']}, LR: {stage['lr']}")
        print(f"🔄 Starting from: {previous_model}")
        print("-" * 40)
        
        # Setup configuration for this stage
        config = CPUGRPOConfig()
        config.model_name = previous_model
        config.learning_rate = stage['lr']
        config.num_train_epochs = stage['epochs']
        config.output_dir = stage['output']
        
        # Create output directory
        os.makedirs(stage['output'], exist_ok=True)
        
        try:
            # Initialize trainer
            trainer = CPUGRPOTrainer(config)
            
            # Prepare dataset
            dataset = trainer.prepare_dataset("gsm8k", num_samples=stage['samples'])
            
            # Create reward function
            reward_fn = trainer.create_reward_function("math")
            
            # Train this stage
            start_time = datetime.now()
            model_path = trainer.train(dataset, reward_fn)
            end_time = datetime.now()
            
            training_time = (end_time - start_time).total_seconds()
            
            # Test the model
            test_prompts = [
                "What is 15 + 27?",
                "Solve: 3x + 5 = 20",
                "If a box contains 24 apples and you eat 1/3 of them, how many are left?"
            ]
            
            test_results = []
            for prompt in test_prompts:
                response = trainer.generate_response(prompt)
                test_results.append({'prompt': prompt, 'response': response})
                print(f"\nTest: {prompt}")
                print(f"Response: {response[:100]}...")
            
            # Save stage results
            stage_result = {
                'stage': i + 1,
                'name': stage['name'],
                'config': stage,
                'training_time': training_time,
                'model_path': model_path,
                'test_results': test_results
            }
            results.append(stage_result)
            
            # Update for next stage
            previous_model = os.path.join(stage['output'], "final_model")
            
            print(f"✅ {stage['name']} completed in {training_time:.2f} seconds!")
            
        except Exception as e:
            print(f"❌ {stage['name']} failed: {str(e)}")
            break
    
    # Save overall results
    final_results = {
        'curriculum_completed': datetime.now().isoformat(),
        'stages_completed': len(results),
        'total_stages': len(curriculum),
        'results': results
    }
    
    with open('progressive_training_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n🎉 Progressive training completed!")
    print(f"📊 Completed {len(results)}/{len(curriculum)} stages")
    print(f"💾 Final model: {previous_model}")
    print(f"📄 Results saved to: progressive_training_results.json")
    
    return previous_model


if __name__ == "__main__":
    progressive_training()
