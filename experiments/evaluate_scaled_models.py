"""
Scaled Model Evaluation Pipeline
Comprehensive evaluation of scaled progressive training results
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from experiments.baseline_accuracy_validation import GSM8KEvaluator
from src.utils.model_utils import load_trained_model

class ScaledEvaluator:
    """Evaluator for scaled progressive training models"""
    
    def __init__(self):
        self.gsm8k_evaluator = GSM8KEvaluator()
        self.results = {}
    
    def evaluate_scaled_models(self):
        """Evaluate all scaled progressive models"""
        
        print("🔍 Evaluating Scaled Progressive Models")
        print("=" * 50)
        
        model_paths = [
            ("Stage 1 (100 samples)", "./models/unified_progressive/stage1_scaled"),
            ("Stage 2 (150 samples)", "./models/unified_progressive/stage2_scaled"),
            ("Stage 3 (200 samples)", "./models/unified_progressive/stage3_scaled")
        ]
        
        for stage_name, model_path in model_paths:
            if os.path.exists(model_path):
                print(f"\n📊 Evaluating {stage_name}")
                print("-" * 30)
                
                try:
                    # Load model
                    model, tokenizer = load_trained_model(model_path)
                    
                    # Run GSM8K evaluation
                    accuracy = self.gsm8k_evaluator.evaluate_model(model, tokenizer)
                    
                    self.results[stage_name] = {
                        'accuracy': accuracy,
                        'model_path': model_path,
                        'evaluation_time': time.time()
                    }
                    
                    print(f"✅ {stage_name}: {accuracy:.2%} accuracy")
                    
                except Exception as e:
                    print(f"❌ Failed to evaluate {stage_name}: {e}")
                    self.results[stage_name] = {'error': str(e)}
            else:
                print(f"⚠️  Model not found: {model_path}")
    
    def compare_with_baseline(self):
        """Compare scaled results with baseline"""
        
        print("\n📈 Comparison with Baseline Models")
        print("=" * 40)
        
        # Load baseline results if available
        baseline_path = "./experiments/results/unified_eval/performance_metrics.json"
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
            
            print("Baseline (15 samples) vs Scaled (450 samples):")
            print("-" * 45)
            
            for stage in ['Stage 1', 'Stage 2', 'Stage 3']:
                baseline_key = stage.replace(' ', '_').lower()
                scaled_key = f"{stage} (100 samples)" if stage == "Stage 1" else \
                           f"{stage} (150 samples)" if stage == "Stage 2" else \
                           f"{stage} (200 samples)"
                
                if baseline_key in baseline_data and scaled_key in self.results:
                    baseline_acc = baseline_data[baseline_key].get('accuracy', 0)
                    scaled_acc = self.results[scaled_key].get('accuracy', 0)
                    
                    improvement = scaled_acc - baseline_acc
                    print(f"{stage}:")
                    print(f"  Baseline: {baseline_acc:.2%}")
                    print(f"  Scaled:   {scaled_acc:.2%}")
                    print(f"  Change:   {improvement:+.2%}")
                    print()
    
    def analyze_scaling_effects(self):
        """Analyze the effects of scaling up sample size"""
        
        print("\n🔬 Scaling Analysis")
        print("=" * 25)
        
        sample_sizes = [100, 150, 200]  # Stage 1, 2, 3
        accuracies = []
        
        for i, size in enumerate(sample_sizes):
            stage_name = f"Stage {i+1} ({size} samples)"
            if stage_name in self.results:
                acc = self.results[stage_name].get('accuracy', 0)
                accuracies.append(acc)
                print(f"{size:3d} samples → {acc:.2%} accuracy")
        
        if len(accuracies) >= 2:
            trend = "📈 Improving" if accuracies[-1] > accuracies[0] else \
                   "📉 Declining" if accuracies[-1] < accuracies[0] else \
                   "➡️  Stable"
            print(f"\nTrend: {trend}")
    
    def save_results(self):
        """Save evaluation results"""
        
        output_dir = "./experiments/results/scaled_evaluation"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        with open(f"{output_dir}/evaluation_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary report
        summary = {
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_models_evaluated': len(self.results),
            'scaling_factor': '30x increase in training data',
            'key_findings': self._generate_key_findings()
        }
        
        with open(f"{output_dir}/summary_report.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_dir}/")
    
    def _generate_key_findings(self):
        """Generate key findings from evaluation"""
        findings = []
        
        if self.results:
            accuracies = [r.get('accuracy', 0) for r in self.results.values() if 'accuracy' in r]
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                findings.append(f"Average accuracy across scaled models: {avg_accuracy:.2%}")
                findings.append(f"Best performing stage: {max(accuracies):.2%}")
                findings.append(f"Performance range: {min(accuracies):.2%} - {max(accuracies):.2%}")
        
        return findings

def main():
    """Run scaled model evaluation"""
    
    evaluator = ScaledEvaluator()
    
    # Run evaluation
    evaluator.evaluate_scaled_models()
    evaluator.compare_with_baseline()
    evaluator.analyze_scaling_effects()
    evaluator.save_results()
    
    print("\n🎯 Evaluation complete!")
    print("Check ./experiments/results/scaled_evaluation/ for detailed results")

if __name__ == "__main__":
    main()
