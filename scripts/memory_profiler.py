#!/usr/bin/env python3
"""
Memory Profiler for GRPO Training
Analyzes and profiles memory usage during training
"""

import os
import sys
import time
import argparse
import json
import tracemalloc
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project utilities
from src.utils.cpu_utils import optimize_memory


class MemoryProfiler:
    """
    Profiles memory usage in GRPO training
    
    Features:
    - Detailed memory snapshots
    - Top memory consumers identification
    - Memory leak detection
    - Optimization recommendations
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the memory profiler
        
        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_counter = 0
        self.snapshots = []
        
    def start_profiling(self):
        """Start memory profiling"""
        tracemalloc.start()
        print("Memory profiling started")
        
    def take_snapshot(self, tag: str) -> Dict:
        """
        Take a memory snapshot
        
        Args:
            tag: Label for this snapshot
            
        Returns:
            Dictionary with snapshot statistics
        """
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            print("Memory profiling started")
        
        # Take snapshot
        snapshot = tracemalloc.take_snapshot()
        self.snapshot_counter += 1
        
        # Get top stats
        top_stats = snapshot.statistics('lineno')
        
        # Calculate total
        total_size = sum(stat.size for stat in top_stats)
        
        # Format snapshot data
        snapshot_data = {
            "id": self.snapshot_counter,
            "tag": tag,
            "timestamp": datetime.datetime.now().isoformat(),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "top_consumers": [
                {
                    "file": str(stat.traceback.frame.filename),
                    "line": stat.traceback.frame.lineno,
                    "size_bytes": stat.size,
                    "size_mb": stat.size / (1024 * 1024)
                }
                for stat in top_stats[:10]  # Top 10 consumers
            ]
        }
        
        self.snapshots.append(snapshot_data)
        
        # Print summary
        print(f"\nSnapshot #{self.snapshot_counter} - {tag}")
        print(f"Total memory: {snapshot_data['total_size_mb']:.2f} MB")
        print("Top 3 memory consumers:")
        for i, consumer in enumerate(snapshot_data['top_consumers'][:3]):
            print(f"  {i+1}. {consumer['size_mb']:.2f} MB - {consumer['file']}:{consumer['line']}")
        
        return snapshot_data
    
    def compare_snapshots(self, snapshot1_id: int, snapshot2_id: int) -> Dict:
        """
        Compare two memory snapshots to identify leaks/changes
        
        Args:
            snapshot1_id: ID of the first snapshot
            snapshot2_id: ID of the second snapshot
            
        Returns:
            Dictionary with comparison results
        """
        # Find snapshots by ID
        snapshot1 = next((s for s in self.snapshots if s['id'] == snapshot1_id), None)
        snapshot2 = next((s for s in self.snapshots if s['id'] == snapshot2_id), None)
        
        if not snapshot1 or not snapshot2:
            print(f"Error: Snapshots with IDs {snapshot1_id} and {snapshot2_id} not found")
            return {}
        
        # Calculate difference
        diff_mb = snapshot2['total_size_mb'] - snapshot1['total_size_mb']
        percentage = (diff_mb / snapshot1['total_size_mb']) * 100 if snapshot1['total_size_mb'] > 0 else 0
        
        # Create comparison result
        comparison = {
            "snapshot1_id": snapshot1_id,
            "snapshot2_id": snapshot2_id,
            "snapshot1_tag": snapshot1['tag'],
            "snapshot2_tag": snapshot2['tag'],
            "memory_diff_mb": diff_mb,
            "percentage_change": percentage,
            "increased": diff_mb > 0,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Print summary
        print(f"\nMemory comparison: {snapshot1['tag']} -> {snapshot2['tag']}")
        print(f"Change: {diff_mb:.2f} MB ({percentage:.1f}%)")
        if diff_mb > 0:
            print("Memory usage increased - potential leak or expected growth")
        else:
            print("Memory usage decreased - good cleanup or normal fluctuation")
        
        return comparison
    
    def stop_profiling(self):
        """Stop memory profiling and save results"""
        if not tracemalloc.is_tracing():
            print("Memory profiling was not running")
            return
        
        # Save all snapshots
        self._save_results()
        
        # Stop profiling
        tracemalloc.stop()
        print("Memory profiling stopped")
    
    def _save_results(self):
        """Save profiling results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"memory_profile_{timestamp}.json"
        
        results = {
            "timestamp": timestamp,
            "snapshots": self.snapshots
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Memory profiling results saved to: {results_file}")
        
    def provide_recommendations(self) -> List[str]:
        """Provide memory optimization recommendations based on profiling"""
        if not self.snapshots:
            return ["No profiling data available for recommendations"]
        
        recommendations = []
        
        # Check for memory growth patterns
        if len(self.snapshots) >= 2:
            first = self.snapshots[0]
            last = self.snapshots[-1]
            
            diff_mb = last['total_size_mb'] - first['total_size_mb']
            if diff_mb > 0:
                growth_percentage = (diff_mb / first['total_size_mb']) * 100
                if growth_percentage > 50:
                    recommendations.append(
                        f"HIGH MEMORY GROWTH: {growth_percentage:.1f}% increase over the profiling session. "
                        f"Consider investigating memory leaks."
                    )
                elif growth_percentage > 20:
                    recommendations.append(
                        f"MODERATE MEMORY GROWTH: {growth_percentage:.1f}% increase. "
                        f"Monitor memory usage during longer training runs."
                    )
        
        # Check top consumers across all snapshots
        all_consumers = {}
        for snapshot in self.snapshots:
            for consumer in snapshot['top_consumers']:
                key = f"{consumer['file']}:{consumer['line']}"
                if key not in all_consumers:
                    all_consumers[key] = []
                all_consumers[key].append(consumer['size_mb'])
        
        # Find consistent top consumers
        consistent_top_consumers = {
            k: sum(v) / len(v) 
            for k, v in all_consumers.items() 
            if len(v) >= len(self.snapshots) / 2  # Appears in at least half of snapshots
        }
        
        # Sort by average memory usage
        top_consistent_consumers = sorted(
            consistent_top_consumers.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3
        
        if top_consistent_consumers:
            recommendations.append("TOP MEMORY CONSUMERS (consider optimizing):")
            for file_line, avg_mb in top_consistent_consumers:
                recommendations.append(f"  - {avg_mb:.2f} MB: {file_line}")
        
        # General recommendations
        recommendations.append("\nGENERAL RECOMMENDATIONS:")
        recommendations.append("  - Ensure garbage collection is called regularly")
        recommendations.append("  - Consider using gradient checkpointing")
        recommendations.append("  - Use dynamic quantization for model weights")
        recommendations.append("  - Reduce batch size if memory usage is consistently high")
        
        return recommendations


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Memory Profiler for GRPO Training")
    parser.add_argument("--output", type=str, default="./logs/memory_profiles",
                        help="Directory to save profiling results")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode (manual snapshots)")
    return parser.parse_args()


def interactive_profiling(profiler):
    """Run interactive profiling session"""
    print("\nMemory Profiler Interactive Mode")
    print("Commands:")
    print("  s <tag>  - Take snapshot with tag")
    print("  c <id1> <id2>  - Compare snapshots by ID")
    print("  l  - List all snapshots")
    print("  r  - Show recommendations")
    print("  q  - Quit")
    
    profiler.start_profiling()
    
    while True:
        cmd = input("\n> ").strip().split()
        if not cmd:
            continue
            
        if cmd[0] == "q":
            break
        elif cmd[0] == "s":
            tag = " ".join(cmd[1:]) if len(cmd) > 1 else f"Snapshot {len(profiler.snapshots) + 1}"
            profiler.take_snapshot(tag)
        elif cmd[0] == "c":
            if len(cmd) != 3:
                print("Usage: c <id1> <id2>")
                continue
            try:
                id1 = int(cmd[1])
                id2 = int(cmd[2])
                profiler.compare_snapshots(id1, id2)
            except ValueError:
                print("Error: Snapshot IDs must be integers")
        elif cmd[0] == "l":
            print("\nSnapshots:")
            for snapshot in profiler.snapshots:
                print(f"  {snapshot['id']}: {snapshot['tag']} - {snapshot['total_size_mb']:.2f} MB")
        elif cmd[0] == "r":
            recommendations = profiler.provide_recommendations()
            print("\nRecommendations:")
            for rec in recommendations:
                print(rec)
        else:
            print("Unknown command")
    
    profiler.stop_profiling()
    print("Interactive profiling session ended")


def main():
    """Main function"""
    args = parse_args()
    
    profiler = MemoryProfiler(args.output)
    
    if args.interactive:
        interactive_profiling(profiler)
    else:
        # Automatic profiling example
        profiler.start_profiling()
        
        # Take initial snapshot
        profiler.take_snapshot("Initial state")
        
        # Simulate memory-intensive operations
        print("\nSimulating memory-intensive operations...")
        large_list = [0] * 10000000  # Allocate ~80MB
        profiler.take_snapshot("After large allocation")
        
        # Free memory
        large_list = None
        optimize_memory()
        profiler.take_snapshot("After cleanup")
        
        # Compare snapshots
        profiler.compare_snapshots(1, 2)
        profiler.compare_snapshots(2, 3)
        
        # Print recommendations
        print("\nMemory Optimization Recommendations:")
        for recommendation in profiler.provide_recommendations():
            print(recommendation)
        
        # Stop profiling
        profiler.stop_profiling()


if __name__ == "__main__":
    main()
