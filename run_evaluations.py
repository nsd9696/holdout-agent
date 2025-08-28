#!/usr/bin/env python3
"""
Run both evaluation processes and generate a summary report for paper publication.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

def run_distributional_evaluation():
    """Run the distributional parity evaluation."""
    print("="*60)
    print("RUNNING DISTRIBUTIONAL PARITY EVALUATION")
    print("="*60)
    
    try:
        # Import and run the evaluation
        from evaluation_distributional_parity import run_distributional_parity_evaluation
        results = run_distributional_parity_evaluation()
        return results
    except Exception as e:
        print(f"Distributional evaluation failed: {e}")
        return []

def run_training_efficacy_evaluation():
    """Run the training efficacy A/B testing evaluation."""
    print("\n" + "="*60)
    print("RUNNING TRAINING EFFICACY A/B TESTING")
    print("="*60)
    
    try:
        # Import and run the evaluation
        from evaluation_training_efficacy import run_training_efficacy_evaluation
        results = run_training_efficacy_evaluation()
        return results
    except Exception as e:
        print(f"Training efficacy evaluation failed: {e}")
        return []

def generate_paper_tables(dist_results, efficacy_results):
    """Generate publication-ready tables."""
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY FOR PAPER")
    print("="*80)
    
    # Table 1: Distributional Parity Results
    if dist_results:
        print("\nTable 1: Distributional Parity Evaluation")
        print("-" * 60)
        
        dist_data = []
        for r in dist_results:
            dist_data.append({
                'Scenario': r['goal'][:40] + '...' if len(r['goal']) > 40 else r['goal'],
                'Benchmarks': ', '.join(r['benchmarks']),
                'Total Items': r['total_items'],
                'MMD Linear': f"{r['mmd_linear']:.4f}",
                'MMD RBF': f"{r['mmd_rbf']:.4f}",
                'Wasserstein': f"{r['wasserstein_distance']:.4f}",
                'Difficulty KL': f"{r['difficulty_kl_divergence']:.4f}",
                'Benchmark KL': f"{r['benchmark_kl_divergence']:.4f}"
            })
        
        df_dist = pd.DataFrame(dist_data)
        print(df_dist.to_string(index=False))
        
        # Summary statistics
        print(f"\nDistributional Parity Summary:")
        print(f"Average MMD Linear: {np.mean([r['mmd_linear'] for r in dist_results]):.4f}")
        print(f"Average MMD RBF: {np.mean([r['mmd_rbf'] for r in dist_results]):.4f}")
        print(f"Average Wasserstein: {np.mean([r['wasserstein_distance'] for r in dist_results]):.4f}")
    
    # Table 2: Training Efficacy Results
    if efficacy_results:
        print(f"\n\nTable 2: Training Efficacy A/B Testing Results")
        print("-" * 80)
        
        efficacy_data = []
        for r in efficacy_results:
            efficacy_data.append({
                'Scenario': r['scenario'],
                'Method': r['split_type'],
                'Overall Score': f"{r['overall_score']:.4f}",
                'Embedding Div.': f"{r['held_in_diversity']['embedding_diversity']:.4f}",
                'Topic Div.': f"{r['held_in_diversity']['topic_diversity']:.4f}",
                'Benchmark Div.': f"{r['held_in_diversity']['benchmark_diversity']:.4f}",
                'Difficulty Bal.': f"{r['held_in_difficulty_balance']:.4f}",
                'Benchmark Cov.': f"{r['held_in_coverage']['benchmark_coverage']:.4f}"
            })
        
        df_efficacy = pd.DataFrame(efficacy_data)
        print(df_efficacy.to_string(index=False))
        
        # Method comparison
        from collections import defaultdict
        method_scores = defaultdict(list)
        for r in efficacy_results:
            method_scores[r['split_type']].append(r['overall_score'])
        
        print(f"\n\nMethod Comparison (Average Overall Scores):")
        print("-" * 40)
        for method, scores in method_scores.items():
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{method:12s}: {avg_score:.4f} ± {std_score:.4f}")
        
        # Winner
        if method_scores:
            best_method = max(method_scores.keys(), key=lambda x: np.mean(method_scores[x]))
            print(f"\n🏆 Best Method: {best_method}")
            print(f"   Average Score: {np.mean(method_scores[best_method]):.4f}")

def save_results(dist_results, efficacy_results):
    """Save all results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual results
    if dist_results:
        with open(f'distributional_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(dist_results, f, indent=2)
    
    if efficacy_results:
        with open(f'training_efficacy_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(efficacy_results, f, indent=2)
    
    # Save combined summary
    summary = {
        'timestamp': timestamp,
        'distributional_parity': dist_results,
        'training_efficacy': efficacy_results,
        'summary_stats': {}
    }
    
    if dist_results:
        summary['summary_stats']['avg_mmd_linear'] = np.mean([r['mmd_linear'] for r in dist_results])
        summary['summary_stats']['avg_mmd_rbf'] = np.mean([r['mmd_rbf'] for r in dist_results])
        summary['summary_stats']['avg_wasserstein'] = np.mean([r['wasserstein_distance'] for r in dist_results])
    
    if efficacy_results:
        from collections import defaultdict
        method_scores = defaultdict(list)
        for r in efficacy_results:
            method_scores[r['split_type']].append(r['overall_score'])
        
        summary['summary_stats']['method_averages'] = {
            method: np.mean(scores) for method, scores in method_scores.items()
        }
        
        if method_scores:
            best_method = max(method_scores.keys(), key=lambda x: np.mean(method_scores[x]))
            summary['summary_stats']['best_method'] = best_method
            summary['summary_stats']['best_score'] = np.mean(method_scores[best_method])
    
    with open(f'evaluation_summary_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📁 Results saved:")
    print(f"   - evaluation_summary_{timestamp}.json")
    if dist_results:
        print(f"   - distributional_results_{timestamp}.json")
    if efficacy_results:
        print(f"   - training_efficacy_results_{timestamp}.json")

def main():
    """Main evaluation runner."""
    print("🚀 Starting Holdout Agent Evaluation Suite")
    print(f"📅 Timestamp: {datetime.now()}")
    
    # Check prerequisites
    try:
        from es_index import get_es_client
        es = get_es_client()
        # Test connection without requiring data
        print("✅ Elasticsearch connection available")
    except Exception as e:
        print(f"⚠️  Elasticsearch connection issue: {e}")
        print("   Some evaluations may fail without proper ES setup")
    
    # Run evaluations
    dist_results = run_distributional_evaluation()
    efficacy_results = run_training_efficacy_evaluation()
    
    # Generate summary
    generate_paper_tables(dist_results, efficacy_results)
    
    # Save results
    save_results(dist_results, efficacy_results)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Distributional tests: {len(dist_results)} scenarios")
    print(f"   Efficacy tests: {len(efficacy_results)} results")

if __name__ == "__main__":
    main()
