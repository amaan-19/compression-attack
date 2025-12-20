#!/usr/bin/env python3
"""
Render QA attack results in a readable format.
Converts JSON results to formatted tables and summaries.
"""

import json
import sys
from typing import Dict, List


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 80}")
    print(text.center(80))
    print(f"{char * 80}\n")


def print_summary_table(results: List[Dict]):
    """Print summary table of all configurations."""
    print_header("SUMMARY - ATTACK SUCCESS RATES", "=")
    
    print(f"{'Configuration':<30} {'ASR':>10} {'Successes':>12} {'Total':>8}")
    print("-" * 80)
    
    for result in results:
        config_name = result['config_name']
        asr = result['success_rate']
        successes = result['successful_attacks']
        total = result['total_examples']
        
        print(f"{config_name:<30} {asr:>10.3f} {successes:>12} {total:>8}")
    
    print("-" * 80)


def print_detailed_results(results: List[Dict], show_all: bool = False):
    """Print detailed results for each configuration."""
    for config_result in results:
        print_header(f"CONFIGURATION: {config_result['config_name']}", "-")
        
        # Print config parameters
        config = config_result['config']
        print(f"Compression Rate: {config['compression_rate']}")
        print(f"Words Attacked: {config['num_words_to_attack']}")
        print(f"Attack Strategy: {config['attack_strategy']}")
        print(f"\nASR: {config_result['success_rate']:.3f} ({config_result['successful_attacks']}/{config_result['total_examples']})")
        
        # Count successes and failures
        successes = [ex for ex in config_result['examples'] if ex.get('success', False)]
        failures = [ex for ex in config_result['examples'] if not ex.get('success', False)]
        
        print(f"\n✓ Successful attacks: {len(successes)}")
        print(f"✗ Failed attacks: {len(failures)}")
        
        # Show examples
        if show_all:
            print("\n" + "=" * 80)
            print("ALL EXAMPLES:")
            print("=" * 80)
            for i, ex in enumerate(config_result['examples'], 1):
                print_example(ex, i)
        else:
            # Show first 3 successes and first 3 failures
            print("\n" + "=" * 80)
            print("SAMPLE SUCCESSFUL ATTACKS (first 3):")
            print("=" * 80)
            for i, ex in enumerate(successes[:3], 1):
                print_example(ex, i, show_full=True)
            
            if failures:
                print("\n" + "=" * 80)
                print("SAMPLE FAILED ATTACKS (first 3):")
                print("=" * 80)
                for i, ex in enumerate(failures[:3], 1):
                    print_example(ex, i, show_full=True)


def print_example(example: Dict, num: int, show_full: bool = False):
    """Print a single example result."""
    status = "✓ SUCCESS" if example.get('success', False) else "✗ FAILED"
    
    print(f"\n[{num}] {status}")
    print(f"ID: {example.get('id', 'N/A')}")
    print(f"Title: {example.get('title', 'N/A')}")
    
    if 'error' in example:
        print(f"ERROR: {example['error']}")
        return
    
    print(f"\nQuestion: {example.get('question', 'N/A')}")
    print(f"Correct Answer: {example.get('correct_answer', 'N/A')}")
    print(f"Original Answer: {example.get('original_answer', 'N/A')}")
    print(f"Adversarial Answer: {example.get('adversarial_answer', 'N/A')}")
    
    if show_full:
        print("-" * 80)


def compare_to_paper(results: List[Dict]):
    """Compare results to paper's reported performance."""
    print_header("COMPARISON TO PAPER", "=")
    
    print("Our Results:")
    print("-" * 40)
    for result in results:
        print(f"  {result['config_name']:<30} ASR = {result['success_rate']:.3f}")
    
    print("\n" + "=" * 80)
    print("Paper's Results (Table 1, QA Task):")
    print("-" * 40)
    print("  HardCom (token-level)              ASR = 0.680 (average)")
    print("  HardCom (word-level, Sel. Context) ASR = 0.800")
    print("  HardCom (word-level, LLMLingua)    ASR = 0.620")
    print("  SoftCom (average)                  ASR = 0.870")
    
    print("\n" + "=" * 80)
    print("Analysis:")
    print("-" * 40)
    
    # Find best config
    best = max(results, key=lambda x: x['success_rate'])
    print(f"  Best configuration: {best['config_name']}")
    print(f"  Best ASR: {best['success_rate']:.3f}")
    
    # Compare to paper's hybrid approach
    paper_avg = 0.68  # Token-level baseline
    if best['success_rate'] >= paper_avg:
        print(f"  ✓ Exceeds paper's token-level baseline (0.680)")
    else:
        diff = paper_avg - best['success_rate']
        print(f"  ✗ Below paper's baseline by {diff:.3f}")


def export_csv(results: List[Dict], output_file: str = "results_summary.csv"):
    """Export summary to CSV."""
    with open(output_file, 'w') as f:
        # Header
        f.write("Config,Compression_Rate,Words_Attacked,ASR,Successes,Total\n")
        
        # Data
        for result in results:
            config = result['config']
            f.write(f"{result['config_name']},")
            f.write(f"{config['compression_rate']},")
            f.write(f"{config['num_words_to_attack']},")
            f.write(f"{result['success_rate']:.3f},")
            f.write(f"{result['successful_attacks']},")
            f.write(f"{result['total_examples']}\n")
    
    print(f"\n✓ Summary exported to {output_file}")


def main():
    """Main function to render results."""
    # Get JSON file from command line or use default
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        json_file = sys.argv[1]
    else:
        json_file = "squad_qa_attack_results.json"
    
    # Check for flags
    show_all = "--all" in sys.argv
    export = "--csv" in sys.argv
    
    # Check for config filter
    config_filter = None
    for arg in sys.argv:
        if arg.startswith("--config="):
            config_filter = arg.split("=")[1].lower()
    
    # Load results
    try:
        with open(json_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found")
        print("\nUsage: python view_results.py [results.json] [options]")
        print("\nOptions:")
        print("  --all                Show all examples (default: show samples)")
        print("  --csv                Export summary to CSV")
        print("  --config=NAME        Filter by configuration name")
        print("\nConfig filters:")
        print("  --config=moderate    Show only 'Moderate Compression'")
        print("  --config=aggressive  Show only 'Aggressive Compression'")
        print("  --config=very        Show only 'Very Aggressive Compression'")
        return
    
    # Apply config filter if specified
    if config_filter:
        original_count = len(results)
        results = [r for r in results if config_filter in r['config_name'].lower()]
        if not results:
            print(f"Error: No configurations match filter '{config_filter}'")
            print("\nAvailable configurations:")
            with open(json_file, 'r') as f:
                all_results = json.load(f)
            for r in all_results:
                print(f"  - {r['config_name']}")
            return
        print(f"Filtered: Showing {len(results)} of {original_count} configurations\n")
    
    print_header("QA ATTACK RESULTS VIEWER", "=")
    print(f"File: {json_file}")
    print(f"Configurations: {len(results)}")
    
    # Print summary
    print_summary_table(results)
    
    # Print detailed results
    print_detailed_results(results, show_all=show_all)
    
    # Compare to paper (only if showing all configs)
    if not config_filter:
        compare_to_paper(results)
    
    # Export CSV if requested
    if export:
        export_csv(results)
    
    print("\n" + "=" * 80)
    if not config_filter:
        print("Options:")
        print("  --all              Show all examples")
        print("  --csv              Export summary to CSV")
        print("  --config=moderate  Filter by configuration")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()