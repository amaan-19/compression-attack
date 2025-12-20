import json
from experiments.qa_eval import QAEvaluator, QA_EXAMPLES
from typing import List, Dict
from tqdm import tqdm


def run_batch_evaluation():
    """
    Run QA attacks on 50 SQuAD examples with multiple configurations.
    """
    evaluator = QAEvaluator()
    
    # test configurations
    configs = [
        {
            "name": "Moderate Compression",
            "compression_rate": 0.5,
            "num_words_to_attack": 10,
            "attack_strategy": "hybrid"
        },
        {
            "name": "Aggressive Compression",
            "compression_rate": 0.3,
            "num_words_to_attack": 15,
            "attack_strategy": "hybrid"
        },
        {
            "name": "Very Aggressive Compression",
            "compression_rate": 0.2,
            "num_words_to_attack": 20,
            "attack_strategy": "hybrid"
        }
    ]
    
    print("="*80)
    print("BATCH QA ATTACK EVALUATION - SQUAD DATASET")
    print("="*80)
    print(f"\nTotal examples: {len(QA_EXAMPLES)} (from SQuAD)")
    print(f"Configurations to test: {len(configs)}")
    print("\n" + "="*80)
    
    all_results = []
    
    for config in configs:
        print(f"\n\n{'='*80}")
        print(f"CONFIGURATION: {config['name']}")
        print(f"{'='*80}")
        print(f"  Compression rate: {config['compression_rate']}")
        print(f"  Words to attack: {config['num_words_to_attack']}")
        print(f"  Attack strategy: {config['attack_strategy']}")
        
        config_results = {
            "config_name": config['name'],
            "config": config,
            "examples": []
        }
        
        # Use tqdm for progress bar
        for i, example in enumerate(tqdm(QA_EXAMPLES, desc=f"Processing {config['name']}")):
            try:
                result = evaluator.evaluate_qa_attack(
                    context=example["context"],
                    question=example["question"],
                    correct_answer=example["correct_answer"],
                    compression_rate=config["compression_rate"],
                    num_words_to_attack=config["num_words_to_attack"],
                    attack_strategy=config["attack_strategy"]
                )
                
                example_result = {
                    "id": example["id"],
                    "title": example["title"],
                    "question": example["question"],
                    "correct_answer": example["correct_answer"],
                    "success": result["success"],
                    "answer_changed": result["answer_changed"],
                    "original_answer": result["original_answer"],
                    "adversarial_answer": result["adversarial_answer"]
                }
                
            except Exception as e:
                print(f"\nERROR on example {example['id']}: {e}")
                example_result = {
                    "id": example["id"],
                    "title": example["title"],
                    "error": str(e)
                }
            
            config_results["examples"].append(example_result)
        
        # Calculate success rate for this config
        successes = sum(1 for ex in config_results["examples"] if ex.get("success", False))
        total = len(config_results["examples"])
        success_rate = successes / total if total > 0 else 0
        
        config_results["success_rate"] = success_rate
        config_results["total_examples"] = total
        config_results["successful_attacks"] = successes
        
        print(f"\n\n{'='*80}")
        print(f"CONFIG SUMMARY: {config['name']}")
        print(f"{'='*80}")
        print(f"Success Rate: {success_rate:.2%} ({successes}/{total})")
        print(f"Attack Success Rate (ASR): {success_rate:.3f}")
        print(f"{'='*80}")
        
        all_results.append(config_results)
    
    # Save results
    output_file = "squad_qa_attack_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison to paper
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - COMPARISON TO PAPER")
    print(f"{'='*80}")
    print("\nOur Results:")
    for result in all_results:
        print(f"  {result['config_name']}: ASR = {result['success_rate']:.3f}")
    
    print("\nPaper's Results (Table 1, QA task):")
    print("  HardCom (token-level): ASR = 0.68 (average)")
    print("  HardCom (word-level): ASR = 0.80 (Selective Context)")
    print("  SoftCom: ASR = 0.87 (average)")
    
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_batch_evaluation()