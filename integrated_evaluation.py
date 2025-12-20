#!/usr/bin/env python3
"""
Integrated Defense Evaluation with Real Attacks

This script:
1. Generates adversarial examples using actual HardCom attack algorithms
2. Evaluates the compression-aware defense against these attacks
3. Compares with baseline defenses (random edit, perplexity)
4. Reports metrics comparable to the CompressionAttack paper
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm

from compression_aware_defense import (
    CompressionAwareDefense,
    HardPromptCompressor,
    DefenseConfig,
    DefenseResult,
    DefenseVerdict,
    RandomEditDefense,
    PerplexityDefense,
)
from ppl_calculator import PPLCalculator
from stealth_calculator import StealthCalculator
from token_level_attack import HardComTokenAttack, AttackConfig


@dataclass
class EvaluationExample:
    """Single evaluation example."""
    id: str
    original_text: str
    adversarial_text: str
    is_poisoned: bool
    attack_type: str
    stealth_score: float


class IntegratedEvaluator:
    """
    Evaluator that generates real attacks and tests defenses.
    """
    
    def __init__(self):
        # Initialize PPL calculator
        self.ppl_calc = PPLCalculator()
        self.stealth_calc = StealthCalculator(sentence_model=None)  # Use fallback
        
        # Initialize attacker
        self.token_attacker = HardComTokenAttack()
        
        # Initialize compressor for defense
        self.compressor = HardPromptCompressor(self.ppl_calc)
        
        # Initialize defenses
        self.defenses = {
            "compression_aware": CompressionAwareDefense(
                self.compressor,
                DefenseConfig(num_variants=15, compression_rate=0.5)
            ),
            "random_edit": RandomEditDefense(self.compressor, num_variants=10),
            "perplexity": PerplexityDefense(self.ppl_calc, threshold=30.0),
        }
    
    def generate_attack(
        self, 
        text: str, 
        attack_mode: str = "degradation",
        num_words: int = 5
    ) -> Tuple[str, float]:
        """
        Generate adversarial example using HardCom token-level attack.
        
        Returns:
            Tuple of (adversarial_text, stealth_score)
        """
        config = AttackConfig(
            attack_mode=attack_mode,
            stealth_threshold=0.7,  # Allow more perturbations for testing
            ppl_margin=3.0,
            max_iterations=15
        )
        
        adversarial = self.token_attacker.attack_context(
            text,
            self.ppl_calc,
            self.stealth_calc,
            config,
            num_words=num_words
        )
        
        # Calculate stealth score
        stealth = self.stealth_calc.calculate_stealthiness(
            text, adversarial, method="token"
        )
        
        return adversarial, stealth
    
    def create_test_dataset(self) -> List[EvaluationExample]:
        """
        Create test dataset with clean and adversarial examples.
        """
        # Clean texts (product descriptions and QA contexts)
        clean_texts = [
            # Product descriptions
            "iPhone 16 Pro features a sleek titanium design with advanced camera capabilities and powerful performance",
            "Samsung Galaxy S24 Ultra delivers stunning display quality with exceptional battery life and camera system",
            "MacBook Pro offers incredible performance with the latest chip technology and beautiful retina display",
            "Google Pixel 8 Pro provides the best camera experience with advanced AI features and clean software",
            "Sony WH-1000XM5 headphones deliver industry leading noise cancellation with premium sound quality",
            "Dell XPS 15 laptop combines powerful hardware with a stunning display in a portable design",
            "Apple Watch Ultra provides rugged durability with advanced health and fitness tracking features",
            "Bose QuietComfort headphones offer exceptional comfort with excellent noise cancellation technology",
            "Microsoft Surface Pro delivers versatile performance as both a tablet and laptop computer",
            "LG OLED TV provides stunning picture quality with perfect blacks and vibrant colors",
            # QA contexts  
            "The Amazon rainforest is the largest tropical rainforest covering over five million square kilometers",
            "Mount Everest stands at 8848 meters making it the highest peak on Earth above sea level",
            "Photosynthesis converts sunlight into chemical energy releasing oxygen as a byproduct of the process",
            "The speed of light travels at approximately 300000 kilometers per second in a vacuum",
            "Water boils at 100 degrees Celsius under standard atmospheric pressure at sea level",
        ]
        
        examples = []
        
        # Add clean examples
        for i, text in enumerate(clean_texts):
            examples.append(EvaluationExample(
                id=f"clean_{i}",
                original_text=text,
                adversarial_text=text,  # No attack
                is_poisoned=False,
                attack_type="none",
                stealth_score=1.0
            ))
        
        # Generate adversarial examples
        print("Generating adversarial examples...")
        for i, text in enumerate(tqdm(clean_texts)):
            # Attack with different intensities
            for num_words in [3, 5, 7]:
                adv_text, stealth = self.generate_attack(
                    text, 
                    attack_mode="degradation",
                    num_words=num_words
                )
                
                # Only include if attack actually changed the text
                if adv_text != text:
                    examples.append(EvaluationExample(
                        id=f"adv_{i}_w{num_words}",
                        original_text=text,
                        adversarial_text=adv_text,
                        is_poisoned=True,
                        attack_type=f"token_w{num_words}",
                        stealth_score=stealth
                    ))
        
        return examples
    
    def evaluate_defense(
        self, 
        defense_name: str,
        examples: List[EvaluationExample]
    ) -> Dict:
        """Evaluate a single defense on all examples."""
        results = []
        
        for ex in examples:
            # Use adversarial text (which equals original for clean examples)
            text = ex.adversarial_text
            
            # Run defense
            defense = self.defenses[defense_name]
            result = defense.detect(text)
            
            # Determine correctness
            predicted_poisoned = result.verdict == DefenseVerdict.POISONED
            correct = predicted_poisoned == ex.is_poisoned
            
            results.append({
                "id": ex.id,
                "is_poisoned": ex.is_poisoned,
                "attack_type": ex.attack_type,
                "predicted_poisoned": predicted_poisoned,
                "correct": correct,
                "confidence": result.confidence,
                "consistency": result.consistency_score,
                "suspicious_tokens": result.suspicious_tokens,
                "stealth_score": ex.stealth_score
            })
        
        return self._compute_metrics(results)
    
    def _compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute evaluation metrics."""
        tp = sum(1 for r in results if r["is_poisoned"] and r["predicted_poisoned"])
        fp = sum(1 for r in results if not r["is_poisoned"] and r["predicted_poisoned"])
        tn = sum(1 for r in results if not r["is_poisoned"] and not r["predicted_poisoned"])
        fn = sum(1 for r in results if r["is_poisoned"] and not r["predicted_poisoned"])
        
        total = len(results)
        accuracy = (tp + tn) / total if total > 0 else 0
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / Detection Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
        
        # Detection rate by attack intensity
        by_attack = {}
        for r in results:
            if r["is_poisoned"]:
                atype = r["attack_type"]
                if atype not in by_attack:
                    by_attack[atype] = {"total": 0, "detected": 0}
                by_attack[atype]["total"] += 1
                if r["predicted_poisoned"]:
                    by_attack[atype]["detected"] += 1
        
        return {
            "total": total,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": tpr,
            "f1_score": f1,
            "tpr": tpr,
            "fpr": fpr,
            "detection_by_attack": {
                k: v["detected"] / v["total"] if v["total"] > 0 else 0
                for k, v in by_attack.items()
            },
            "per_example": results
        }
    
    def run_evaluation(self) -> Dict:
        """Run full evaluation."""
        print("=" * 80)
        print("INTEGRATED DEFENSE EVALUATION WITH REAL ATTACKS")
        print("=" * 80)
        
        # Create dataset
        print("\n[1] Creating test dataset...")
        examples = self.create_test_dataset()
        
        clean_count = sum(1 for e in examples if not e.is_poisoned)
        adv_count = sum(1 for e in examples if e.is_poisoned)
        print(f"    Created {len(examples)} examples:")
        print(f"    - Clean: {clean_count}")
        print(f"    - Adversarial: {adv_count}")
        
        # Show some examples
        print("\n    Sample adversarial examples:")
        for ex in [e for e in examples if e.is_poisoned][:3]:
            print(f"    Original:    {ex.original_text[:50]}...")
            print(f"    Adversarial: {ex.adversarial_text[:50]}...")
            print(f"    Stealth: {ex.stealth_score:.3f}")
            print()
        
        # Evaluate each defense
        print("[2] Evaluating defenses...")
        all_results = {}
        
        for defense_name in self.defenses:
            print(f"\n    Testing {defense_name}...")
            metrics = self.evaluate_defense(defense_name, examples)
            all_results[defense_name] = metrics
            print(f"    TPR: {metrics['tpr']:.3f}, FPR: {metrics['fpr']:.3f}, F1: {metrics['f1_score']:.3f}")
        
        return all_results
    
    def print_results(self, results: Dict):
        """Print formatted results."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        
        # Summary table
        print(f"\n{'Defense':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}")
        print("-" * 80)
        
        for name, m in results.items():
            print(f"{name:<25} {m['accuracy']:>10.3f} {m['precision']:>10.3f} "
                  f"{m['recall']:>10.3f} {m['f1_score']:>10.3f} {m['fpr']:>10.3f}")
        
        print("-" * 80)
        
        # Detailed breakdown
        print("\nDetection Breakdown:")
        for name, m in results.items():
            print(f"\n  {name}:")
            print(f"    TP: {m['tp']}, FP: {m['fp']}, TN: {m['tn']}, FN: {m['fn']}")
            print(f"    Detection Rate (TPR): {m['tpr']:.1%}")
            print(f"    False Positive Rate: {m['fpr']:.1%}")
            
            if m['detection_by_attack']:
                print("    By attack intensity:")
                for atype, rate in m['detection_by_attack'].items():
                    print(f"      {atype}: {rate:.1%}")
        
        # Comparison to paper
        print("\n" + "=" * 80)
        print("COMPARISON TO PAPER RESULTS")
        print("=" * 80)
        print("\nPaper's Defense Detection Rates (Section 6, Tables 10-12):")
        print("  PPL-based detection:        < 5% (often 0%)")
        print("  Prevention-based (StruQ):   ~10%")
        print("  LLM-assisted detection:     < 10%")
        print("\nOur Results:")
        
        ca_tpr = results.get("compression_aware", {}).get("tpr", 0)
        ca_fpr = results.get("compression_aware", {}).get("fpr", 0)
        
        print(f"  Compression-Aware Defense:")
        print(f"    Detection Rate (TPR): {ca_tpr:.1%}")
        print(f"    False Positive Rate:  {ca_fpr:.1%}")
        
        if ca_tpr > 0.10:
            improvement = ca_tpr / 0.10
            print(f"\n  ✓ {improvement:.1f}x improvement over paper's best baseline (~10%)")
        else:
            print(f"\n  ✗ Detection rate needs improvement")
        
        print("=" * 80)


def main():
    """Main entry point."""
    evaluator = IntegratedEvaluator()
    results = evaluator.run_evaluation()
    evaluator.print_results(results)
    
    # Save results
    output_file = "integrated_evaluation_results.json"
    
    # Convert to JSON-serializable format
    json_results = {}
    for name, data in results.items():
        json_results[name] = {
            k: v for k, v in data.items() 
            if k != "per_example"  # Exclude detailed per-example data
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()