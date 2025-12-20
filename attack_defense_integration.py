#!/usr/bin/env python3
"""
Integration Module: Attack + Defense Pipeline

This module provides end-to-end integration between:
1. HardCom attacks (token-level and word-level)
2. Compression-aware counterfactual defense
3. Baseline defenses for comparison

Allows testing the defense against actual attack outputs.
"""

import json
import os
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

# Try to import attack modules (may need path adjustment)
try:
    from token_level_attack import HardComTokenAttack, AttackConfig as TokenAttackConfig
    from word_level_attack import HardComWordAttack, AttackConfig as WordAttackConfig
    from ppl_calculator import PPLCalculator
    from stealth_calculator import StealthCalculator
    ATTACKS_AVAILABLE = True
except ImportError:
    ATTACKS_AVAILABLE = False
    print("Warning: Attack modules not found. Using synthetic adversarial examples.")

from compression_aware_defense import (
    CompressionAwareDefense,
    PPLBasedCompressor,
    DefenseConfig,
    DefenseResult,
    DefenseVerdict,
    RandomEditDefense,
    PerplexityDefense,
)


@dataclass
class AttackDefenseResult:
    """Result from attack-defense evaluation."""
    example_id: str
    original_text: str
    adversarial_text: str
    attack_type: str
    attack_success: bool  # Did attack change model preference?
    defense_verdict: str
    defense_confidence: float
    defense_consistency: float
    defense_detected: bool  # Did defense catch the attack?
    stealth_score: float


class AttackDefensePipeline:
    """
    End-to-end pipeline for evaluating defense against attacks.
    """
    
    def __init__(self, model=None, tokenizer=None):
        """Initialize pipeline with optional model for PPL calculation."""
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize compressor
        self.compressor = PPLBasedCompressor(model, tokenizer)
        
        # Initialize defense
        self.defense_config = DefenseConfig(
            num_variants=10,
            consistency_threshold=0.7,
            compression_rate=0.5
        )
        self.defense = CompressionAwareDefense(self.compressor, self.defense_config)
        
        # Initialize baselines
        self.random_defense = RandomEditDefense(self.compressor, num_variants=10)
        self.ppl_defense = PerplexityDefense(model, tokenizer)
        
        # Initialize attackers if available
        if ATTACKS_AVAILABLE and model is not None:
            self.ppl_calc = PPLCalculator(model, tokenizer)
            self.stealth_calc = StealthCalculator()
            self.token_attacker = HardComTokenAttack()
            self.word_attacker = HardComWordAttack()
        else:
            self.ppl_calc = None
            self.stealth_calc = None
            self.token_attacker = None
            self.word_attacker = None
    
    def generate_adversarial(
        self,
        text: str,
        attack_type: str = "token",
        num_words: int = 5
    ) -> Tuple[str, float]:
        """
        Generate adversarial example using attack modules.
        
        Returns:
            Tuple of (adversarial_text, stealth_score)
        """
        if not ATTACKS_AVAILABLE or self.token_attacker is None:
            # Fallback: synthetic perturbation
            return self._synthetic_perturbation(text, attack_type)
        
        config = TokenAttackConfig(
            attack_mode="degradation",
            stealth_threshold=0.8,
            ppl_margin=2.0
        )
        
        if attack_type == "token":
            adversarial = self.token_attacker.attack_context(
                text, self.ppl_calc, self.stealth_calc, config, num_words
            )
        elif attack_type == "word":
            word_config = WordAttackConfig(
                attack_mode="degradation",
                stealth_threshold=0.8,
                ppl_margin=2.0
            )
            adversarial = self.word_attacker.attack_context(
                text, self.ppl_calc, self.stealth_calc, word_config, num_words
            )
        else:
            adversarial = self._synthetic_perturbation(text, attack_type)[0]
        
        # Calculate stealth score
        if self.stealth_calc:
            stealth = self.stealth_calc.calculate_stealthiness(
                text, adversarial, method=attack_type
            )
        else:
            stealth = 0.9  # Assume stealthy
        
        return adversarial, stealth
    
    def _synthetic_perturbation(
        self, 
        text: str, 
        attack_type: str
    ) -> Tuple[str, float]:
        """Generate synthetic adversarial perturbation without attack modules."""
        words = text.split()
        
        if attack_type == "token":
            # Character-level substitutions
            char_subs = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$'}
            modified = []
            for word in words:
                new_word = word
                for old, new in char_subs.items():
                    if old in word.lower() and np.random.random() < 0.3:
                        new_word = new_word.replace(old, new, 1)
                        break
                modified.append(new_word)
            return ' '.join(modified), 0.85
        
        elif attack_type == "word":
            # Punctuation injection
            modifiers = ['"', "'", "...", ",", "-"]
            modified = []
            for word in words:
                if np.random.random() < 0.2:
                    mod = np.random.choice(modifiers)
                    word = f"{mod}{word}"
                modified.append(word)
            return ' '.join(modified), 0.90
        
        return text, 1.0
    
    def evaluate_single(
        self,
        text: str,
        attack_type: str = "token"
    ) -> AttackDefenseResult:
        """
        Run full attack-defense evaluation on single example.
        """
        # Generate adversarial example
        adversarial, stealth = self.generate_adversarial(text, attack_type)
        
        # Run defense on adversarial text
        result = self.defense.detect(adversarial)
        
        # Determine if defense detected the attack
        detected = result.verdict == DefenseVerdict.POISONED
        
        return AttackDefenseResult(
            example_id=f"example_{hash(text) % 10000}",
            original_text=text,
            adversarial_text=adversarial,
            attack_type=attack_type,
            attack_success=True,  # Assume attack succeeded for evaluation
            defense_verdict=result.verdict.value,
            defense_confidence=result.confidence,
            defense_consistency=result.consistency_score,
            defense_detected=detected,
            stealth_score=stealth
        )
    
    def evaluate_batch(
        self,
        texts: List[str],
        attack_types: List[str] = None
    ) -> Dict:
        """
        Evaluate on batch of texts with multiple attack types.
        """
        if attack_types is None:
            attack_types = ["token", "word"]
        
        results = {
            "compression_aware": [],
            "random_edit": [],
            "perplexity": []
        }
        
        for text in tqdm(texts, desc="Evaluating"):
            for attack_type in attack_types:
                # Generate adversarial
                adversarial, stealth = self.generate_adversarial(text, attack_type)
                
                # Test each defense
                # 1. Compression-aware (our method)
                ca_result = self.defense.detect(adversarial)
                results["compression_aware"].append({
                    "attack_type": attack_type,
                    "detected": ca_result.verdict == DefenseVerdict.POISONED,
                    "consistency": ca_result.consistency_score,
                    "stealth": stealth
                })
                
                # 2. Random edit baseline
                re_result = self.random_defense.detect(adversarial)
                results["random_edit"].append({
                    "attack_type": attack_type,
                    "detected": re_result.verdict == DefenseVerdict.POISONED,
                    "consistency": re_result.consistency_score,
                    "stealth": stealth
                })
                
                # 3. Perplexity baseline
                ppl_result = self.ppl_defense.detect(adversarial)
                results["perplexity"].append({
                    "attack_type": attack_type,
                    "detected": ppl_result.verdict == DefenseVerdict.POISONED,
                    "consistency": ppl_result.consistency_score,
                    "stealth": stealth
                })
        
        return results
    
    def compute_metrics(self, results: Dict) -> Dict:
        """Compute detection metrics for each defense."""
        metrics = {}
        
        for defense_name, defense_results in results.items():
            total = len(defense_results)
            detected = sum(1 for r in defense_results if r["detected"])
            
            # Group by attack type
            by_attack = {}
            for r in defense_results:
                atype = r["attack_type"]
                if atype not in by_attack:
                    by_attack[atype] = {"total": 0, "detected": 0}
                by_attack[atype]["total"] += 1
                if r["detected"]:
                    by_attack[atype]["detected"] += 1
            
            metrics[defense_name] = {
                "total_detection_rate": detected / total if total > 0 else 0,
                "by_attack_type": {
                    atype: data["detected"] / data["total"] if data["total"] > 0 else 0
                    for atype, data in by_attack.items()
                },
                "avg_consistency": np.mean([r["consistency"] for r in defense_results])
            }
        
        return metrics


def run_integration_test():
    """Run integration test with sample data."""
    print("=" * 80)
    print("ATTACK-DEFENSE INTEGRATION TEST")
    print("=" * 80)
    
    # Sample texts for evaluation
    test_texts = [
        "iPhone 16 Pro features a sleek lightweight titanium design with advanced capabilities. The powerful A18 Pro chip delivers exceptional performance.",
        "Samsung Galaxy S24 Ultra is the ultimate Android flagship. Stunning 6.8-inch Dynamic AMOLED display with 120Hz refresh rate.",
        "The Amazon rainforest is the world's largest tropical rainforest covering over 5.5 million square kilometers.",
        "Mount Everest is the highest mountain on Earth with a peak at 8,848.86 meters above sea level.",
        "Photosynthesis is the process by which plants convert sunlight into chemical energy using chlorophyll.",
    ]
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = AttackDefensePipeline()
    
    # Run evaluation
    print("\nRunning attack-defense evaluation...")
    results = pipeline.evaluate_batch(test_texts, attack_types=["token", "word"])
    
    # Compute metrics
    metrics = pipeline.compute_metrics(results)
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print(f"\n{'Defense':<25} {'Detection Rate':>15} {'Avg Consistency':>18}")
    print("-" * 60)
    
    for defense_name, m in metrics.items():
        print(f"{defense_name:<25} {m['total_detection_rate']:>15.1%} {m['avg_consistency']:>18.3f}")
    
    print("\n\nDetection Rate by Attack Type:")
    print("-" * 60)
    for defense_name, m in metrics.items():
        print(f"\n  {defense_name}:")
        for atype, rate in m["by_attack_type"].items():
            print(f"    {atype}: {rate:.1%}")
    
    # Comparison to paper
    print("\n" + "=" * 80)
    print("COMPARISON TO PAPER")
    print("=" * 80)
    print("\nPaper reports existing defenses achieve <10% detection rate.")
    print("Our compression-aware defense should significantly improve this.")
    
    ca_rate = metrics["compression_aware"]["total_detection_rate"]
    if ca_rate > 0.1:
        improvement = ca_rate / 0.1
        print(f"\n✓ Compression-aware defense: {ca_rate:.1%} detection rate")
        print(f"  ({improvement:.1f}x improvement over paper's reported <10% baseline)")
    else:
        print(f"\n✗ Compression-aware defense: {ca_rate:.1%} detection rate")
        print("  (Further optimization needed)")
    
    print("=" * 80)
    
    return metrics


def main():
    """Main entry point."""
    # Check for model flag
    use_model = "--with-model" in sys.argv
    
    model, tokenizer = None, None
    if use_model:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print("Loading GPT-2 model...")
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            model = AutoModelForCausalLM.from_pretrained('gpt2')
            print("Model loaded!")
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    # Run test
    results = run_integration_test()
    
    # Save results
    output_file = "attack_defense_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()