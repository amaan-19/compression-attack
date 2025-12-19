#!/usr/bin/env python3

from typing import Dict, List, Tuple

class CompressionTester:
    def __init__(self):
        self.compressor = None
        self.setup_llmlingua()

    def setup_llmlingua(self):
        try:
            from llmlingua import PromptCompressor
            self.compressor = PromptCompressor(model_name="gpt-2", use_llmlingua2=True, device_map="cpu")
        except ImportError:
            print("Error: LLMLingua not installed")
            raise

    def compress(self, context: str, compression_rate: float) -> Dict:
        result = self.compressor.compress_prompt(context=[context], rate=compression_rate)
        return {
            "compressed_text": result["compressed_prompt"],
            "compression_ratio": result["ratio"]
        }

    def compare_compressions(self, original_context: str, adversarial_context: str, compression_rate: float) -> Dict:
        print(f"\nCompressing with LLMLingua...")
        print(f"Compression rate: {compression_rate}")

        # compress both contexts
        original_result = self.compress(context=original_context, compression_rate=compression_rate)
        adversarial_result = self.compress(context=adversarial_context, compression_rate=compression_rate)
        return {
            "original_compressed": original_result["compressed_text"],
            "adversarial_compressed": adversarial_result["compressed_text"],
            "compression_ratio_orig": original_result["compression_ratio"],
            "compression_ratio_adv": adversarial_result["compression_ratio"]
        }

    def print_comparison(self, comparison: Dict):
        print("COMPRESSION COMPARISON RESULTS")
        print("\nORIGINAL COMPRESSED:")
        print(f"  {comparison['original_compressed']}")
        print(f"  Ratio: {comparison['compression_ratio_orig']}")
        print("\nADVERSARIAL COMPRESSED:")
        print(f"  {comparison['adversarial_compressed']}")
        print(f"  Ratio: {comparison['compression_ratio_adv']}")


if __name__ == "__main__":
    # Test with token-level attack example
    from token_level_attack import HardComTokenAttack, AttackConfig
    from ppl_calculator import PPLCalculator
    from stealth_calculator import StealthCalculator

    # Setup
    ppl_calc = PPLCalculator()
    stealth_calc = StealthCalculator()
    perturber = HardComTokenAttack()

    # Original context
    original = "iPhone 16 Pro features a sleek lightweight titanium design with advanced capabilities. The powerful A18 Pro chip delivers exceptional performance for demanding tasks. The stunning 6.3-inch Super Retina XDR display offers incredible brightness and color accuracy. Advanced camera system with 48MP main sensor captures professional-quality photos and videos. All-day battery life ensures you stay connected throughout your day."

    # Generate adversarial context
    config = AttackConfig(
        attack_mode="promotion",
        stealth_threshold=0.8,
        ppl_margin=2.0
    )

    print("Generating adversarial context...")
    adversarial = perturber.attack_context(
        original, ppl_calc, stealth_calc, config, num_words=5
    )

    print(f"\nOriginal: {original}")
    print(f"Adversarial: {adversarial}")

    # Test with LLMLingua
    print("\n" + "="*70)
    print("TESTING WITH LLMLINGUA")
    print("="*70)

    tester = CompressionTester()
    comparison = tester.compare_compressions(
        original, adversarial, compression_rate=0.2
    )
    tester.print_comparison(comparison)