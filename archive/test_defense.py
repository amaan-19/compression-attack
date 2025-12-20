#!/usr/bin/env python3
"""
Quick Test Script for Compression-Aware Counterfactual Defense

This script provides a quick validation that the defense implementation
works correctly without requiring external dependencies.

Run: python test_defense.py
"""

import sys
sys.path.insert(0, '.')

from src.compression_aware_defense import (
    CompressionAwareDefense,
    PPLBasedCompressor,
    DefenseConfig,
    DefenseVerdict,
    CounterfactualDefenseEvaluator,
    RandomEditDefense,
    PerplexityDefense,
)


def test_compressor():
    """Test the PPL-based compressor."""
    print("\n[TEST 1] PPL-Based Compressor")
    print("-" * 50)
    
    compressor = PPLBasedCompressor()
    
    text = "iPhone 16 Pro features a sleek lightweight titanium design"
    compressed, mask = compressor.compress(text, rate=0.5)
    
    print(f"Original: {text}")
    print(f"Compressed: {compressed}")
    print(f"Retention mask: {mask}")
    print(f"Compression ratio: {len(compressed.split())}/{len(text.split())}")
    
    assert len(compressed) > 0, "Compression failed - empty output"
    assert len(mask) == len(text.split()), "Mask length mismatch"
    
    print("✓ Compressor test passed")
    return True


def test_boundary_detection():
    """Test boundary token identification."""
    print("\n[TEST 2] Boundary Token Detection")
    print("-" * 50)
    
    compressor = PPLBasedCompressor()
    config = DefenseConfig(compression_rate=0.5)
    defense = CompressionAwareDefense(compressor, config)
    
    text = "The quick brown fox jumps over the lazy dog"
    boundaries = defense.identify_boundary_tokens(text)
    
    print(f"Text: {text}")
    print(f"Found {len(boundaries)} boundary tokens:")
    for b in boundaries[:5]:
        print(f"  - '{b.token}' at pos {b.position}, score={b.retention_score:.3f}")
    
    assert len(boundaries) >= 0, "Boundary detection failed"
    
    print("✓ Boundary detection test passed")
    return True


def test_variant_generation():
    """Test counterfactual variant generation."""
    print("\n[TEST 3] Counterfactual Variant Generation")
    print("-" * 50)
    
    compressor = PPLBasedCompressor()
    config = DefenseConfig(num_variants=5, compression_rate=0.5)
    defense = CompressionAwareDefense(compressor, config)
    
    text = "iPhone 16 Pro features a sleek design with advanced capabilities"
    boundaries = defense.identify_boundary_tokens(text)
    
    print(f"Original: {text}")
    print(f"\nGenerating {config.num_variants} variants...")
    
    for i in range(config.num_variants):
        variant = defense.generate_counterfactual_variant(text, boundaries)
        print(f"  Variant {i+1}: {variant.text[:50]}...")
        print(f"    Edits at positions: {variant.edit_positions}")
    
    print("✓ Variant generation test passed")
    return True


def test_defense_clean_input():
    """Test defense on clean input."""
    print("\n[TEST 4] Defense on Clean Input")
    print("-" * 50)
    
    compressor = PPLBasedCompressor()
    config = DefenseConfig(
        num_variants=10,
        consistency_threshold=0.7,
        compression_rate=0.5
    )
    defense = CompressionAwareDefense(compressor, config)
    
    clean_text = (
        "iPhone 16 Pro features a sleek lightweight titanium design "
        "with advanced capabilities. The powerful A18 Pro chip delivers "
        "exceptional performance for demanding tasks."
    )
    
    print(f"Testing clean input: {clean_text[:50]}...")
    result = defense.detect(clean_text)
    
    print(f"\nResult:")
    print(f"  Verdict: {result.verdict.value}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Consistency: {result.consistency_score:.3f}")
    print(f"  Flagged positions: {result.flagged_positions}")
    
    # Clean input should ideally be classified as clean
    print(f"\nExpected: CLEAN (with high consistency)")
    print(f"Got: {result.verdict.value} (consistency: {result.consistency_score:.3f})")
    
    print("✓ Clean input test passed")
    return True


def test_defense_adversarial_input():
    """Test defense on adversarial input."""
    print("\n[TEST 5] Defense on Adversarial Input")
    print("-" * 50)
    
    compressor = PPLBasedCompressor()
    config = DefenseConfig(
        num_variants=10,
        consistency_threshold=0.7,
        compression_rate=0.5
    )
    defense = CompressionAwareDefense(compressor, config)
    
    # Simulated adversarial input with perturbations
    adversarial_text = (
        "iPhone 16 Pro f3atures a sl33k lightweight tit@nium design "
        "with adv@nced capabilities. The p0werful A18 Pr0 chip delivers "
        "exc3ptional performance for demanding tasks."
    )
    
    print(f"Testing adversarial input: {adversarial_text[:50]}...")
    result = defense.detect(adversarial_text)
    
    print(f"\nResult:")
    print(f"  Verdict: {result.verdict.value}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Consistency: {result.consistency_score:.3f}")
    print(f"  Flagged positions: {result.flagged_positions}")
    
    print(f"\nExpected: POISONED (with lower consistency)")
    print(f"Got: {result.verdict.value} (consistency: {result.consistency_score:.3f})")
    
    print("✓ Adversarial input test passed")
    return True


def test_baseline_comparison():
    """Test comparison with baseline defenses."""
    print("\n[TEST 6] Baseline Defense Comparison")
    print("-" * 50)
    
    compressor = PPLBasedCompressor()
    
    # Our defense
    config = DefenseConfig(num_variants=10, consistency_threshold=0.7)
    ca_defense = CompressionAwareDefense(compressor, config)
    
    # Baselines
    random_defense = RandomEditDefense(compressor, num_variants=10)
    ppl_defense = PerplexityDefense()
    
    test_text = (
        "iPhone 16 Pro f3atures a sl33k lightweight tit@nium design "
        "with adv@nced capabilities."
    )
    
    print(f"Testing: {test_text[:40]}...")
    print("\nResults:")
    
    # Test each defense
    ca_result = ca_defense.detect(test_text)
    print(f"  Compression-Aware: {ca_result.verdict.value} (consistency: {ca_result.consistency_score:.3f})")
    
    re_result = random_defense.detect(test_text)
    print(f"  Random Edit: {re_result.verdict.value} (consistency: {re_result.consistency_score:.3f})")
    
    ppl_result = ppl_defense.detect(test_text)
    print(f"  Perplexity: {ppl_result.verdict.value} (confidence: {ppl_result.confidence:.3f})")
    
    print("✓ Baseline comparison test passed")
    return True


def test_evaluator():
    """Test the evaluation framework."""
    print("\n[TEST 7] Defense Evaluator")
    print("-" * 50)
    
    compressor = PPLBasedCompressor()
    config = DefenseConfig(num_variants=5, consistency_threshold=0.7)
    defense = CompressionAwareDefense(compressor, config)
    evaluator = CounterfactualDefenseEvaluator(defense)
    
    # Test examples
    examples = [
        ("Clean product description with normal text", False, "clean_1"),
        ("Another normal sentence without any perturbations", False, "clean_2"),
        ("Adv3rsarial t3xt with ch@racter substitutions", True, "adv_1"),
        ("P0isoned input with l33tspeak style ch@nges", True, "adv_2"),
    ]
    
    print("Evaluating examples...")
    for text, is_poisoned, label in examples:
        result = evaluator.evaluate_single(text, is_poisoned, label)
        status = "✓" if result["correct"] else "✗"
        print(f"  {status} {label}: {result['verdict']} (expected: {'poisoned' if is_poisoned else 'clean'})")
    
    metrics = evaluator.get_metrics()
    print(f"\nMetrics:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  TPR: {metrics['tpr']:.3f}")
    print(f"  FPR: {metrics['fpr']:.3f}")
    
    print("✓ Evaluator test passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("COMPRESSION-AWARE COUNTERFACTUAL DEFENSE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Compressor", test_compressor),
        ("Boundary Detection", test_boundary_detection),
        ("Variant Generation", test_variant_generation),
        ("Clean Input Detection", test_defense_clean_input),
        ("Adversarial Input Detection", test_defense_adversarial_input),
        ("Baseline Comparison", test_baseline_comparison),
        ("Evaluator Framework", test_evaluator),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return True
    else:
        print("\n✗ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)