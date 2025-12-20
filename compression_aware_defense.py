#!/usr/bin/env python3
"""
Compression-Aware Counterfactual Defense

This module implements a defense against prompt compression attacks by:
1. Identifying tokens near compression decision boundaries
2. Generating semantically-neutral perturbations near boundaries
3. Measuring retention pattern stability across variants
4. Flagging inputs with high instability as adversarial

Main classes:
- CompressionAwareDefense: Primary defense implementation
- HardPromptCompressor: Simulates hard prompt compression
- DefenseConfig: Configuration parameters
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Literal, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import re


class DefenseVerdict(Enum):
    """Classification result from the defense."""
    CLEAN = "clean"
    POISONED = "poisoned"
    UNCERTAIN = "uncertain"


@dataclass
class CompressionBoundary:
    """Represents a token at or near a compression decision boundary."""
    token: str
    position: int
    retention_score: float
    is_retained: bool
    boundary_distance: float
    # NEW: Track if token has suspicious characteristics
    has_special_chars: bool = False
    has_unusual_spacing: bool = False


@dataclass
class CounterfactualVariant:
    """A counterfactual variant of the original input."""
    text: str
    edit_positions: List[int]
    edit_type: str
    retained_tokens: Set[int]  # Positions of retained tokens
    retention_pattern: List[bool]


@dataclass
class DefenseResult:
    """Result from running the counterfactual defense."""
    verdict: DefenseVerdict
    confidence: float
    consistency_score: float
    num_variants: int
    retention_variance: float
    flagged_positions: List[int]
    suspicious_tokens: List[str]
    details: Dict = field(default_factory=dict)


@dataclass
class DefenseConfig:
    """Configuration for the counterfactual defense."""
    num_variants: int = 15
    consistency_threshold: float = 0.75  # Below = poisoned
    boundary_window: int = 2
    min_edits_per_variant: int = 1
    max_edits_per_variant: int = 3
    compression_rate: float = 0.5
    # NEW: Threshold for flagging suspicious tokens
    suspicious_char_threshold: int = 1  # Tokens with this many special chars are suspicious


class HardPromptCompressor:
    """
    Simulates hard prompt compression (like LLMLingua/Selective Context).
    
    Key behavior: Tokens with HIGH perplexity are RETAINED (important).
    Tokens with LOW perplexity are DROPPED (predictable/redundant).
    """
    
    def __init__(self, ppl_calculator=None):
        self.ppl_calc = ppl_calculator
    
    def calculate_token_importance(self, text: str) -> List[Tuple[str, float]]:
        """
        Calculate importance score for each token.
        Higher score = more likely to be retained.
        """
        words = text.split()
        
        if self.ppl_calc is not None:
            # Use actual PPL calculator
            word_ppls = self.ppl_calc.calculate_word_ppls(text)
            # Sort back to original order
            ppl_dict = {w: p for w, p in word_ppls}
            return [(w, ppl_dict.get(w, 10.0)) for w in words]
        
        # Fallback: heuristic-based importance
        importances = []
        for word in words:
            score = 10.0  # Base score
            
            # Content words are more important
            if len(word) > 4:
                score += 5
            
            # Special characters increase "importance" (adversarial signal)
            special_chars = sum(1 for c in word if not c.isalnum() and c not in ' -')
            score += special_chars * 10
            
            # Capitalized words (proper nouns) are important
            if word[0].isupper():
                score += 3
            
            # Numbers are important
            if any(c.isdigit() for c in word):
                score += 4
            
            importances.append((word, score))
        
        return importances
    
    def compress(self, text: str, rate: float = 0.5) -> Tuple[str, List[bool], List[float]]:
        """
        Compress text by retaining high-importance tokens.
        
        Returns:
            - compressed_text: The compressed output
            - retention_mask: Boolean mask of which tokens were kept
            - importance_scores: Importance score for each token
        """
        token_importances = self.calculate_token_importance(text)
        words = [t[0] for t in token_importances]
        scores = [t[1] for t in token_importances]
        
        num_to_keep = max(1, int(len(words) * rate))
        
        # Get indices sorted by importance (descending)
        sorted_indices = np.argsort(scores)[::-1]
        keep_indices = set(sorted_indices[:num_to_keep])
        
        # Create retention mask (preserves original order)
        retention_mask = [i in keep_indices for i in range(len(words))]
        
        # Build compressed text
        compressed_words = [w for i, w in enumerate(words) if retention_mask[i]]
        compressed_text = ' '.join(compressed_words)
        
        return compressed_text, retention_mask, scores
    
    def get_boundary_tokens(
        self, 
        text: str, 
        rate: float = 0.5
    ) -> List[CompressionBoundary]:
        """
        Identify tokens near the compression decision boundary.
        
        These are tokens whose retention could change with small perturbations.
        """
        _, retention_mask, scores = self.compress(text, rate)
        words = text.split()
        
        if not scores:
            return []
        
        # Normalize scores to [0, 1]
        min_s, max_s = min(scores), max(scores)
        if max_s > min_s:
            norm_scores = [(s - min_s) / (max_s - min_s) for s in scores]
        else:
            norm_scores = [0.5] * len(scores)
        
        # Find the retention threshold (score of least important retained token)
        retained_scores = [scores[i] for i, kept in enumerate(retention_mask) if kept]
        if retained_scores:
            threshold = min(retained_scores)
        else:
            threshold = np.median(scores)
        
        boundaries = []
        for i, (word, score, norm_score, retained) in enumerate(
            zip(words, scores, norm_scores, retention_mask)
        ):
            # Distance to threshold
            boundary_dist = abs(score - threshold) / (max_s - min_s + 1e-6)
            
            # Check for suspicious characteristics
            has_special = sum(1 for c in word if not c.isalnum() and c not in ' -.,!?') > 0
            has_spacing = '\u00a0' in word or '  ' in word or any(
                c in word for c in ['\t', '_', '~']
            )
            
            boundaries.append(CompressionBoundary(
                token=word,
                position=i,
                retention_score=norm_score,
                is_retained=retained,
                boundary_distance=boundary_dist,
                has_special_chars=has_special,
                has_unusual_spacing=has_spacing
            ))
        
        # Sort by boundary distance (closest to threshold first)
        boundaries.sort(key=lambda x: x.boundary_distance)
        
        return boundaries


class CompressionAwareDefense:
    """
    Improved counterfactual defense targeting compression boundaries.
    
    Key insight: Adversarial perturbations in CompressionAttack are fragile.
    They're tuned to specific PPL thresholds. Small edits near them should
    cause retention patterns to flip, which we can detect.
    """
    
    def __init__(
        self,
        compressor: Optional[HardPromptCompressor] = None,
        config: Optional[DefenseConfig] = None
    ):
        self.compressor = compressor or HardPromptCompressor()
        self.config = config or DefenseConfig()
        
        # Neutral character normalizations (shouldn't change semantics)
        self.char_normalizations = {
            '@': 'a', '4': 'a',
            '3': 'e',
            '1': 'i', '|': 'i',
            '0': 'o',
            '5': 's', '$': 's',
            '7': 't', '+': 't',
            '9': 'g',
            '6': 'b',
            '2': 'z',
        }
        
        # Common word variations for neutral edits
        self.neutral_variations = {
            'the': ['a', 'this'],
            'a': ['the', 'one'],
            'is': ['was', 'remains'],
            'are': ['were', 'remain'],
            'has': ['had', 'contains'],
            'with': ['featuring', 'having'],
            'and': ['plus', '&'],
            'for': ['to', 'toward'],
        }
    
    def detect_suspicious_tokens(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Detect tokens that have characteristics of adversarial perturbations.
        
        IMPORTANT: Must distinguish between:
        - Normal text features (proper nouns, numbers, punctuation)
        - Adversarial perturbations (leetspeak substitutions, unusual separators)
        
        Returns list of (position, token, reason)
        """
        words = text.split()
        suspicious = []
        
        # Characters that indicate leetspeak (NOT including standalone digits)
        leetspeak_indicators = set('@$|+')
        
        # Patterns that suggest adversarial modification
        digit_letter_mix_pattern = re.compile(r'[a-zA-Z][0-9]|[0-9][a-zA-Z]')  # e.g., "f3atures", "sl33k"
        
        for i, word in enumerate(words):
            reasons = []
            
            # Skip pure numbers (like "16", "2024") - these are normal
            if word.isdigit():
                continue
            
            # Skip words that are just capitalized normally (like "iPhone", "MacBook")
            # Only flag mixed case if it's unusual patterns like "tItAnIuM"
            
            # Check for leetspeak-style substitutions WITHIN words
            # Must have letters mixed with suspicious chars
            has_letters = any(c.isalpha() for c in word)
            leetspeak_chars = sum(1 for c in word if c in leetspeak_indicators)
            
            if has_letters and leetspeak_chars >= 1:
                reasons.append(f"leetspeak_chars:{leetspeak_chars}")
            
            # Check for digit-letter mixing (e.g., "f3atures", "adv4nced")
            # This is different from normal words with numbers like "iPhone16"
            if has_letters and digit_letter_mix_pattern.search(word):
                # Make sure it's not a normal product name pattern
                # Normal: "iPhone16", "PS5", "M1" (letter then digits at end, or short)
                # Suspicious: "f3atures", "sl33k", "tit4nium"
                if len(word) > 3:  # Only flag longer words
                    # Check if digits are embedded in the middle of letters
                    stripped = word.strip('0123456789')
                    if any(c.isdigit() for c in stripped):
                        reasons.append("embedded_digits")
            
            # Check for unusual spacing/separators within word
            unusual_seps = ['\u00a0', '~', '...']
            for sep in unusual_seps:
                if sep in word:
                    reasons.append("unusual_separator")
                    break
            
            # Check for underscore or hyphen in unusual positions (mid-word splits)
            # Normal: "well-known", "self-driving"  
            # Suspicious: "ti_tanium", "fea-tures" (splitting normal words)
            if '_' in word and not word.startswith('_') and not word.endswith('_'):
                parts = word.split('_')
                if len(parts) == 2 and all(len(p) >= 2 for p in parts):
                    # Check if this looks like a split word
                    combined = ''.join(parts).lower()
                    if combined.isalpha() and len(combined) > 4:
                        reasons.append("word_split_underscore")
            
            if reasons:
                suspicious.append((i, word, ';'.join(reasons)))
        
        return suspicious
    
    def normalize_suspicious_token(self, token: str) -> str:
        """
        Normalize a suspicious token by reverting likely adversarial changes.
        """
        normalized = token
        for adv_char, normal_char in self.char_normalizations.items():
            normalized = normalized.replace(adv_char, normal_char)
        
        # Remove unusual separators
        for sep in ['\u00a0', '_', '~']:
            normalized = normalized.replace(sep, '')
        
        return normalized
    
    def generate_probing_variant(
        self,
        text: str,
        boundaries: List[CompressionBoundary],
        suspicious_positions: Set[int]
    ) -> CounterfactualVariant:
        """
        Generate a variant that probes compression stability.
        
        Strategy: Make small edits near boundary/suspicious tokens.
        For clean text: retention should stay stable.
        For adversarial text: retention should flip (fragile perturbations).
        """
        words = text.split()
        modified = words.copy()
        edit_positions = []
        
        # Prioritize editing near suspicious tokens
        target_positions = []
        
        # First: positions with suspicious tokens
        for pos in suspicious_positions:
            if pos < len(words):
                target_positions.append(pos)
                # Also add neighbors
                if pos > 0:
                    target_positions.append(pos - 1)
                if pos < len(words) - 1:
                    target_positions.append(pos + 1)
        
        # Second: boundary tokens
        for b in boundaries[:5]:
            if b.position not in target_positions:
                target_positions.append(b.position)
        
        # Remove duplicates, limit edits
        target_positions = list(dict.fromkeys(target_positions))
        num_edits = min(
            len(target_positions),
            random.randint(self.config.min_edits_per_variant, 
                          self.config.max_edits_per_variant)
        )
        
        if not target_positions:
            # Fallback: random positions
            target_positions = list(range(len(words)))
        
        edit_targets = random.sample(
            target_positions, 
            min(num_edits, len(target_positions))
        )
        
        for pos in edit_targets:
            if pos >= len(modified):
                continue
            
            word = modified[pos]
            
            # Choose edit strategy
            if pos in suspicious_positions:
                # For suspicious tokens: try to normalize them
                new_word = self.normalize_suspicious_token(word)
            else:
                # For normal tokens: small neutral edit
                new_word = self._apply_neutral_edit(word)
            
            if new_word != word:
                modified[pos] = new_word
                edit_positions.append(pos)
        
        variant_text = ' '.join(modified)
        
        # Get compression result for variant
        _, retention_mask, _ = self.compressor.compress(
            variant_text, 
            self.config.compression_rate
        )
        
        retained_positions = {i for i, kept in enumerate(retention_mask) if kept}
        
        return CounterfactualVariant(
            text=variant_text,
            edit_positions=edit_positions,
            edit_type="probing",
            retained_tokens=retained_positions,
            retention_pattern=retention_mask
        )
    
    def _apply_neutral_edit(self, word: str) -> str:
        """Apply a small, semantically-neutral edit to a word."""
        word_lower = word.lower()
        
        # Try word replacement
        if word_lower in self.neutral_variations:
            replacement = random.choice(self.neutral_variations[word_lower])
            if word[0].isupper():
                replacement = replacement.capitalize()
            return replacement
        
        # Try punctuation variation
        if random.random() < 0.3:
            if word.endswith(','):
                return word[:-1]
            elif len(word) > 2 and random.random() < 0.3:
                return word + ','
        
        return word
    
    def calculate_retention_instability(
        self,
        original_pattern: List[bool],
        variant_patterns: List[List[bool]],
        suspicious_positions: Set[int]
    ) -> Tuple[float, float, List[int]]:
        """
        Calculate how unstable retention is, especially near suspicious tokens.
        
        Returns:
            - overall_instability: How much retention changes across variants (0-1)
            - suspicious_instability: Instability specifically near suspicious tokens
            - unstable_positions: Positions with high retention variance
        """
        if not variant_patterns:
            return 0.0, 0.0, []
        
        num_positions = len(original_pattern)
        
        # Count how often each position's retention flips
        flip_counts = [0] * num_positions
        
        for variant_pattern in variant_patterns:
            for i in range(min(num_positions, len(variant_pattern))):
                if variant_pattern[i] != original_pattern[i]:
                    flip_counts[i] += 1
        
        num_variants = len(variant_patterns)
        flip_rates = [c / num_variants for c in flip_counts]
        
        # Overall instability
        overall_instability = np.mean(flip_rates) if flip_rates else 0.0
        
        # Instability near suspicious positions
        suspicious_flip_rates = []
        for pos in suspicious_positions:
            # Include position and neighbors
            for p in [pos - 1, pos, pos + 1]:
                if 0 <= p < len(flip_rates):
                    suspicious_flip_rates.append(flip_rates[p])
        
        suspicious_instability = np.mean(suspicious_flip_rates) if suspicious_flip_rates else 0.0
        
        # Positions with high instability (>30% flip rate)
        unstable_positions = [i for i, rate in enumerate(flip_rates) if rate > 0.3]
        
        return overall_instability, suspicious_instability, unstable_positions
    
    def detect(self, text: str) -> DefenseResult:
        """
        Run compression-aware counterfactual defense.
        
        Algorithm:
        1. Detect suspicious tokens (potential adversarial perturbations)
        2. Get compression boundaries
        3. Generate probing variants that edit near suspicious/boundary tokens
        4. Measure retention instability across variants
        5. High instability near suspicious tokens = likely poisoned
        """
        # Step 1: Detect suspicious tokens
        suspicious = self.detect_suspicious_tokens(text)
        suspicious_positions = {pos for pos, _, _ in suspicious}
        suspicious_tokens = [token for _, token, _ in suspicious]
        
        # Step 2: Get compression boundaries
        boundaries = self.compressor.get_boundary_tokens(text, self.config.compression_rate)
        
        # Step 3: Get original compression
        original_compressed, original_pattern, _ = self.compressor.compress(
            text, self.config.compression_rate
        )
        
        # Step 4: Generate probing variants
        variants = []
        for _ in range(self.config.num_variants):
            variant = self.generate_probing_variant(
                text, boundaries, suspicious_positions
            )
            variants.append(variant)
        
        # Step 5: Analyze retention instability
        variant_patterns = [v.retention_pattern for v in variants]
        overall_instability, suspicious_instability, unstable_positions = \
            self.calculate_retention_instability(
                original_pattern, variant_patterns, suspicious_positions
            )
        
        # Step 6: Make verdict
        # Key insight: Adversarial inputs have:
        # 1. Suspicious tokens (leetspeak, unusual separators)
        # 2. High instability when we normalize those suspicious tokens
        
        has_suspicious_tokens = len(suspicious) > 0
        num_suspicious = len(suspicious)
        
        # Thresholds for decision
        HIGH_INSTABILITY = 0.25
        MODERATE_INSTABILITY = 0.15
        STRONG_SUSPICIOUS = 3  # Multiple suspicious tokens is strong signal
        
        # Decision logic - prioritize precision to reduce FPR
        if num_suspicious >= STRONG_SUSPICIOUS:
            # Many suspicious tokens - likely poisoned
            verdict = DefenseVerdict.POISONED
            confidence = min(1.0, 0.5 + num_suspicious * 0.1 + overall_instability)
        elif has_suspicious_tokens and suspicious_instability > HIGH_INSTABILITY:
            # Suspicious tokens AND high instability near them
            verdict = DefenseVerdict.POISONED
            confidence = min(1.0, 0.6 + suspicious_instability)
        elif has_suspicious_tokens and overall_instability > MODERATE_INSTABILITY:
            # Suspicious tokens with moderate overall instability
            verdict = DefenseVerdict.POISONED
            confidence = min(1.0, 0.5 + overall_instability)
        elif not has_suspicious_tokens and overall_instability < 0.1:
            # No suspicious tokens and stable - clean
            verdict = DefenseVerdict.CLEAN
            confidence = 0.9
        elif not has_suspicious_tokens:
            # No suspicious tokens - likely clean even with some instability
            verdict = DefenseVerdict.CLEAN
            confidence = 0.7
        elif has_suspicious_tokens and overall_instability < 0.1:
            # Has suspicious tokens but very stable - might be false positive
            # Be conservative and mark as uncertain or clean
            verdict = DefenseVerdict.UNCERTAIN
            confidence = 0.5
        else:
            # Default: uncertain
            verdict = DefenseVerdict.UNCERTAIN
            confidence = 0.5
        
        # Consistency score (inverse of instability)
        consistency_score = 1.0 - overall_instability
        
        return DefenseResult(
            verdict=verdict,
            confidence=min(1.0, confidence),
            consistency_score=consistency_score,
            num_variants=len(variants),
            retention_variance=overall_instability,
            flagged_positions=unstable_positions,
            suspicious_tokens=suspicious_tokens,
            details={
                "original_compressed": original_compressed,
                "num_suspicious_tokens": len(suspicious),
                "suspicious_details": suspicious,
                "overall_instability": overall_instability,
                "suspicious_instability": suspicious_instability,
                "num_boundary_tokens": len(boundaries),
            }
        )


# Baseline defenses for comparison

class RandomEditDefense:
    """Baseline: Random edit counterfactual testing."""
    
    def __init__(self, compressor: HardPromptCompressor, num_variants: int = 10):
        self.compressor = compressor
        self.num_variants = num_variants
    
    def detect(self, text: str) -> DefenseResult:
        _, original_pattern, _ = self.compressor.compress(text, 0.5)
        
        variant_patterns = []
        words = text.split()
        
        for _ in range(self.num_variants):
            # Random character edits
            modified = words.copy()
            num_edits = random.randint(1, 3)
            
            for _ in range(num_edits):
                if not modified:
                    break
                pos = random.randint(0, len(modified) - 1)
                word = modified[pos]
                if len(word) > 2:
                    char_pos = random.randint(0, len(word) - 1)
                    modified[pos] = word[:char_pos] + '_' + word[char_pos + 1:]
            
            variant_text = ' '.join(modified)
            _, pattern, _ = self.compressor.compress(variant_text, 0.5)
            variant_patterns.append(pattern)
        
        # Calculate consistency
        consistencies = []
        for vp in variant_patterns:
            if len(vp) == len(original_pattern):
                matches = sum(a == b for a, b in zip(original_pattern, vp))
                consistencies.append(matches / len(original_pattern))
        
        consistency = np.mean(consistencies) if consistencies else 1.0
        
        verdict = DefenseVerdict.CLEAN if consistency > 0.7 else DefenseVerdict.POISONED
        
        return DefenseResult(
            verdict=verdict,
            confidence=consistency if verdict == DefenseVerdict.CLEAN else 1 - consistency,
            consistency_score=consistency,
            num_variants=self.num_variants,
            retention_variance=np.var(consistencies) if consistencies else 0,
            flagged_positions=[],
            suspicious_tokens=[],
            details={"method": "random_edit"}
        )


class PerplexityDefense:
    """Baseline: Perplexity-based detection."""
    
    def __init__(self, ppl_calculator=None, threshold: float = 30.0):
        self.ppl_calc = ppl_calculator
        self.threshold = threshold
    
    def detect(self, text: str) -> DefenseResult:
        if self.ppl_calc:
            ppl = self.ppl_calc.calculate_perplexity(text)
        else:
            # Heuristic: count special characters
            special = sum(1 for c in text if c in '@$0123456789|+_~')
            ppl = 15 + special * 3
        
        verdict = DefenseVerdict.POISONED if ppl > self.threshold else DefenseVerdict.CLEAN
        confidence = min(1.0, abs(ppl - self.threshold) / self.threshold)
        
        return DefenseResult(
            verdict=verdict,
            confidence=confidence,
            consistency_score=1.0 - (ppl / 100),
            num_variants=0,
            retention_variance=0,
            flagged_positions=[],
            suspicious_tokens=[],
            details={"perplexity": ppl, "method": "perplexity"}
        )


def demo():
    """Quick demonstration of the defense."""
    print("=" * 70)
    print("COMPRESSION-AWARE COUNTERFACTUAL DEFENSE (REVISED)")
    print("=" * 70)
    
    compressor = HardPromptCompressor()
    config = DefenseConfig(num_variants=15, compression_rate=0.5)
    defense = CompressionAwareDefense(compressor, config)
    
    # Clean example
    clean = "iPhone 16 Pro features a sleek titanium design with advanced camera capabilities"
    
    # Adversarial example (token-level attack style)
    adversarial = "iPhone 16 Pro f3atures a sl33k tit@nium design with adv@nced cam3ra capabilities"
    
    print(f"\n[1] CLEAN INPUT:")
    print(f"    Text: {clean}")
    result = defense.detect(clean)
    print(f"    Verdict: {result.verdict.value}")
    print(f"    Confidence: {result.confidence:.3f}")
    print(f"    Suspicious tokens: {result.suspicious_tokens}")
    print(f"    Instability: {result.details.get('overall_instability', 0):.3f}")
    
    print(f"\n[2] ADVERSARIAL INPUT:")
    print(f"    Text: {adversarial}")
    result = defense.detect(adversarial)
    print(f"    Verdict: {result.verdict.value}")
    print(f"    Confidence: {result.confidence:.3f}")
    print(f"    Suspicious tokens: {result.suspicious_tokens}")
    print(f"    Instability: {result.details.get('overall_instability', 0):.3f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()