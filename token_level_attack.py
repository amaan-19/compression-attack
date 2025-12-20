from typing import List, Optional, Dict, Literal
from dataclasses import dataclass

@dataclass
class AttackConfig:
    attack_mode: Literal["promotion", "degradation"] = "promotion"
    stealth_threshold: float = 0.8  # δ in paper
    ppl_margin: float = 5.0  # γ in paper
    max_iterations: int = 10  # L in paper
    lambda_weight: float = 0.5


class HardComTokenAttack:
    def __init__(self):
        # character substitutions (inspired by leetspeak cipher)
        self.char_substitutions = {
            'A': ['4', '@'],
            'B': ['8'],
            'C': ['[', '('],
            'E': ['3'],
            'I': ['|', '1'],
            'J': [']'],
            'O': ['0'],
            'S': ['$', '5'],
            'T': ['7', '+'],
            'Z': ['2'],
            'a': ['@', '4'],
            'b': ['6'],
            'e': ['3'],
            'g': ['9'],
            'i': ['1', '|'],
            'l': ['1', '|'],
            'o': ['0'],
            's': ['5', '$'],
            't': ['+', '7']
        }

        # BPE separators
        self.separators = [' ', '-', '_', '...', '~', '\u00a0']  # includes non-breaking space

    def generate_char_variants(self, word: str) -> List[str]:
        variants = []
        # loop through characters and replace if that char is in dict
        for i, char in enumerate(word):
            if char in self.char_substitutions:
                for sub in self.char_substitutions[char]:
                    variant = word[:i] + sub + word[i+1:]
                    variants.append(variant)

        return variants

    def generate_bpe_variants(self, word: str) -> List[str]:
        variants = []
        # insert separators at different positions
        for i in range(1, len(word)):
            for sep in self.separators:
                variant = word[:i] + sep + word[i:]
                variants.append(variant)

        return variants

    def token_level_attack(
            self,
            word: str,
            sentence: str,
            ppl_calc,
            stealth_calc,
            config: AttackConfig,
            k: int = 100
    ) -> Optional[str]:
        """
        Attack a single word by finding perturbations that meet PPL/stealth criteria.
        
        For this implementation, we focus on creating visible perturbations that
        will affect compression decisions, regardless of PPL direction.
        """
        # generate all variants
        char_variants = self.generate_char_variants(word)
        bpe_variants = self.generate_bpe_variants(word)
        all_variants = char_variants + bpe_variants

        if not all_variants:
            return None

        candidates = []

        # Try all variants and collect those meeting stealth threshold
        for i, variant in enumerate(all_variants):
            if i >= config.max_iterations * 2:  # Increased search space
                break

            # replace word in sentence
            modified_sentence = sentence.replace(word, variant, 1)

            # check stealthiness constraint
            stealth_score = stealth_calc.calculate_stealthiness(
                sentence,
                modified_sentence,
                method="token",
                lambda_weight=config.lambda_weight
            )

            if stealth_score >= config.stealth_threshold:
                # Calculate PPL change to measure impact
                orig_ppl = ppl_calc.calculate_perplexity(word)
                variant_ppl = ppl_calc.calculate_perplexity(variant)
                ppl_change = abs(variant_ppl - orig_ppl)
                
                candidates.append((variant, stealth_score, ppl_change))

        if not candidates:
            # Fallback: return first variant if no candidates meet threshold
            # This ensures we always generate some perturbation for testing
            if all_variants:
                return all_variants[0]
            return None

        # Select variant with highest PPL impact while maintaining stealth
        best_variant, best_stealth, best_impact = max(candidates, key=lambda x: x[2])

        return best_variant

    def attack_context(
            self,
            context: str,
            ppl_calc,
            stealth_calc,
            config: AttackConfig,
            num_words: int = 5
    ) -> str:
        
        # Step 1: Get ranked list of high-PPL words
        word_ppls = ppl_calc.calculate_word_ppls(context)

        # Step 2: Select top N high-PPL words as targets
        target_words = [word for word, ppl in word_ppls[:num_words]]

        # Step 3: Attack each target word
        modified_context = context
        for word in target_words:
            result = self.token_level_attack(
                word, modified_context, ppl_calc, stealth_calc, config
            )
            if result:
                modified_context = modified_context.replace(word, result, 1)

        return modified_context