#!/usr/bin/env python3

from typing import List, Optional, Literal
from dataclasses import dataclass
import ssl
import nltk
from nltk.corpus import wordnet

# download wordnet data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('wordnet')
nltk.download('omw-1.4')

@dataclass
class AttackConfig:
    attack_mode: Literal["promotion", "degradation"] = "promotion"
    stealth_threshold: float = 0.8  # δ in paper
    ppl_margin: float = 5.0  # γ in paper
    max_iterations: int = 10  # L in paper
    lambda_weight: float = 0.5


class HardComWordAttack:
    def __init__(self):
        # punctuation marks
        self.punctuation_marks = ['"', "'", '-', '...', '~', '_']

        # discourse modifiers
        self.discourse_modifiers = [
            "notably", "however", "in fact", "indeed",
            "particularly", "specifically", "essentially",
            "clearly", "obviously"
        ]

    def generate_synonym_variants(self, word: str) -> List[str]:
        synonyms = set()
        # get all synsets for the word
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                # only add if different from original
                if (synonym.lower() != word.lower()
                and ' ' not in synonym
                and len(synonym) <= len(word) + 4
                and synonym.isalpha()):
                    synonyms.add(synonym)

        return list(synonyms)

    def generate_punctuation_variants(self, word: str, sentence: str) -> List[str]:
        variants = []

        for punct in self.punctuation_marks:
            # before word
            variant = sentence.replace(word, f"{punct}{word}", 1)
            variants.append(variant)

            # after word
            variant = sentence.replace(word, f"{word}{punct}", 1)
            variants.append(variant)

            # both sides
            variant = sentence.replace(word, f"{punct}{word}{punct}", 1)
            variants.append(variant)

        return variants

    def generate_modifier_variants(self, word: str, sentence: str) -> List[str]:
        variants = []

        for modifier in self.discourse_modifiers:
            # prepend modifier before word (with comma for natural flow)
            variant = sentence.replace(word, f"{modifier}, {word}", 1)
            variants.append(variant)

        return variants

    def word_level_attack(
            self,
            word: str,
            sentence: str,
            ppl_calc,  # PPLCalculator instance
            stealth_calc,  # StealthCalculator instance
            config: AttackConfig,
            k: int = 100  # Number of tokens to retain after compression
    ) -> Optional[str]:
        # calculate word PPL scores for full context to get PPL_k(C)
        word_ppls = ppl_calc.calculate_word_ppls(sentence)

        # get k-th highest PPL (PPL_k(C) from paper)
        if len(word_ppls) >= k:
            ppl_k = word_ppls[k-1][1]
        else:
            ppl_k = word_ppls[-1][1] if word_ppls else 0.0

        # determine PPL threshold based on attack mode (Section 4.1.2)
        if config.attack_mode == "promotion":
            # PPL(w) > τ_prom = PPL_k(C) + γ
            threshold = ppl_k + config.ppl_margin
            meets_threshold = lambda ppl: ppl > threshold
        else:  # degradation
            # PPL(w) < τ_dgrad = PPL_k(C)
            threshold = ppl_k
            meets_threshold = lambda ppl: ppl < threshold

        # generate all variants using three techniques (Section 4.1.3)
        synonym_variants = self.generate_synonym_variants(word)
        punctuation_variants = self.generate_punctuation_variants(word, sentence)
        modifier_variants = self.generate_modifier_variants(word, sentence)

        # combine all variants
        all_variants = []

        # for synonyms, replace word in sentence
        for syn in synonym_variants:
            variant_sentence = sentence.replace(word, syn, 1)
            all_variants.append((syn, variant_sentence))

        # punctuation and modifiers already return full sentences
        for variant_sentence in punctuation_variants + modifier_variants:
            # extract the modified word for PPL calculation
            all_variants.append((word, variant_sentence))

        candidates = []

        # local search with early stopping (limited to L iterations)
        for i, (variant_word, modified_sentence) in enumerate(all_variants):
            if i >= config.max_iterations:
                break

            # check stealthiness constraint using word-level metric (Section 4.1.3)
            # uses BERTScore instead of edit distance
            stealth_score = stealth_calc.calculate_stealthiness(
                sentence,
                modified_sentence,
                method="word",  # Uses BERTScore + cosine similarity
                lambda_weight=config.lambda_weight
            )

            if stealth_score < config.stealth_threshold:
                continue

            # check PPL threshold
            variant_ppl = ppl_calc.calculate_perplexity(variant_word)

            if meets_threshold(variant_ppl):
                candidates.append((modified_sentence, stealth_score, variant_ppl))

        if not candidates:
            return None

        # selection: minimize distance to threshold (most effective)
        best_sentence, best_stealth, best_ppl = min(
            candidates,
            key=lambda x: abs(x[2] - threshold)
        )

        return best_sentence

    def attack_context(
            self,
            context: str,
            ppl_calc,
            stealth_calc,
            config: AttackConfig,
            num_words: int = 5,
            k: int = 100
    ) -> str:
        # Step 1: Get ranked list of high-PPL words (Section 4.1.1)
        word_ppls = ppl_calc.calculate_word_ppls(context)

        # Step 2: Select top N high-PPL words as targets
        target_words = [word for word, ppl in word_ppls[:num_words]]

        # Step 3: Attack each target word
        modified_context = context
        for word in target_words:
            result = self.word_level_attack(
                word, modified_context, ppl_calc, stealth_calc, config, k
            )
            if result:
                modified_context = result  # Replace entire sentence

        return modified_context


if __name__ == "__main__":
    from ppl_calculator import PPLCalculator
    from stealth_calculator import StealthCalculator

    # Initialize components
    ppl_calc = PPLCalculator()
    stealth_calc = StealthCalculator()
    perturber = HardComWordAttack()

    # Example context
    context = "iPhone 16 Pro features a sleek lightweight titanium design with advanced capabilities"

    print("Original context:")
    print(f"  {context}\n")

    # Show high-PPL words (attack targets)
    word_ppls = ppl_calc.calculate_word_ppls(context)
    print("Top 5 high-PPL words (will be attacked):")
    for i, (word, ppl) in enumerate(word_ppls[:5], 1):
        print(f"  {i}. {word} (PPL: {ppl:.2f})")
    print()

    # Promotion attack
    config = AttackConfig(
        attack_mode="promotion",
        stealth_threshold=0.8,
        ppl_margin=2.0
    )

    print("Performing word-level promotion attack...")
    adversarial_context = perturber.attack_context(
        context, ppl_calc, stealth_calc, config, num_words=3
    )

    print(f"\nAdversarial context:")
    print(f"  {adversarial_context}")

    stealth = stealth_calc.calculate_stealthiness(
        context, adversarial_context, method="word"
    )
    print(f"\nStealthiness: {stealth:.3f}")