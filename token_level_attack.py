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
        # character substitutions (inspired by leetspeak cipher at https://leetspeak-converter.com)
        self.char_substitutions = {
            'A': '4',
            'B': '8',
            'C': '[',
            'E': '3',
            'I': '|',
            'J': ']',
            'O': '0',
            'S': '$',
            'T': '7',
            'Z': '2',
            'a': '@',
            'b': '6',
            'g': '9',
            'l': '1',
            's': '5',
            't': '+'
        }

        # BPE separators
        self.separators = [' ', '-', '_', '...', '~']

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
        #  calculate word PPL scores for full context to get PPL_k(C)
        word_ppls = ppl_calc.calculate_word_ppls(sentence)

        # get k-th highest PPL (PPL_k(C) from paper)
        if len(word_ppls) >= k:
            ppl_k = word_ppls[k-1][1]  # k-th word's PPL
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

        # generate all variants
        char_variants = self.generate_char_variants(word)
        bpe_variants = self.generate_bpe_variants(word)
        all_variants = char_variants + bpe_variants

        candidates = []

        # local search with early stopping (limited to L iterations)
        for i, variant in enumerate(all_variants):
            if i >= config.max_iterations:
                break

            # replace word in sentence
            modified_sentence = sentence.replace(word, variant, 1)

            # check stealthiness constraint (must be ≥ δ)
            stealth_score = stealth_calc.calculate_stealthiness(
                sentence,
                modified_sentence,
                method="token",
                lambda_weight=config.lambda_weight
            )

            if stealth_score < config.stealth_threshold:
                continue

            # check PPL threshold
            variant_ppl = ppl_calc.calculate_perplexity(variant)

            if meets_threshold(variant_ppl):
                candidates.append((variant, stealth_score, variant_ppl))

                # early stopping: found variant that meets threshold
                # continue searching to find better candidates within L iterations

        if not candidates:
            return None

        # Selection: From paper Section 4.1.2
        # "w* = arg min PPL(w_l)" subject to Stealth ≥ δ
        # Among candidates passing stealth constraint, select minimum PPL
        # Then from final selection, pick max stealthiness (Section 4.1.2 end)

        # Primary: Select by minimum PPL (closer to threshold = more effective)
        best_variant, best_stealth, best_ppl = min(candidates, key=lambda x: abs(x[2] - threshold))

        return best_variant

    def attack_context(
            self,
            context: str,
            ppl_calc,
            stealth_calc,
            config: AttackConfig,
            num_words: int = 5  # How many high-PPL words to attack
    ) -> str:
        
        # Step 1: Get ranked list of high-PPL words (Section 4.1.1)
        print("Getting ranked list of high-ppl words...")
        word_ppls = ppl_calc.calculate_word_ppls(context)

        # Step 2: Select top N high-PPL words as targets
        print("Choosing target words...")
        target_words = [word for word, ppl in word_ppls[:num_words]]
        print("Target Words:")
        for word in target_words:
            print("{word}\n")

        # Step 3: Attack each target word
        modified_context = context
        for word in target_words:
            result = self.token_level_attack(
                word, modified_context, ppl_calc, stealth_calc, config
            )
            if result:
                modified_context = modified_context.replace(word, result, 1)

        return modified_context


if __name__ == "__main__":
    from ppl_calculator import PPLCalculator
    from stealth_calculator import StealthCalculator

    # Initialize components
    ppl_calc = PPLCalculator()
    stealth_calc = StealthCalculator()
    perturber = HardComTokenAttack()

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

    print("Performing promotion attack...")
    adversarial_context = perturber.attack_context(
        context, ppl_calc, stealth_calc, config, num_words=3
    )

    print(f"\nAdversarial context:")
    print(f"  {adversarial_context}")

    stealth = stealth_calc.calculate_stealthiness(context, adversarial_context, method="token")
    print(f"\nStealthiness: {stealth:.3f}")


    