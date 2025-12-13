#!/usr/bin/env python3

import numpy as np
from sentence_transformers import SentenceTransformer
import Levenshtein
from typing import Literal

class StealthCalculator:
    def __init__(self, sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.sentence_encoder = SentenceTransformer(sentence_model)

    def calculate_stealthiness(
            self,
            original: str,
            modified: str,
            method: Literal["token", "word"] = "token",
            lambda_weight: float = 0.5
    ) -> float:

        # semantic similarity (used in both methods)
        semantic_sim = self._cosine_similarity(original, modified)

        if method == "token":
            # token-level: edit distance + semantic similarity
            surface_sim = self._edit_distance_similarity(original, modified)
        elif method == "word":
            # word-level: BERTScore + semantic similarity
            surface_sim = self._bert_score_similarity(original, modified)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'token' or 'word'")

        # combine both metrics
        stealth = lambda_weight * surface_sim + (1 - lambda_weight) * semantic_sim
        return stealth

    def _edit_distance_similarity(self, s1: str, s2: str) -> float:
        edit_dist = Levenshtein.distance(s1, s2)
        max_len = max(len(s1), len(s2))

        if max_len == 0:
            return 1.0

        # convert distance to similarity (1 = identical, 0 = completely different)
        similarity = 1 - (edit_dist / max_len)
        return similarity

    def _cosine_similarity(self, s1: str, s2: str) -> float:
        emb1 = self.sentence_encoder.encode(s1)
        emb2 = self.sentence_encoder.encode(s2)

        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cos_sim)

    def _bert_score_similarity(self, s1: str, s2: str) -> float:
        try:
            from bert_score import score

            # calculate BERTScore (returns P, R, F1)
            P, R, F1 = score([s2], [s1], lang="en", verbose=False)
            return F1.item()

        except ImportError:
            print("Warning: bert-score not installed. Using edit distance instead.")
            print("Install with: pip install bert-score")
            return self._edit_distance_similarity(s1, s2)


# === Usage Example ===
# if __name__ == "__main__":
#
#     calc = StealthCalculator()
#
#     original = "iPhone 16 Pro features a sleek lightweight titanium design"
#
#     # small perturbation (should be stealthy)
#     subtle = "iPhone 16 Pro features a sleek lightweight tit@nium design"
#
#     # large perturbation (should be less stealthy)
#     obvious = "iPhone 16 Pro xyz random words titanium"
#
#     print("Stealthiness Scores (higher = more stealthy):\n")
#
#     print("Token-level method (edit distance):")
#     print(f"  Subtle change: {calc.calculate_stealthiness(original, subtle, method='token'):.3f}")
#     print(f"  Obvious change: {calc.calculate_stealthiness(original, obvious, method='token'):.3f}")
#     print()
#
#     print("Word-level method (BERTScore):")
#     print(f"  Subtle change: {calc.calculate_stealthiness(original, subtle, method='word'):.3f}")
#     print(f"  Obvious change: {calc.calculate_stealthiness(original, obvious, method='word'):.3f}")