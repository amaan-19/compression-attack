import numpy as np
import Levenshtein
from typing import Literal

class StealthCalculator:
    def __init__(self, sentence_model=None):
        self.sentence_encoder = sentence_model
        
        # Try to load sentence transformer if not provided
        if self.sentence_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                self.sentence_encoder = None

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
        if self.sentence_encoder is None:
            # Fallback: use character-level similarity
            return self._edit_distance_similarity(s1, s2)
        
        emb1 = self.sentence_encoder.encode(s1)
        emb2 = self.sentence_encoder.encode(s2)

        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cos_sim)

    def _bert_score_similarity(self, s1: str, s2: str) -> float:
        # Simplified: use edit distance as fallback
        return self._edit_distance_similarity(s1, s2)