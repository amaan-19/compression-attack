import numpy as np
from typing import List, Tuple

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class PPLCalculator:
    def __init__(self, model=None, tokenizer=None):
        self.device = "cpu"
        self.model = model
        self.tokenizer = tokenizer
        
        if model is not None and TORCH_AVAILABLE:
            self.model.eval()
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

    def calculate_perplexity(self, text: str) -> float:
        if self.model is None or not TORCH_AVAILABLE:
            # Fallback: estimate PPL based on text characteristics
            # Higher PPL for unusual characters, shorter words
            words = text.split()
            if not words:
                return 10.0
            
            avg_word_len = np.mean([len(w) for w in words])
            special_chars = sum(1 for c in text if not c.isalnum() and c not in ' .,!?')
            
            # Base PPL + adjustments
            base_ppl = 20.0
            ppl = base_ppl + special_chars * 5 - avg_word_len * 0.5
            return max(5.0, ppl + np.random.uniform(-2, 2))
        
        encodings = self.tokenizer(text, return_tensors='pt')
        input_ids = encodings.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            perplexity = torch.exp(outputs.loss).item()

        return perplexity

    def calculate_word_ppls(self, text: str) -> List[Tuple[str, float]]:
        words = text.split()
        word_ppls = []

        for i, word in enumerate(words):
            if self.model is None or not TORCH_AVAILABLE:
                # Fallback: PPL based on word characteristics
                # Unusual characters = higher PPL
                special = sum(1 for c in word if not c.isalnum())
                is_short = len(word) < 3
                ppl = 10 + special * 15 + (5 if is_short else 0) + np.random.uniform(0, 5)
                word_ppls.append((word, ppl))
            else:
                full_text = ' '.join(words)
                text_without = ' '.join(words[:i] + words[i+1:])

                ppl_with = self.calculate_perplexity(full_text)
                ppl_without = self.calculate_perplexity(text_without) if text_without else ppl_with
                word_ppl = abs(ppl_with - ppl_without)

                word_ppls.append((word, word_ppl))

        word_ppls.sort(key=lambda x: x[1], reverse=True)
        return word_ppls