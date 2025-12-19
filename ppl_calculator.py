import torch
from typing import List, Tuple

class PPLCalculator:
    def __init__(self, model, tokenizer):
        self.device = "cpu"
        self.model = model
        self.model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer

    def calculate_perplexity(self, text: str) -> float:
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
            full_text = ' '.join(words)
            text_without = ' '.join(words[:i] + words[i+1:])

            ppl_with = self.calculate_perplexity(full_text)
            ppl_without = self.calculate_perplexity(text_without) if text_without else ppl_with
            word_ppl = abs(ppl_with - ppl_without)

            word_ppls.append((word, word_ppl))

        word_ppls.sort(key=lambda x: x[1], reverse=True)
        return word_ppls


# === Exampe Usage ===
# if __name__ == "__main__":
#     calc = PPLCalculator()
#
#     text = "iPhone 16 Pro features a sleek lightweight titanium design"
#
#     print(f"Text: {text}")
#     print(f"Overall PPL: {calc.calculate_perplexity(text):.2f}")
#
#     print("Top 5 high-PPL words (attack targets):")
#     for word, ppl in calc.calculate_word_ppls(text)[:5]:
#         print(f"    {word}: {ppl:.2f}")