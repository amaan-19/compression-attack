#!/usr/bin/env python3
"""
Question Answering (QA) Attack Evaluator for CompressionAttack
Implements untargeted preference manipulation as described in the paper.

Key difference from product recommendation:
- Product Rec: Targeted manipulation (steer toward specific product)
- QA: Untargeted manipulation (cause information loss → wrong/no answer)

Attack strategy:
- Perturb high-PPL words throughout the ENTIRE context
- Goal: Maximize information loss during compression
- Success: Answer changes from correct to incorrect/unavailable
"""

import re
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
import random
import string
from datasets import load_dataset


class OpenAIBackend:
    def __init__(self, api_key=None):
        self.client = openai.OpenAI(api_key=api_key)
        print("Using OpenAI API")
    
    def query(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()


class QAEvaluator:
    """Evaluator for QA attacks using prompt compression."""
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize with a surrogate model for PPL calculation."""
        print(f"Loading surrogate model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize OpenAI backend
        self.backend = OpenAIBackend()
        
        print("QA Evaluator initialized.")
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of text using surrogate model."""
        encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings['input_ids'])
            loss = outputs.loss
            
        return torch.exp(loss).item()
    
    def get_high_ppl_words(self, text: str, top_k: int = 15) -> List[Tuple[str, int, float]]:
        """
        Identify high-PPL words in the text.
        Returns list of (word, position, ppl) tuples.
        
        Note: Attacks ENTIRE context uniformly, not targeting specific content.
        """
        words = text.split()
        word_ppls = []
        
        for idx, word in enumerate(words):
            # Skip very short words and punctuation
            if len(word) <= 2 or word in string.punctuation:
                continue
            
            # Calculate PPL for this word
            ppl = self.calculate_perplexity(word)
            word_ppls.append((word, idx, ppl))
        
        # Sort by PPL (highest first) and return top_k
        word_ppls.sort(key=lambda x: x[2], reverse=True)
        return word_ppls[:top_k]
    
    def apply_token_level_attack(
        self,
        text: str,
        target_words: List[Tuple[str, int, float]],
        max_iterations: int = 20,
        stealth_threshold: float = 0.6
    ) -> str:
        """
        Apply token-level perturbations (character substitutions).
        Goal: Increase PPL to cause token to be dropped during compression.
        """
        words = text.split()
        
        for word, pos, orig_ppl in target_words:
            if pos >= len(words):
                continue
            
            best_candidate = words[pos]
            best_ppl = orig_ppl
            
            # Try various character-level perturbations
            for _ in range(max_iterations):
                candidate = self._perturb_word_chars(words[pos])
                
                # Check stealthiness
                if not self._is_stealthy(words[pos], candidate, stealth_threshold):
                    continue
                
                # Check if PPL increased
                candidate_ppl = self.calculate_perplexity(candidate)
                if candidate_ppl > best_ppl:
                    best_candidate = candidate
                    best_ppl = candidate_ppl
            
            words[pos] = best_candidate
        
        return ' '.join(words)
    
    def _perturb_word_chars(self, word: str) -> str:
        """Apply character-level perturbations to a word."""
        if len(word) < 2:
            return word
        
        strategies = [
            lambda w: w.replace('o', '0'),
            lambda w: w.replace('i', '1'),
            lambda w: w.replace('e', '3'),
            lambda w: w.replace('a', '@'),
            lambda w: w.replace('s', '$'),
            lambda w: w[:len(w)//2] + ' ' + w[len(w)//2:],  # Insert space
            lambda w: w + "'",  # Add apostrophe
            lambda w: w.replace('l', 'I'),
        ]
        
        strategy = random.choice(strategies)
        try:
            return strategy(word)
        except:
            return word
    
    def _is_stealthy(self, original: str, perturbed: str, threshold: float) -> bool:
        """Check if perturbation is stealthy enough."""
        # Character-level similarity
        from difflib import SequenceMatcher
        char_sim = SequenceMatcher(None, original.lower(), perturbed.lower()).ratio()
        
        return char_sim >= threshold
    
    def apply_word_level_attack(
        self,
        text: str,
        target_words: List[Tuple[str, int, float]],
        max_iterations: int = 20,
        stealth_threshold: float = 0.6
    ) -> str:
        """
        Apply word-level perturbations (punctuation, modifiers).
        Goal: Increase PPL to cause word to be dropped during compression.
        """
        words = text.split()
        
        for word, pos, orig_ppl in target_words:
            if pos >= len(words):
                continue
            
            best_candidate = words[pos]
            best_ppl = orig_ppl
            
            # Try various word-level perturbations
            for _ in range(max_iterations):
                candidate = self._perturb_word_structure(words[pos])
                
                # Check stealthiness (more lenient for word-level)
                if len(candidate) > len(words[pos]) * 2:
                    continue
                
                # Check if PPL increased
                candidate_ppl = self.calculate_perplexity(candidate)
                if candidate_ppl > best_ppl:
                    best_candidate = candidate
                    best_ppl = candidate_ppl
            
            words[pos] = best_candidate
        
        return ' '.join(words)
    
    def _perturb_word_structure(self, word: str) -> str:
        """Apply structural perturbations to a word."""
        strategies = [
            lambda w: f'"{w}"',  # Add quotes
            lambda w: f'{w}...',  # Add ellipsis
            lambda w: f'{w},',   # Add comma
            lambda w: f'{w};',   # Add semicolon
            lambda w: f'({w})',  # Add parentheses
            lambda w: f'{w}!',   # Add exclamation
        ]
        
        strategy = random.choice(strategies)
        return strategy(word)
    
    def compress_text(self, text: str, compression_rate: float = 0.5) -> str:
        """
        Simulate hard prompt compression.
        Keeps tokens with lowest PPL (most predictable = least important).
        """
        words = text.split()
        target_length = int(len(words) * compression_rate)
        
        # Calculate PPL for each word
        word_ppls = []
        for word in words:
            ppl = self.calculate_perplexity(word)
            word_ppls.append((word, ppl))
        
        # Keep words with LOWEST PPL (compression keeps predictable content)
        word_ppls.sort(key=lambda x: x[1])
        kept_words = [w for w, _ in word_ppls[:target_length]]
        
        return ' '.join(kept_words)
    
    def query_llm(self, context: str, question: str) -> str:
        """Query LLM with context and question using OpenAI backend."""
        prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}

Answer:"""
        
        try:
            # Use OpenAI backend (initialized in __init__)
            response = self.backend.query(prompt)
            return self._extract_answer(response)
        
        except Exception as e:
            print(f"LLM query error: {e}")
            return "ERROR"
    
    def _extract_answer(self, text: str) -> str:
        """Extract clean answer from LLM response."""
        # Remove common prefixes
        text = re.sub(r'^(Answer:|A:|The answer is:?)\s*', '', text, flags=re.IGNORECASE)
        
        # Get first sentence or line
        text = text.split('\n')[0].split('.')[0].strip()
        
        return text
    
    def evaluate_qa_attack(
        self,
        context: str,
        question: str,
        correct_answer: str,
        compression_rate: float = 0.5,
        num_words_to_attack: int = 15,
        attack_strategy: str = "hybrid"
    ) -> Dict:
        """
        Evaluate QA attack using untargeted preference manipulation.
        
        Args:
            context: Background text containing the answer
            question: Question to ask
            correct_answer: Ground truth answer
            compression_rate: Compression ratio (0.5 = keep 50% of tokens)
            num_words_to_attack: Number of high-PPL words to perturb
            attack_strategy: "token", "word", or "hybrid"
        
        Returns:
            Dictionary with attack results
        """
        print("\n" + "="*80)
        print("QA ATTACK EVALUATION")
        print("="*80)
        
        # Step 1: Get baseline (original answer)
        print("\n[1] Compressing original context...")
        compressed_original = self.compress_text(context, compression_rate)
        print(f"Original context length: {len(context.split())} words")
        print(f"Compressed length: {len(compressed_original.split())} words")
        
        print("\n[2] Querying LLM with original compressed context...")
        original_answer = self.query_llm(compressed_original, question)
        print(f"Original answer: {original_answer}")
        print(f"Correct answer: {correct_answer}")
        
        # Step 2: Apply attack to context
        print("\n[3] Identifying high-PPL words in context...")
        high_ppl_words = self.get_high_ppl_words(context, num_words_to_attack)
        print(f"Attacking {len(high_ppl_words)} high-PPL words")
        
        print("\n[4] Applying adversarial perturbations...")
        if attack_strategy == "token":
            adversarial_context = self.apply_token_level_attack(context, high_ppl_words)
        elif attack_strategy == "word":
            adversarial_context = self.apply_word_level_attack(context, high_ppl_words)
        else:  # hybrid
            adversarial_context = self.apply_token_level_attack(context, high_ppl_words)
            high_ppl_adv = self.get_high_ppl_words(adversarial_context, num_words_to_attack // 2)
            adversarial_context = self.apply_word_level_attack(adversarial_context, high_ppl_adv)
        
        # Step 3: Compress adversarial context
        print("\n[5] Compressing adversarial context...")
        compressed_adversarial = self.compress_text(adversarial_context, compression_rate)
        print(f"Adversarial context length: {len(adversarial_context.split())} words")
        print(f"Compressed length: {len(compressed_adversarial.split())} words")
        
        # Step 4: Get adversarial answer
        print("\n[6] Querying LLM with adversarial compressed context...")
        adversarial_answer = self.query_llm(compressed_adversarial, question)
        print(f"Adversarial answer: {adversarial_answer}")
        
        # Step 5: Evaluate
        answer_changed = adversarial_answer.lower() != original_answer.lower()
        success = answer_changed  # Success = any answer change
        
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Answer changed: {answer_changed}")
        print(f"Attack success: {success}")
        print("="*80)
        
        return {
            "success": success,
            "answer_changed": answer_changed,
            "original_answer": original_answer,
            "adversarial_answer": adversarial_answer,
            "correct_answer": correct_answer,
            "compression_rate": compression_rate,
            "num_words_attacked": len(high_ppl_words)
        }


def load_squad_examples(num_examples: int = 50, seed: int = 42) -> List[Dict]:
    """
    Load examples from SQuAD dataset.
    
    Args:
        num_examples: Number of examples to sample
        seed: Random seed for reproducibility
    
    Returns:
        List of dictionaries with context, question, and answer
    """
    print(f"Loading {num_examples} examples from SQuAD dataset...")
    
    # Load SQuAD validation set
    dataset = load_dataset("squad", split="validation")
    
    # Sample examples
    sampled = dataset.shuffle(seed=seed).select(range(num_examples))
    
    # Convert to our format
    examples = []
    for i, item in enumerate(sampled):
        examples.append({
            "id": f"squad_{i}",
            "context": item["context"],
            "question": item["question"],
            "correct_answer": item["answers"]["text"][0],  # Take first answer
            "title": item.get("title", f"Example {i}")
        })
    
    print(f"Loaded {len(examples)} examples from SQuAD")
    return examples


# Load SQuAD examples by default
QA_EXAMPLES = load_squad_examples(num_examples=50)


if __name__ == "__main__":
    # Simple test with first SQuAD example
    evaluator = QAEvaluator()
    
    example = QA_EXAMPLES[0]
    
    print(f"\nTesting with SQuAD example: {example['title']}")
    print(f"Question: {example['question']}")
    print(f"Correct answer: {example['correct_answer']}")
    
    result = evaluator.evaluate_qa_attack(
        context=example["context"],
        question=example["question"],
        correct_answer=example["correct_answer"],
        compression_rate=0.3,  # Aggressive compression
        num_words_to_attack=15,
        attack_strategy="hybrid"
    )
    
    print("\nFinal Result:")
    print(f"Success: {result['success']}")
    print(f"Original: {result['original_answer']}")
    print(f"Adversarial: {result['adversarial_answer']}")