#!/usr/bin/env python3

import torch
import torch.nn.functional as functional
from transformers import AutoTokenizer, AutoModel
from typing import Literal, Optional
from dataclasses import dataclass


@dataclass
class SoftComConfig:
    attack_mode: Literal["targeted", "untargeted"] = "targeted"
    num_suffix_tokens: int = 5  # m in paper
    num_epochs: int = 100
    learning_rate: float = 0.01
    device: str = None  # None = auto-detect (tries cuda -> mps -> cpu)


class SoftComSuffixAttack:

    def __init__(
            self,
            compression_model_name: str = "princeton-nlp/AutoCompressor-2.7b-6k",
            device: str = None
    ):
        # auto-detect best device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon GPU
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Loading compression model: {compression_model_name}")
        print(f"Using device: {self.device}")

        # load compression model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(compression_model_name)
        self.compression_model = AutoModel.from_pretrained(compression_model_name).to(self.device)
        self.compression_model.eval()

        # get embedding matrix W_embed
        self.W_embed = self.compression_model.get_input_embeddings().weight
        self.vocab_size = self.W_embed.shape[0]
        self.embed_dim = self.W_embed.shape[1]

        print(f"✓ Model loaded (vocab_size={self.vocab_size}, embed_dim={self.embed_dim})")

    def get_base_embedding(self, text: str) -> torch.Tensor:
        # tokenize text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # get embeddings
        with torch.no_grad():
            embeddings = self.compression_model.get_input_embeddings()(input_ids)

        return embeddings.squeeze(0)  # remove batch dimension

    def generate_suffix_embedding(self, Z: torch.Tensor) -> torch.Tensor:
        # apply softmax to get token distribution
        token_probs = functional.softmax(Z, dim=-1)  # (m, |V|)

        # multiply by embedding matrix
        suffix_embedding = torch.matmul(token_probs, self.W_embed)  # (m, d)

        return suffix_embedding

    def compute_compressed_embedding(
            self,
            E_prime: torch.Tensor,
            raw_input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # pass through compression model
        # for AutoCompressors (decoder-only), we use the model's forward pass
        outputs = self.compression_model(
            inputs_embeds=E_prime.unsqueeze(0),  # Add batch dimension
            output_hidden_states=True
        )

        # get last hidden state as compressed representation
        h = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # Average pool over sequence

        return h

    def compute_loss(
            self,
            h: torch.Tensor,
            h_target: Optional[torch.Tensor],
            h_base: torch.Tensor,
            mode: Literal["targeted", "untargeted"]
    ) -> torch.Tensor:

        if mode == "targeted":
            # L_target = -cos(h, h_target)
            # Minimize cosine distance = maximize similarity
            if h_target is None:
                raise ValueError("h_target required for targeted attack")
            loss = -functional.cosine_similarity(h, h_target, dim=0)
        else:  # untargeted
            # L_nontarget = cos(h, h_base)
            # Maximize cosine distance = minimize similarity
            loss = functional.cosine_similarity(h, h_base, dim=0)

        return loss

    def attack(
            self,
            context: str,
            config: SoftComConfig,
            target_context: Optional[str] = None
    ) -> tuple[torch.Tensor, list[float]]:

        print(f"\nStarting {config.attack_mode} attack...")
        print(f"Suffix tokens: {config.num_suffix_tokens}")
        print(f"Epochs: {config.num_epochs}")

        # Step 1: Get base embedding E_base
        E_base = self.get_base_embedding(context)
        n, d = E_base.shape
        print(f"Base embedding shape: {E_base.shape}")

        # Step 2: Compute base compressed embedding (for untargeted)
        with torch.no_grad():
            h_base = self.compute_compressed_embedding(E_base)

        # Step 3: Get target embedding (for targeted)
        h_target = None
        if config.attack_mode == "targeted":
            if target_context is None:
                raise ValueError("target_context required for targeted attack")
            E_target = self.get_base_embedding(target_context)
            with torch.no_grad():
                h_target = self.compute_compressed_embedding(E_target)

        # Step 4: Initialize learnable logit matrix Z
        # Z ∈ R^(m × |V|)
        Z = torch.randn(
            config.num_suffix_tokens,
            self.vocab_size,
            device=self.device,
            requires_grad=True
        )

        # Step 5: Optimization loop
        optimizer = torch.optim.Adam([Z], lr=config.learning_rate)
        loss_history = []

        for epoch in range(config.num_epochs):
            optimizer.zero_grad()

            # Generate suffix embedding
            E_suffix = self.generate_suffix_embedding(Z)

            # Concatenate: E' = [E_base; E_suffix]
            E_prime = torch.cat([E_base, E_suffix], dim=0)

            # Compress
            h = self.compute_compressed_embedding(E_prime)

            # Compute loss
            loss = self.compute_loss(h, h_target, h_base, config.attack_mode)

            # Backward pass
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {loss.item():.4f}")

        # Step 6: Generate final adversarial embedding
        with torch.no_grad():
            E_suffix_final = self.generate_suffix_embedding(Z)
            E_adversarial = torch.cat([E_base, E_suffix_final], dim=0)

        print(f"\n✓ Attack complete!")
        print(f"Final loss: {loss_history[-1]:.4f}")
        print(f"Adversarial embedding shape: {E_adversarial.shape}")

        return E_adversarial, loss_history


if __name__ == "__main__":
    # Example usage
    config = SoftComConfig(
        attack_mode="targeted",
        num_suffix_tokens=5,
        num_epochs=50,
        learning_rate=0.01
    )

    # Initialize attack
    attacker = SoftComSuffixAttack()

    # Original context
    original_context = "iPhone 16 Pro features a sleek lightweight titanium design"

    # Target context (for targeted attack)
    target_context = "Samsung Galaxy S24 Ultra has excellent performance and display"

    # Perform attack
    adversarial_embedding, loss_history = attacker.attack(
        context=original_context,
        config=config,
        target_context=target_context
    )

    print(f"\nOriginal context: {original_context}")
    print(f"Target context: {target_context}")
    print(f"\nAdversarial embedding created with {config.num_suffix_tokens} learnable suffix tokens")