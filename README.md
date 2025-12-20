# Compression-Aware Counterfactual Defense

## Overview
Implementation of compression-aware counterfactual defense against prompt compression attacks, as described by [Liu et al.](https://arxiv.org/abs/2510.22963v1)

## Project Structure
- `compression_aware_defense.py` - Core defense implementation
- `token_level_attack.py` - HardCom token-level attack
- `qa_eval.py` - Question answering task evaluation (SQuAD)
- `integrated_evaluation.py` - Complete defense evaluation framework

## Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

## Running Experiments

### Product Recommendation Attack
```bash
python product_rec_eval.py
```

### QA Attack (SQuAD)
```bash
python qa_eval.py
```

### Defense Evaluation
```bash
python integrated_evaluation.py
```

## Results
- `integrated_evaluation_results.json` - Defense performance metrics
- `squad_qa_attack_results.json` - QA attack results across compression rates

## Requirements
- Python 3.10+
- OpenAI API key
- See requirements.txt for dependencies
```