# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ActionPiece is a generative recommendation model that contextually tokenizes action sequences. Unlike existing generative recommendation models that tokenize each action independently, ActionPiece explicitly incorporates context into action sequence tokenization by merging frequently co-occurring feature patterns within and across adjacent actions.

This is a research implementation from an ICML 2025 Spotlight paper published by Google LLC.

## Development Setup

### Environment

The project requires Python 3.10+ and uses conda for environment management:

```bash
# Create environment from env.yaml
conda env create -f env.yaml
conda activate TIGER

# Key dependencies:
# - PyTorch 2.6.0 with CUDA 12.9
# - Transformers 4.41.2
# - sentence-transformers 3.0.1
# - accelerate 0.31.0
# - faiss-gpu 1.9.0
# - wandb (optional for experiment tracking)
```

### Running Training

**Basic training (standard ActionPiece):**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --category=Sports_and_Outdoors
```

**Training with GRAM features:**
```bash
python main_actionpiece_gram.py \
    --dataset amazon_beauty \
    --gram_enable \
    --deployment_stage 4 \
    --batch_size 128 \
    --learning_rate 5e-4 \
    --gpu 0 \
    --train --eval
```

**Common hyperparameters** (see `genrec/default.yaml`, `genrec/models/ActionPiece/config.yaml`, `genrec/datasets/AmazonReviews2014/config.yaml`):
- `--category`: Dataset category (Beauty, Sports_and_Outdoors, CDs_and_Vinyl)
- `--lr`: Learning rate (0.001-0.005)
- `--weight_decay`: Weight decay (0.07-0.15)
- `--n_hash_buckets`: Number of hash buckets (64-256)
- `--rand_seed`: Random seed for reproducibility (default: 2024)

### Testing

No explicit test suite exists. Validation happens during training via the `Evaluator` class.

## High-Level Architecture

### Core Pipeline Flow

1. **Pipeline** (`genrec/pipeline.py`): Orchestrates the entire training/evaluation workflow
   - Initializes datasets, tokenizer, model, and trainer
   - Manages device configuration and distributed training setup
   - Handles configuration merging from YAML files and command-line arguments

2. **Dataset** (`genrec/dataset.py`): Manages raw data loading and preprocessing
   - Downloads Amazon Reviews 2014 datasets automatically
   - Splits data using leave-one-out strategy
   - Provides metadata (titles, descriptions) for items

3. **Tokenizer** (`genrec/models/ActionPiece/tokenizer.py`): Converts action sequences to tokens
   - Core vocabulary construction in `ActionPieceCore.train()` (`genrec/models/ActionPiece/core.py`)
   - Implements BPE-style merging of frequently co-occurring features
   - Caches constructed vocabularies for reuse
   - Handles Set Permutation Regularization (SPR) during training and inference

4. **Model** (`genrec/models/ActionPiece/model.py`): T5-based sequence-to-sequence architecture
   - Encoder-decoder transformer for next-item prediction
   - Beam search generation with ensemble inference
   - Configurable via `genrec/models/ActionPiece/config.yaml`

5. **Trainer** (`genrec/trainer.py`): Manages training loop
   - Handles gradient accumulation, checkpointing, early stopping
   - Integrates with WandB for experiment tracking (optional)
   - Uses Hugging Face Accelerate for distributed training

6. **Evaluator** (`genrec/evaluator.py`): Computes recommendation metrics
   - NDCG@k, Recall@k, ERR@k

### Key Implementation Details

**Vocabulary Construction** (`genrec/models/ActionPiece/core.py:ActionPieceCore.train()`):
- Starts with atomic item features as initial vocabulary
- Iteratively merges frequent feature co-occurrences (within items and across adjacent items)
- Assigns priorities based on co-occurrence frequency
- Saves vocabulary to cache for reuse

**Segmentation with SPR**:
- **Training** (`ActionPieceTokenizer.collate_fn_train()`): Randomly permutes feature order within sets to encourage order-invariance
- **Inference** (`ActionPieceTokenizer.collate_fn_test()`, `ActionPiece.generate()`): Uses greedy segmentation to maximize token priority

**GRAM Features Extension** (experimental, in `gram` branch):
- Adds semantic embeddings (T5), TF-IDF keywords, and collaborative filtering signals
- Four deployment stages for gradual feature integration
- Preprocessing via `scripts/preprocess_gram_features.py`

### Configuration System

Configuration is hierarchical with precedence order (highest to lowest):
1. Command-line arguments
2. Custom config file (via `--config`)
3. Model config (`genrec/models/{MODEL}/config.yaml`)
4. Dataset config (`genrec/datasets/{DATASET}/config.yaml`)
5. Default config (`genrec/default.yaml`)

The `genrec.utils.get_config()` function merges these layers.

## Important File Paths

### Core Implementation
- **Vocabulary construction**: `genrec/models/ActionPiece/core.py:ActionPieceCore.train()`
- **Tokenization with SPR**:
  - Training: `genrec/models/ActionPiece/tokenizer.py:ActionPieceTokenizer.collate_fn_train()`
  - Inference: `genrec/models/ActionPiece/tokenizer.py:ActionPieceTokenizer.collate_fn_test()` and `genrec/models/ActionPiece/model.py:ActionPiece.generate()`

### Cached Artifacts
- **Tokenizer vocabularies**: `cache/AmazonReviews2014/{category}/tokenizer/{hash}.json`
  - Structure: `{"n_categories": int, "n_init_feats": int, "token2feat": list, "priority": list}`
  - `token2feat` format: `[a, b]` = b-th choice of a-th feature; `[-1, u, v]` = merge of tokens u and v
- **Processed datasets**: `cache/AmazonReviews2014/{category}/processed/`
- **Model checkpoints**: `ckpt/` (configured via `--ckpt_dir`)
- **Logs**: `logs/` (configured via `--log_dir`)

## Working with GRAM Features

GRAM (Generative Recommendation with Augmented Metadata) is an experimental extension that adds richer item representations:

### Deployment Stages
1. **Stage 1**: Baseline (no GRAM features)
2. **Stage 2**: + T5 semantic embeddings (clustered)
3. **Stage 3**: + TF-IDF keywords
4. **Stage 4**: + Collaborative filtering signals

### Prerequisites
Run feature preprocessing before training with GRAM:
```bash
python scripts/preprocess_gram_features.py \
    --dataset_path ./cache/AmazonReviews2014/Beauty \
    --output_path ./cache/AmazonReviews2014/Beauty/processed/gram_features.json
```

### Configuration
Set `gram_features.enable: true` in `genrec/models/ActionPiece/config.yaml` or pass via command-line:
```bash
--gram_enable --deployment_stage 4
```

## Logging and Experiment Tracking

### TensorBoard
Enabled by default. Logs saved to `tensorboard/`:
```bash
tensorboard --logdir tensorboard/
```

### Weights & Biases
Optional. Enable with:
```bash
python main.py --use_wandb --wandb_project "my-project" --wandb_entity "my-team"
```

Requires `wandb login` before first use.

## Reproducibility

The paper results use seeds 2024-2028 for 5 runs. Set via:
```bash
--rand_seed=2024
```

Deterministic behavior is enforced when `reproducibility: true` in config (default).

## Branch Structure

- **main**: Stable release with standard ActionPiece
- **gram**: Experimental branch with GRAM feature integration

## Notes for Development

- **Tokenizer caching**: Vocabularies are cached by a hash of construction parameters. Changing feature extraction or merging logic requires clearing the cache manually.
- **Dataset auto-download**: First run automatically downloads Amazon Reviews 2014 data to `cache/`. Expect ~1-5GB per category.
- **Memory requirements**: Training requires ~16GB GPU memory for default hyperparameters. Reduce `batch_size` or `d_model` if OOM occurs.
- **Distributed training**: Use `CUDA_VISIBLE_DEVICES=0,1,2,3` with `--distributed` flag. Not required for single GPU.
- **Early stopping**: Controlled by `patience` parameter (default: 20 epochs without improvement on `val_metric`).
