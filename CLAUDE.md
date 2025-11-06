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

ActionPiece supports two workflows:

#### Workflow 1: All-in-One (Original)
Build vocabulary and train in a single step:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --category=Sports_and_Outdoors
```

#### Workflow 2: Two-Step (Recommended for Experimentation)
**Step 1: Build vocabulary once**
```bash
# Build vocabulary for a category (only needs to be done once)
python build_vocab.py --category=CDs_and_Vinyl --rand_seed=42
```

**Step 2: Train multiple times with different hyperparameters**
```bash
# First training run
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --rand_seed=42 \
    --lr=0.001 \
    --d_model=256

# Second training run (vocabulary already built, just train)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --rand_seed=43 \
    --lr=0.005 \
    --d_model=512
```

**Advantages of two-step workflow:**
- Vocabulary construction can be time-consuming (especially for large datasets)
- Build vocabulary once, experiment with different training hyperparameters multiple times
- Clearer separation of data preprocessing and model training

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

**Two-step workflow with multimodal features:**
```bash
# Step 1: Build vocabulary with multimodal features
python build_vocab.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --rand_seed=42

# Step 2: Train with different hyperparameters
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --rand_seed=42 \
    --lr=0.001 \
    --d_model=256
```

**Common hyperparameters** (see `genrec/default.yaml`, `genrec/models/ActionPiece/config.yaml`, `genrec/datasets/AmazonReviews2014/config.yaml`):
- `--category`: Dataset category (Beauty, Sports_and_Outdoors, CDs_and_Vinyl)
- `--lr`: Learning rate (0.001-0.005)
- `--weight_decay`: Weight decay (0.07-0.15)
- `--n_hash_buckets`: Number of hash buckets (64-256)
- `--rand_seed`: Random seed for reproducibility (default: 2024)
- `--multimodal.enable`: Enable/disable multimodal features (true/false)
- `--multimodal.image_pca_dim`: Image embedding PCA dimension (default: 128)
- `--multimodal.final_pca_dim`: Final fused embedding dimension (default: 128)

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

### Entry Points
- **`main.py`**: All-in-one training (builds vocabulary + trains model)
- **`build_vocab.py`**: Build ActionPiece vocabulary only (no training)
- **`train.py`**: Train model using pre-built vocabulary

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

## Multimodal Configuration

ActionPiece supports integrating image embeddings with text embeddings for richer item representations. This feature can be configured through the configuration system.

### Configuration Options

All multimodal settings are in `genrec/models/ActionPiece/config.yaml` under the `multimodal` section:

```yaml
multimodal:
  enable: true  # Enable/disable multimodal fusion
  image_path_template: "/path/to/{category}/{category}.emb-ViT-L-14.npy"
  image_pca_dim: 128  # PCA dimension for image embeddings
  final_pca_dim: 128  # Final dimension after text+image fusion (0 to disable)
  fill_strategy: "zero"  # How to handle missing images: "zero" or "mean"
  # Complete category mapping for Amazon 2018 dataset (full name -> short name)
  category_mapping:
    All_Beauty: "Beauty"
    CDs_and_Vinyl: "CDs"
    Sports_and_Outdoors: "Sports"
    # ... (29 categories total, see config.yaml for full list)
```

The `category_mapping` automatically converts full category names (e.g., `CDs_and_Vinyl`) to short names (e.g., `CDs`) for constructing image file paths. This supports all 29 Amazon 2018 categories.

### How Category Mapping Works

When you run:
```bash
python main.py --category=CDs_and_Vinyl
```

The system automatically:
1. Detects full category name: `CDs_and_Vinyl`
2. Maps it to short name via `category_mapping`: `CDs`
3. Constructs image path: `/path/to/CDs/CDs.emb-ViT-L-14.npy`

If a category is not in the mapping, it defaults to the first part before underscore (e.g., `New_Category` → `New`).

### Command-line Override

You can override multimodal settings via command-line using dot notation:

```bash
# Enable multimodal with custom dimensions
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256 \
    --lr=0.001 \
    --d_model=256

# Disable multimodal (text-only mode)
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=false \
    --lr=0.001 \
    --d_model=256

# Change fill strategy for missing images
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors \
    --multimodal.fill_strategy=mean \
    --rand_seed=42
```

The dot notation (`--multimodal.enable`) allows you to override specific nested configuration values without modifying the config file.

### Image Embedding Format

Image embeddings should be saved as `.npy` files in dictionary format:

```python
{
    'asins': ['B001...', 'B002...', ...],  # List of ASINs
    'embeddings': np.array([...])  # Shape: (N_items, embedding_dim)
}
```

The tokenizer automatically aligns image embeddings with text embeddings using ASIN matching.

### Missing Images Handling

- **`fill_strategy: "zero"`**: Use zero vectors for items without images
- **`fill_strategy: "mean"`**: Use mean of available image embeddings

### Cached Files

Multimodal processing creates cached files in `cache/AmazonReviews2014/{category}/processed/`:
- `image_pca_{dim}_{strategy}.npy`: PCA-reduced image embeddings
- `multimodal_final_pca_{dim}_{strategy}.npy`: Final fused embeddings

Clear these caches if you change image embeddings or fusion parameters.

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

## Complete Workflow Examples

### Example 1: Quick Start (All-in-One)
```bash
# Single command: build vocab + train
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=CDs_and_Vinyl \
    --rand_seed=42 \
    --lr=0.001
```

### Example 2: Hyperparameter Tuning (Two-Step)
```bash
# Step 1: Build vocabulary once (takes time)
python build_vocab.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256 \
    --rand_seed=42

# Step 2a: First training run
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256 \
    --rand_seed=42 \
    --lr=0.001 \
    --d_model=256 \
    --d_ff=2048

# Step 2b: Second training run (different lr, no vocab rebuild)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256 \
    --rand_seed=42 \
    --lr=0.005 \
    --d_model=256 \
    --d_ff=2048

# Step 2c: Third training run (different model size)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256 \
    --rand_seed=42 \
    --lr=0.001 \
    --d_model=512 \
    --d_ff=4096
```

### Example 3: Compare Text-Only vs Multimodal
```bash
# Build vocabulary with multimodal
python build_vocab.py \
    --category=Sports_and_Outdoors \
    --multimodal.enable=true

# Train with multimodal
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=Sports_and_Outdoors \
    --multimodal.enable=true \
    --rand_seed=42

# Train without multimodal (text-only, uses same vocab structure)
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=Sports_and_Outdoors \
    --multimodal.enable=false \
    --rand_seed=42
```

## Notes for Development

- **Tokenizer caching**: Vocabularies are cached by a hash of construction parameters. Changing feature extraction or merging logic requires clearing the cache manually.
- **Dataset auto-download**: First run automatically downloads Amazon Reviews 2014 data to `cache/`. Expect ~1-5GB per category.
- **Memory requirements**: Training requires ~16GB GPU memory for default hyperparameters. Reduce `batch_size` or `d_model` if OOM occurs.
- **Distributed training**: Use `CUDA_VISIBLE_DEVICES=0,1,2,3` with `--distributed` flag. Not required for single GPU.
- **Early stopping**: Controlled by `patience` parameter (default: 20 epochs without improvement on `val_metric`).
- **Vocabulary reuse**: Once built, the vocabulary file (`cache/.../processed/actionpiece.json`) can be reused for multiple training runs with the same category and feature configuration.
