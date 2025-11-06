# Quick Start Guide - Two-Step Workflow

## 🚀 Build Vocabulary Once, Train Multiple Times

### Step 1: Build Vocabulary (Run Once)

```bash
python build_vocab.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256 \
    --rand_seed=42
```

**Expected Output:**
```
============================================================
Building ActionPiece Vocabulary
============================================================
[DATASET] Loading dataset: AmazonReviews2014
[TOKENIZER] Generating item features...
[TOKENIZER] Constructing ActionPiece vocabulary...
============================================================
Vocabulary Construction Complete!
============================================================
Vocabulary saved to: cache/AmazonReviews2014/CDs_and_Vinyl/processed/actionpiece.json
Vocabulary size: 40000
Number of categories: 5
Number of initial features: 256
============================================================

✓ Vocabulary built successfully!
✓ Saved to: cache/AmazonReviews2014/CDs_and_Vinyl/processed/actionpiece.json

You can now train the model using:
  python train.py --category=CDs_and_Vinyl [other args...]
```

### Step 2: Train Multiple Times (Use Pre-built Vocabulary)

**Experiment 1: Baseline**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --rand_seed=42 \
    --weight_decay=0.07 \
    --lr=0.001 \
    --d_model=256 \
    --d_ff=2048 \
    --n_hash_buckets=256 \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256
```

**Experiment 2: Higher Learning Rate**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --rand_seed=42 \
    --weight_decay=0.07 \
    --lr=0.005 \
    --d_model=256 \
    --d_ff=2048 \
    --n_hash_buckets=256 \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256
```

**Experiment 3: Larger Model**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --rand_seed=42 \
    --weight_decay=0.07 \
    --lr=0.001 \
    --d_model=512 \
    --d_ff=4096 \
    --n_hash_buckets=256 \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256
```

**Experiment 4: Text-Only (No Multimodal)**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --rand_seed=42 \
    --weight_decay=0.07 \
    --lr=0.001 \
    --d_model=256 \
    --d_ff=2048 \
    --n_hash_buckets=256 \
    --multimodal.enable=false
```

## 📊 Complete Hyperparameter Grid Search

```bash
# Build vocabulary once
python build_vocab.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256

# Grid search over learning rates and model sizes
for lr in 0.001 0.003 0.005; do
  for d_model in 256 512; do
    d_ff=$((d_model * 8))

    CUDA_VISIBLE_DEVICES=0 python train.py \
        --category=CDs_and_Vinyl \
        --rand_seed=42 \
        --lr=$lr \
        --d_model=$d_model \
        --d_ff=$d_ff \
        --multimodal.enable=true \
        --multimodal.image_pca_dim=256 \
        --multimodal.final_pca_dim=256
  done
done
```

## ⚠️ Important Notes

1. **Vocabulary must match training settings**: If you change `multimodal.image_pca_dim` or `multimodal.final_pca_dim`, you need to rebuild the vocabulary.

2. **Category-specific vocabulary**: Each category needs its own vocabulary. Build separately:
   ```bash
   python build_vocab.py --category=CDs_and_Vinyl ...
   python build_vocab.py --category=Sports_and_Outdoors ...
   python build_vocab.py --category=Beauty ...
   ```

3. **Vocabulary location**:
   ```
   cache/AmazonReviews2014/{category}/processed/actionpiece.json
   ```

4. **Check vocabulary exists**: `train.py` will automatically check if vocabulary exists and give helpful error message if not found.

## 🔧 Troubleshooting

**Error: "Vocabulary not found!"**
```
Solution: Run build_vocab.py first for that category
```

**Error: "KeyError: 'accelerator'"**
```
Solution: This has been fixed. Make sure you're using the updated build_vocab.py
```

**Different results between runs:**
```
Solution: Use the same --rand_seed value for reproducible results
```

## 📈 All-in-One Alternative

If you prefer the original workflow (build vocab + train in one step):
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=CDs_and_Vinyl \
    --rand_seed=42 \
    --lr=0.001 \
    --multimodal.enable=true
```

This is slower for hyperparameter tuning but simpler for single runs.
