# ActionPiece Pipeline工作流程详解

本文档详细说明ActionPiece的两种训练工作流程：**原始All-in-One模式**和**新的Two-Step模式**。

---

## 目录
1. [核心组件概述](#核心组件概述)
2. [Workflow 1: All-in-One (main.py)](#workflow-1-all-in-one-mainpy)
3. [Workflow 2: Two-Step (build_vocab.py + train.py)](#workflow-2-two-step-build_vocabpy--trainpy)
4. [Pipeline类详解](#pipeline类详解)
5. [关键差异对比](#关键差异对比)
6. [使用场景推荐](#使用场景推荐)

---

## 核心组件概述

### 1. **genrec/pipeline.py - Pipeline类**
核心管道类，负责编排整个训练/评估流程。

**主要职责：**
- **配置管理**：合并命令行参数、配置文件、默认配置
- **初始化组件**：
  - 数据集加载（Dataset）
  - Tokenizer初始化（包含vocabulary构建/加载）
  - 模型创建（Model）
  - 训练器配置（Trainer）
- **执行训练**：调用`pipeline.run()`启动训练和评估

**关键方法：**
```python
class Pipeline:
    def __init__(model_name, dataset_name, config_dict):
        # 1. 加载配置
        # 2. 初始化dataset
        # 3. 初始化tokenizer (自动构建/加载vocabulary)
        # 4. 初始化model
        # 5. 初始化trainer

    def run():
        # 1. 创建DataLoader
        # 2. 调用trainer.fit()训练模型
        # 3. 加载最佳checkpoint
        # 4. 在测试集上评估
        # 5. 记录结果
```

---

## Workflow 1: All-in-One (main.py)

### 文件：`main.py`

### 执行流程

```
用户命令
    ↓
main.py
    ├─ 解析命令行参数 (--category, --lr, etc.)
    ├─ 配置WandB (可选)
    └─ 创建Pipeline对象
         ↓
    Pipeline.__init__()
         ├─ 加载数据集 (AmazonReviews2014)
         ├─ 初始化Tokenizer
         │    ↓
         │  Tokenizer.__init__()
         │    ├─ 检查vocabulary cache是否存在
         │    ├─ 如果不存在：
         │    │    ├─ 编码sentence embeddings (T5)
         │    │    ├─ 应用PCA降维 (768→128)
         │    │    ├─ 提取item features
         │    │    ├─ 执行BPE-style merging
         │    │    └─ 保存vocabulary到cache
         │    └─ 如果存在：直接加载
         ├─ 初始化Model (T5 Encoder-Decoder)
         └─ 初始化Trainer
              ↓
    Pipeline.run()
         ├─ 创建DataLoader (train/val/test)
         ├─ Trainer.fit()
         │    ├─ 训练循环
         │    ├─ 验证集评估
         │    ├─ Early stopping
         │    └─ 保存最佳checkpoint
         ├─ 加载最佳模型
         ├─ 测试集评估
         └─ 记录最终结果
```

### 适用场景
✅ **首次运行**，没有预先构建的vocabulary
✅ **快速实验**，一次性完成所有步骤
✅ **单次训练**，不需要重复调整超参数

### 优点
- **简单直接**：一个命令完成所有步骤
- **无需额外管理**：不需要关心vocabulary是否已构建

### 缺点
- **重复构建vocabulary**：每次改变超参数都要重新构建vocabulary（耗时！）
- **实验效率低**：超参数调优时浪费时间在vocabulary构建上

### 使用示例
```bash
# 一条命令完成所有操作
CUDA_VISIBLE_DEVICES=0 python main.py \
    --category=Sports_and_Outdoors \
    --lr=0.001 \
    --d_model=256 \
    --rand_seed=42
```

---

## Workflow 2: Two-Step (build_vocab.py + train.py)

这个工作流程将**vocabulary构建**和**模型训练**分离，提高实验效率。

---

### Step 1: 构建Vocabulary - `build_vocab.py`

#### 功能
**只构建vocabulary，不训练模型**。

#### 执行流程

```
用户命令
    ↓
build_vocab.py
    ├─ 解析命令行参数 (--category, --multimodal.*, etc.)
    ├─ 加载配置
    ├─ 初始化Accelerator (用于日志)
    ├─ 加载数据集
    └─ 初始化Tokenizer
         ↓
    Tokenizer.__init__()
         ├─ 检查vocabulary cache
         │    ├─ Cache key = hash(category, multimodal config, sent_emb_pca, etc.)
         │    └─ Cache路径: cache/{category}/processed/actionpiece.json
         ├─ 如果cache不存在：
         │    ├─ _encode_sent_emb()
         │    │    ├─ 提取item metadata (title + description)
         │    │    ├─ 使用sentence-t5编码 (→ 768维)
         │    │    └─ 应用PCA降维 (→ 128维)
         │    ├─ _load_and_fuse_image_embeddings() [如果启用multimodal]
         │    │    ├─ 加载预提取的image embeddings
         │    │    ├─ ASIN-based对齐
         │    │    ├─ 应用PCA到image (→ 128维)
         │    │    ├─ 拼接text+image (→ 256维)
         │    │    └─ Final PCA (→ 128维)
         │    ├─ _get_item2feat()
         │    │    └─ 将embeddings聚类为semantic IDs
         │    └─ ActionPieceCore.train()
         │         ├─ 初始化vocabulary为atomic features
         │         ├─ 迭代BPE-style merging:
         │         │    ├─ 统计feature co-occurrence
         │         │    ├─ 合并高频patterns (within & across items)
         │         │    └─ 更新priority scores
         │         └─ 保存vocabulary到cache
         └─ 如果cache存在：跳过所有构建步骤
              ↓
    报告结果
         ├─ Vocabulary路径
         ├─ Vocabulary大小
         └─ 特征统计
```

#### 关键代码位置
- **Vocabulary构建逻辑**：`genrec/models/ActionPiece/core.py:ActionPieceCore.train()`
- **Sentence embedding**：`genrec/models/ActionPiece/tokenizer.py:_encode_sent_emb()`
- **Multimodal fusion**：`genrec/models/ActionPiece/tokenizer.py:_load_and_fuse_image_embeddings()`

#### 使用示例
```bash
# 构建text-only vocabulary
python build_vocab.py \
    --category=CDs_and_Vinyl \
    --rand_seed=42

# 构建multimodal vocabulary
python build_vocab.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256 \
    --rand_seed=42
```

#### 输出
```
Vocabulary saved to: cache/AmazonReviews2014/CDs_and_Vinyl/processed/actionpiece.json
Vocabulary size: 35672
Number of categories: 8
Number of initial features: 1024
```

---

### Step 2: 训练模型 - `train.py`

#### 功能
**使用预先构建的vocabulary训练模型**。

#### 执行流程

```
用户命令
    ↓
train.py
    ├─ 解析命令行参数 (--category, --lr, --d_model, etc.)
    ├─ 检查vocabulary是否存在
    │    ├─ 路径: cache/{category}/processed/actionpiece.json
    │    ├─ 如果不存在：报错并提示先运行build_vocab.py
    │    └─ 如果存在：继续
    ├─ 配置WandB (可选)
    └─ 创建Pipeline对象
         ↓
    Pipeline.__init__()
         ├─ 加载数据集
         ├─ 初始化Tokenizer
         │    └─ 直接从cache加载vocabulary（跳过构建！）
         ├─ 初始化Model
         └─ 初始化Trainer
              ↓
    Pipeline.run()
         └─ (与main.py相同的训练流程)
```

#### Safety检查
如果vocabulary不存在，train.py会报错：
```
============================================================
ERROR: Vocabulary not found!
============================================================
Category: CDs_and_Vinyl
Expected vocabulary at: cache/.../actionpiece.json

Please build the vocabulary first using:
  python build_vocab.py --category=CDs_and_Vinyl
============================================================
```

#### 使用示例
```bash
# 第一次训练
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --rand_seed=42 \
    --lr=0.001 \
    --d_model=256

# 第二次训练（不同超参，vocabulary直接复用！）
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --rand_seed=42 \
    --lr=0.005 \
    --d_model=512

# 第三次训练（再次复用！）
CUDA_VISIBLE_DEVICES=0 python train.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --rand_seed=43 \
    --lr=0.001 \
    --d_model=256
```

---

## Pipeline类详解

### Pipeline初始化流程

```python
Pipeline(model_name, dataset_name, config_dict)
    ↓
1. 配置合并 (utils.get_config)
    ├─ 默认配置: genrec/default.yaml
    ├─ 数据集配置: genrec/datasets/{DATASET}/config.yaml
    ├─ 模型配置: genrec/models/{MODEL}/config.yaml
    ├─ 自定义配置文件: --config (可选)
    └─ 命令行参数: config_dict (最高优先级)
         ↓
2. 设备初始化
    ├─ 检测可用GPU
    ├─ 设置torch device
    └─ 配置分布式训练 (DDP)
         ↓
3. Accelerator初始化
    ├─ TensorBoard日志
    └─ 分布式训练支持
         ↓
4. 数据集加载
    ├─ 下载原始数据 (首次运行)
    ├─ 处理reviews和metadata
    ├─ Leave-one-out分割
    └─ 保存到cache
         ↓
5. Tokenizer初始化 ⭐关键步骤⭐
    ├─ 计算cache key = hash(配置参数)
    ├─ 检查: cache/{category}/processed/actionpiece.json
    ├─ 如果cache存在:
    │    └─ 加载vocabulary (秒级)
    └─ 如果cache不存在:
         ├─ 构建vocabulary (分钟级)
         └─ 保存到cache
              ↓
6. 模型初始化
    ├─ 创建T5 Encoder-Decoder
    └─ 加载vocabulary配置
         ↓
7. Trainer初始化
    ├─ 配置优化器
    ├─ 配置学习率调度
    └─ 配置early stopping
```

### Pipeline.run() 训练流程

```python
Pipeline.run()
    ↓
1. 创建DataLoader
    ├─ train_dataloader (shuffle=True)
    ├─ val_dataloader (shuffle=False)
    └─ test_dataloader (shuffle=False)
         ↓
2. 模型训练 (Trainer.fit)
    ├─ For each epoch:
    │    ├─ 训练循环 (train_dataloader)
    │    ├─ 验证集评估 (val_dataloader)
    │    │    ├─ 计算NDCG@10, Recall@10, etc.
    │    │    └─ Early stopping检查
    │    └─ 保存最佳checkpoint
    └─ 返回最佳epoch
         ↓
3. 加载最佳模型
    └─ torch.load(best_checkpoint)
         ↓
4. 测试集评估
    ├─ 使用ensemble inference (n_inference_ensemble=5)
    ├─ Beam search (num_beams=50)
    └─ 计算最终metrics
         ↓
5. 记录结果
    ├─ TensorBoard
    ├─ WandB (可选)
    └─ 日志文件
```

---

## 关键差异对比

### 1. Vocabulary构建时机

| Workflow | Vocabulary构建时机 | 是否可复用 |
|----------|-------------------|-----------|
| **All-in-One (main.py)** | 每次运行Pipeline时检查cache，不存在则构建 | ✅ 可复用（通过cache） |
| **Two-Step (build_vocab.py)** | 显式构建，明确分离 | ✅ 可复用 |
| **Two-Step (train.py)** | 不构建，必须预先存在 | ✅ 必须复用 |

### 2. 执行效率对比

假设vocabulary构建耗时：**10分钟**
假设单次训练耗时：**30分钟**

#### 场景：调试3组不同超参数

**All-in-One模式：**
```
Run 1: 10min (vocab) + 30min (train) = 40min
Run 2: 0min (cache hit) + 30min (train) = 30min
Run 3: 0min (cache hit) + 30min (train) = 30min
总计: 100min
```

**Two-Step模式：**
```
Step 1: 10min (build_vocab.py) = 10min
Run 1: 0min + 30min (train.py) = 30min
Run 2: 0min + 30min (train.py) = 30min
Run 3: 0min + 30min (train.py) = 30min
总计: 100min
```

**差异：**
- All-in-one首次运行需要等待vocabulary构建完成
- Two-step可以先构建vocabulary，然后并行运行多个训练任务

### 3. Cache Key机制

Vocabulary cache的唯一性由以下参数决定：
```python
cache_key = hash(
    category,                    # 数据集类别
    sent_emb_pca,               # Sentence embedding PCA维度
    multimodal.enable,          # 是否启用multimodal
    multimodal.image_pca_dim,   # Image PCA维度
    multimodal.final_pca_dim,   # Final PCA维度
    multimodal.fill_strategy,   # 缺失图像填充策略
    # ... 其他影响vocabulary构建的参数
)
```

**重要：** 以下参数**不影响**cache key（可以自由调整）：
- `lr` (学习率)
- `d_model` (模型维度)
- `weight_decay`
- `batch_size`
- `rand_seed`
- ... 所有训练相关参数

---

## 使用场景推荐

### 场景1：首次探索新数据集
**推荐：All-in-One (main.py)**

```bash
python main.py --category=Beauty --lr=0.001
```

**原因：** 简单快速，一次性完成所有步骤。

---

### 场景2：超参数调优 (同一数据集，调整lr, d_model, etc.)
**推荐：Two-Step**

```bash
# Step 1: 构建一次vocabulary
python build_vocab.py --category=CDs_and_Vinyl --rand_seed=42

# Step 2: 并行运行多个训练任务
CUDA_VISIBLE_DEVICES=0 python train.py --category=CDs_and_Vinyl --lr=0.001 &
CUDA_VISIBLE_DEVICES=1 python train.py --category=CDs_and_Vinyl --lr=0.005 &
CUDA_VISIBLE_DEVICES=2 python train.py --category=CDs_and_Vinyl --d_model=512 &
wait
```

**原因：** 避免重复构建vocabulary，提高实验效率。

---

### 场景3：多seed重复实验（论文结果复现）
**推荐：Two-Step**

```bash
# Step 1: 构建vocabulary (seed无关)
python build_vocab.py --category=Sports_and_Outdoors

# Step 2: 运行多个seed
for seed in 2024 2025 2026 2027 2028; do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --category=Sports_and_Outdoors \
        --rand_seed=$seed \
        --lr=0.001
done
```

**原因：** Vocabulary与seed无关，一次构建，多次复用。

---

### 场景4：对比text-only vs multimodal
**推荐：All-in-One (不同cache)**

```bash
# Text-only (cache key不同)
python main.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=false

# Multimodal (cache key不同)
python main.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=128
```

**原因：** 两种配置需要不同的vocabulary，分别构建即可。

---

### 场景5：修改multimodal配置（PCA维度、fusion策略）
**推荐：Two-Step**

```bash
# 配置1: image_pca=128, final_pca=128
python build_vocab.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=128 \
    --multimodal.final_pca_dim=128

python train.py --category=CDs_and_Vinyl --multimodal.* (同上)

# 配置2: image_pca=256, final_pca=256 (需要重新构建vocabulary!)
python build_vocab.py \
    --category=CDs_and_Vinyl \
    --multimodal.enable=true \
    --multimodal.image_pca_dim=256 \
    --multimodal.final_pca_dim=256

python train.py --category=CDs_and_Vinyl --multimodal.* (同上)
```

**原因：** 不同multimodal配置需要重新构建vocabulary。

---

## 总结

### All-in-One (main.py)
✅ **优点：**
- 简单易用，一条命令完成
- 自动管理cache

❌ **缺点：**
- 超参数调优时效率较低（虽然有cache）
- 首次运行需要等待vocabulary构建

### Two-Step (build_vocab.py + train.py)
✅ **优点：**
- 明确分离数据预处理和训练
- 超参数调优时效率高
- 可以并行运行多个训练任务
- 便于调试（分步骤检查）

❌ **缺点：**
- 需要记住两步流程
- 多一个命令

### 推荐策略
- **首次探索**：使用 `main.py`
- **正式实验/调优**：使用 `build_vocab.py` + `train.py`
- **论文复现**：使用 Two-Step（清晰可控）
