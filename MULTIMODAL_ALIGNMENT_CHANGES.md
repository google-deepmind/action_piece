# 多模态ID对齐修改总结

## 修改概览

根据`MULTIMODAL_ID_ALIGNMENT_GUIDE.md`的指导，已完成以下修改以实现基于ASIN的精确对齐。

---

## 主要修改

### 1. 修改了 `genrec/models/ActionPiece/tokenizer.py`

#### 新增函数：`_load_external_asins()` (第114-171行)
- 功能：从外部文件加载ASIN信息
- 支持3种格式：
  - `.asins.txt` (每行一个ASIN)
  - `.asins.json` (JSON数组)
  - `.config.json` (包含'asins'键的配置文件)

#### 重写函数：`_load_and_fuse_image_embeddings()` (第173-378行)
核心改进：
- ✓ **精确ASIN对齐**：使用ASIN作为唯一标识符匹配文本和图像
- ✓ **支持两种图像数据格式**：
  - 字典格式：`{'asins': [...], 'embeddings': array}`
  - 纯数组 + 外部ASIN文件
- ✓ **智能缺失处理**：
  - 有图像：使用实际embedding
  - 无图像：填充零向量（可配置为均值）
- ✓ **PCA优化**：训练时排除零向量，避免影响模型
- ✓ **详细日志**：显示对齐统计、缺失ASIN等信息
- ✓ **缓存优化**：缓存文件名包含填充策略，避免冲突

---

## 新增诊断工具

### 1. `scripts/diagnose_image_data.py`
检查图像数据格式和ASIN信息

**使用方法：**
```bash
python scripts/diagnose_image_data.py \
    --image_path /path/to/your/image_embeddings.npy
```

**功能：**
- 检测数据格式（字典 vs 纯数组）
- 查找ASIN信息（内嵌 vs 外部文件）
- 显示embedding统计信息
- 提供修改建议

### 2. `scripts/validate_multimodal_alignment.py`
验证文本和图像的ASIN对齐情况

**使用方法：**
```bash
python scripts/validate_multimodal_alignment.py \
    --cache_dir cache/AmazonReviews2014/CDs_and_Vinyl \
    --image_path /path/to/image_embeddings.npy \
    --verbose
```

**功能：**
- 统计文本和图像的重叠率
- 检查具体item的对齐情况
- 验证multimodal embeddings的有效性
- 检测异常值（NaN、Inf、全零向量）

---

## 新增文档

### `IMAGE_DATA_FORMAT_GUIDE.md`
完整的图像数据格式修改指南，包括：
- 推荐的数据格式
- 3种修改方法（从简单到复杂）
- 详细的代码示例
- 验证步骤
- 常见问题解答

---

## 使用流程

### Step 1: 诊断当前数据
```bash
python scripts/diagnose_image_data.py \
    --image_path /scratch/zl4789/MQL4GRec/data_process/MQL4GRec/CDs/CDs.emb-ViT-L-14.npy
```

### Step 2: 修改图像数据格式

**选项A：转换为字典格式（推荐）**
```python
import numpy as np

# 加载现有数据
image_embs = np.load('CDs.emb-ViT-L-14.npy')

# 准备ASIN列表（按embedding顺序）
asins = ['B00001', 'B00005', 'B00007', ...]

# 打包
image_data = {
    'asins': asins,
    'embeddings': image_embs
}

# 保存
np.save('CDs.emb-ViT-L-14_aligned.npy', image_data, allow_pickle=True)
```

**选项B：创建外部ASIN文件（最快）**
```bash
# 创建 CDs.emb-ViT-L-14.asins.txt
echo "B00001" > CDs.emb-ViT-L-14.asins.txt
echo "B00005" >> CDs.emb-ViT-L-14.asins.txt
# ... 或用脚本批量写入
```

详见 `IMAGE_DATA_FORMAT_GUIDE.md` 获取完整示例。

### Step 3: 验证对齐
```bash
python scripts/validate_multimodal_alignment.py \
    --cache_dir cache/AmazonReviews2014/CDs_and_Vinyl \
    --image_path /path/to/aligned_image_embeddings.npy \
    --verbose
```

### Step 4: 清除旧缓存
```bash
# 删除旧的缓存文件（重要！）
rm -f cache/AmazonReviews2014/*/processed/image_pca_*.npy
rm -f cache/AmazonReviews2014/*/processed/multimodal_*.npy
```

### Step 5: 运行训练
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --category=CDs_and_Vinyl
```

---

## 关键改进点

### 修改前（有问题）
```python
# ❌ 简单截断，假设顺序一致
min_items = min(text_embs.shape[0], image_embs.shape[0])
text_embs = text_embs[:min_items]
image_embs = image_embs[:min_items]

# 结果：text_embs[0]可能对应image_embs[0]，但ASIN不匹配！
```

### 修改后（正确）
```python
# ✓ 基于ASIN精确对齐
asin2image_emb = dict(zip(image_asins, image_embs_raw))

aligned_image_embs = []
for i in range(1, dataset.n_items):
    asin = dataset.id_mapping['id2item'][i]
    if asin in asin2image_emb:
        aligned_image_embs.append(asin2image_emb[asin])
    else:
        aligned_image_embs.append(np.zeros(768))  # 缺失填充

# 结果：text_embs[i]和aligned_image_embs[i]确保ASIN匹配
```

---

## 预期日志输出

运行训练时，应该看到类似的日志：

```
[TOKENIZER] Loading image embeddings from /scratch/.../CDs.emb-ViT-L-14.npy...
[TOKENIZER] Loaded from dict format: 4523 items, 768D
[TOKENIZER] Built ASIN→Image mapping: 4523 items
[TOKENIZER] ✓ ASIN alignment complete: 4500 matched, 500 missing (10.0% missing)
[TOKENIZER] Training PCA on 4500 valid images (excluding 500 zero-filled items)
[TOKENIZER] Image embeddings after PCA: (5000, 128)
[TOKENIZER] Fusing text and image embeddings...
[TOKENIZER] Fused embeddings shape: (5000, 256) (text:128D + image:128D)
[TOKENIZER] Applying final PCA: 256D → 128D
[TOKENIZER] Cached to cache/.../multimodal_final_pca_128_zero.npy
[TOKENIZER] ✓ Multimodal fusion complete: (5000, 128)
```

关键指标：
- ✓ 显示ASIN对齐统计（matched vs missing）
- ✓ PCA训练只用有效图像
- ✓ 最终shape与文本items数量一致

---

## 配置选项

可以在 `genrec/models/ActionPiece/tokenizer.py` 中调整：

```python
# 第189-193行
IMAGE_PATH_TEMPLATE = "/scratch/zl4789/MQL4GRec/data_process/MQL4GRec/{category}/{category}.emb-ViT-L-14.npy"
USE_MULTIMODAL = True        # 是否启用多模态
IMAGE_PCA_DIM = 128          # 图像PCA维度
FINAL_PCA_DIM = 128          # 最终融合PCA维度
FILL_STRATEGY = 'zero'       # 缺失填充策略：'zero' 或 'mean'
```

---

## 常见问题

### Q: 为什么需要ASIN信息？
A: 因为文本embedding和图像embedding的生成顺序可能不同：
- 文本按 `id2item[1], id2item[2], ...` 顺序
- 图像可能按ASIN字母序或其他顺序
- 需要ASIN作为"桥梁"进行精确匹配

### Q: 如果很多商品没有图像怎么办？
A: 代码已处理：
- 有图像的商品：使用实际embedding
- 无图像的商品：填充零向量
- PCA训练时排除零向量，不影响模型质量

### Q: 需要修改其他代码吗？
A: 不需要。只需要：
1. 修改图像数据格式（添加ASIN信息）
2. 清除旧缓存
3. 重新运行训练

### Q: 如何验证对齐是否正确？
A: 运行验证脚本：
```bash
python scripts/validate_multimodal_alignment.py --verbose
```
检查输出的对齐统计和sample item验证。

---

## 文件清单

### 修改的文件
- `genrec/models/ActionPiece/tokenizer.py` (主要修改)

### 新增的文件
- `scripts/diagnose_image_data.py` (诊断工具)
- `scripts/validate_multimodal_alignment.py` (验证工具)
- `IMAGE_DATA_FORMAT_GUIDE.md` (格式修改指南)
- `MULTIMODAL_ALIGNMENT_CHANGES.md` (本文件)

### 相关文档
- `MULTIMODAL_ID_ALIGNMENT_GUIDE.md` (原始问题分析)
- `DATA_PREPROCESSING_PIPELINE.md` (数据预处理流程)

---

## 后续步骤

1. ✅ 代码已修改完成
2. ⏭️ 检查您的图像数据格式
3. ⏭️ 按照 `IMAGE_DATA_FORMAT_GUIDE.md` 修改数据
4. ⏭️ 运行诊断脚本验证
5. ⏭️ 清除旧缓存
6. ⏭️ 重新训练模型

---

## 技术支持

如有问题，请提供：
1. `diagnose_image_data.py` 的输出
2. `validate_multimodal_alignment.py` 的输出
3. 训练日志中的 `[TOKENIZER]` 相关输出

根据这些信息可以快速定位问题。
