# Visual-Textual Alignment 使用指南

## 方法说明

本代码使用 **方案一：基于ASIN的精确对齐 + 零向量填充** 来merge visual和textual embeddings。

---

## Image Data要求格式

Image data **必须**是字典格式的NPY文件，包含两个键：

```python
{
    'asins': ['B00001', 'B00005', 'B00007', ...],  # ASIN列表
    'embeddings': np.array([                        # 对应的768维向量
        [0.12, -0.34, ..., 0.78],  # B00001的图像embedding
        [0.45, 0.67, ..., -0.23],  # B00005的图像embedding
        [0.89, 0.12, ..., 0.56],   # B00007的图像embedding
        ...
    ])  # shape: (N_items, 768)
}
```

**保存示例：**
```python
import numpy as np

image_data = {
    'asins': your_asin_list,
    'embeddings': your_image_embeddings  # shape: (N, 768)
}

# 必须使用 allow_pickle=True
np.save('CDs.emb-ViT-L-14.npy', image_data, allow_pickle=True)
```

---

## Alignment流程说明

### Step 1: ASIN映射构建
```python
# 从字典中提取数据
image_asins = image_dict['asins']
image_embs_raw = image_dict['embeddings']

# 构建ASIN到embedding的映射
asin2image_emb = {asin: emb for asin, emb in zip(image_asins, image_embs_raw)}
```

### Step 2: 精确对齐
```python
aligned_image_embs = []

# 遍历所有文本items
for i in range(1, dataset.n_items):  # 跳过PAD (index 0)
    asin = dataset.id_mapping['id2item'][i]

    if asin in asin2image_emb:
        # 找到匹配 → 使用真实图像embedding
        aligned_image_embs.append(asin2image_emb[asin])
    else:
        # 未找到 → 填充零向量
        aligned_image_embs.append(np.zeros(768))
```

**结果：**
- `text_embs[i]` 和 `aligned_image_embs[i]` 的ASIN完全匹配
- 没有图像的items用零向量标记

### Step 3: PCA降维（智能处理零向量）
```python
# 只用有图像的items训练PCA
non_zero_mask = ~np.all(aligned_image_embs == 0, axis=1)
train_data = aligned_image_embs[non_zero_mask]

# 训练并应用PCA
image_pca = PCA(n_components=128, whiten=True)
image_pca.fit(train_data)  # 只用真实图像训练
image_embs_reduced = image_pca.transform(aligned_image_embs)  # 应用到所有
```

### Step 4: 融合
```python
# 拼接文本和图像
fused_embs = np.concatenate([text_embs, image_embs_reduced], axis=1)
# shape: (N_items, 256) = (N_items, 128+128)

# 最终PCA
final_pca = PCA(n_components=128, whiten=True)
final_embs = final_pca.fit_transform(fused_embs)
# shape: (N_items, 128)
```

---

## 使用流程

### 1️⃣ 验证Image Data格式
```bash
python scripts/verify_image_dict_format.py \
    --image_path /path/to/your/CDs.emb-ViT-L-14.npy
```

**预期输出：**
```
======================================================================
IMAGE DICTIONARY FORMAT VERIFICATION
======================================================================
File: /path/to/CDs.emb-ViT-L-14.npy
✓ File loaded successfully
✓ Dictionary extracted

Keys found: ['asins', 'embeddings']
✓ Required keys present: ['asins', 'embeddings']

'asins' field:
  Type: <class 'list'>
  Length: 4523
  Sample (first 5): ['B00001', 'B00005', ...]

'embeddings' field:
  Type: <class 'numpy.ndarray'>
  Shape: (4523, 768)
  Dtype: float32

✓ Counts match: 4523 items
✓ All ASINs are unique

======================================================================
✅ FORMAT VERIFICATION PASSED
======================================================================
```

### 2️⃣ 验证ASIN对齐
```bash
python scripts/validate_multimodal_alignment.py \
    --cache_dir cache/AmazonReviews2014/CDs_and_Vinyl \
    --image_path /path/to/CDs.emb-ViT-L-14.npy \
    --verbose
```

**预期输出：**
```
======================================================================
MULTIMODAL ALIGNMENT VALIDATION
======================================================================

1. Text Data (from reviews)
   Total items: 5000

2. Image Data
   Total items: 4523

3. ASIN Overlap Analysis
   Common items (with both text & image): 4500 (90.0%)
   Text-only items (no image): 500 (10.0%)
   Image-only items (no text): 23 (0.5%)

4. Sample Alignment Verification
   Item ID    1: ASIN=B00001         | Has image: ✓
   Item ID   10: ASIN=B00010         | Has image: ✓

======================================================================
SUMMARY & RECOMMENDATIONS
======================================================================
✓ Good alignment: 90.0% of text items have images
  Missing images will be filled with zero vectors
======================================================================
```

### 3️⃣ 配置Image Path

修改 `genrec/models/ActionPiece/tokenizer.py:127`：

```python
IMAGE_PATH_TEMPLATE = "/your/path/to/{category}/{category}.emb-ViT-L-14.npy"
```

或者直接硬编码路径：
```python
image_path = "/absolute/path/to/CDs.emb-ViT-L-14.npy"
```

### 4️⃣ 清除旧缓存
```bash
# 删除旧的缓存文件（重要！）
rm -f cache/AmazonReviews2014/*/processed/image_pca_*.npy
rm -f cache/AmazonReviews2014/*/processed/multimodal_*.npy
```

### 5️⃣ 运行训练
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --category=CDs_and_Vinyl
```

**检查日志：**
```
[TOKENIZER] Loading image embeddings from /path/to/CDs.emb-ViT-L-14.npy...
[TOKENIZER] ✓ Loaded dictionary format: 4523 items, 768D
[TOKENIZER] Built ASIN→Image mapping: 4523 items
[TOKENIZER] ✓ ASIN alignment complete: 4500 matched, 500 missing (10.0% missing)
[TOKENIZER] Training PCA on 4500 valid images (excluding 500 zero-filled items)
[TOKENIZER] Image embeddings after PCA: (5000, 128)
[TOKENIZER] Fusing text and image embeddings...
[TOKENIZER] Fused embeddings shape: (5000, 256) (text:128D + image:128D)
[TOKENIZER] Applying final PCA: 256D → 128D
[TOKENIZER] ✓ Multimodal fusion complete: (5000, 128)
```

---

## 配置选项

在 `genrec/models/ActionPiece/tokenizer.py` 中可以修改：

```python
# 第127-131行
IMAGE_PATH_TEMPLATE = "/path/to/{category}/{category}.emb-ViT-L-14.npy"
USE_MULTIMODAL = True        # 是否启用多模态
IMAGE_PCA_DIM = 128          # 图像PCA降维目标维度
FINAL_PCA_DIM = 128          # 最终融合后PCA降维目标维度
FILL_STRATEGY = 'zero'       # 缺失填充策略：'zero' 或 'mean'
```

**推荐配置：**
- `FILL_STRATEGY = 'zero'`：零向量填充，PCA训练时自动排除
- `IMAGE_PCA_DIM = 128`：与文本embedding维度一致
- `FINAL_PCA_DIM = 128`：与T5模型的d_model一致

---

## 关键优势

### ✅ 完全正确的对齐
- 使用ASIN作为唯一标识符
- 每个text embedding都与正确的image embedding配对
- 避免了简单截断导致的错误匹配

### ✅ 数据零丢失
- 保留所有文本items
- 即使没有图像也不会丢弃item
- 保持`n_items`与原始数据集一致

### ✅ 智能处理缺失
- 有图像：使用真实embedding
- 无图像：零向量标记"无信息"
- PCA训练：自动排除零向量

### ✅ 详细日志追踪
- 显示匹配/缺失统计
- 列出缺失的ASINs（少量时）
- 验证对齐正确性

---

## 常见问题

### Q1: 必须使用字典格式吗？
**A:** 是的。代码已简化为只支持字典格式，确保数据质量和对齐正确性。

### Q2: 如果很多items没有图像怎么办？
**A:** 没问题！代码会：
- 对有图像的items使用真实embedding
- 对无图像的items填充零向量
- PCA训练时只用有图像的items，不影响模型质量

### Q3: 零向量会影响模型吗？
**A:** 不会，因为：
- PCA训练时排除零向量
- 零向量只是占位，表示"无图像信息"
- 最终的PCA transformation会将其映射到合理的空间

### Q4: 如何确认对齐是否正确？
**A:** 运行验证脚本：
```bash
python scripts/validate_multimodal_alignment.py \
    --cache_dir cache/AmazonReviews2014/CDs_and_Vinyl \
    --image_path /path/to/image.npy \
    --verbose
```
检查输出的对齐统计。

### Q5: 可以用均值填充吗？
**A:** 可以，修改 `FILL_STRATEGY = 'mean'`，但不推荐，因为：
- 均值可能引入噪声
- 零向量明确表示"无信息"
- 零向量在PCA时会被自动排除

---

## 数据流示例

假设您有：
- **文本items**: 5000个商品（来自Amazon Reviews 2014）
- **图像items**: 4500个商品（部分商品没有图片）

**Alignment过程：**

1. **构建映射**
   ```
   asin2image_emb = {
       'B00001': [0.12, -0.34, ..., 0.78],
       'B00005': [0.45, 0.67, ..., -0.23],
       ...
   }  # 4500个
   ```

2. **精确对齐**
   ```
   for i in 1..5000:
       asin = id2item[i]
       if asin in asin2image_emb:
           aligned[i] = asin2image_emb[asin]  ← 4500个
       else:
           aligned[i] = [0, 0, ..., 0]         ← 500个零向量
   ```

3. **PCA降维**
   ```
   # 只用4500个真实图像训练PCA
   pca.fit(aligned[non_zero_indices])

   # 应用到全部5000个
   image_embs_128d = pca.transform(aligned)  # (5000, 128)
   ```

4. **融合**
   ```
   text_embs_128d:  (5000, 128)
   image_embs_128d: (5000, 128)

   fused: (5000, 256)
   final: (5000, 128)  ← 最终输出
   ```

---

## 文件清单

**修改的文件：**
- `genrec/models/ActionPiece/tokenizer.py`
  - 删除了外部ASIN文件加载逻辑
  - 专注于字典格式处理
  - 添加了格式验证

**新增工具：**
- `scripts/verify_image_dict_format.py` - 验证字典格式
- `scripts/validate_multimodal_alignment.py` - 验证ASIN对齐

**文档：**
- `VISUAL_TEXTUAL_ALIGNMENT_USAGE.md` - 本文件
- `MULTIMODAL_ALIGNMENT_CHANGES.md` - 修改总结

---

## 快速检查清单

在运行训练前，请确认：

- [ ] Image data是字典格式：`{'asins': [...], 'embeddings': array}`
- [ ] 运行了格式验证脚本，输出 ✅ PASSED
- [ ] 运行了对齐验证脚本，overlap > 80%
- [ ] 配置了正确的`IMAGE_PATH_TEMPLATE`
- [ ] 删除了旧的缓存文件
- [ ] 检查训练日志中的`[TOKENIZER]`输出

全部✅后即可开始训练！
