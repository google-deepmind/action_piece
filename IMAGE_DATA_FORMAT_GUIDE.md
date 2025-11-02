# Image Data格式修改指南

## 目录
1. [问题说明](#问题说明)
2. [推荐的数据格式](#推荐的数据格式)
3. [修改方法](#修改方法)
4. [验证步骤](#验证步骤)

---

## 问题说明

当前的image data只包含每个物品图片的768维向量，但**缺少ASIN信息**，导致无法与文本embedding进行精确对齐。

### 当前格式（有问题）
```python
# CDs.emb-ViT-L-14.npy
# 纯数组格式，shape: (N_items, 768)
image_embs = np.array([
    [0.12, -0.34, ..., 0.78],  # 第1个商品的图像embedding
    [0.45, 0.67, ..., -0.23],  # 第2个商品的图像embedding
    ...
])
# ❌ 问题：不知道每个向量对应哪个ASIN
```

---

## 推荐的数据格式

### 格式1：字典式NPY（推荐，最简单）

将image data保存为包含ASIN信息的字典格式：

```python
import numpy as np

# 准备数据
image_data = {
    'asins': ['B00001', 'B00005', 'B00007', ...],  # ASIN列表
    'embeddings': np.array([                        # 对应的768维向量
        [0.12, -0.34, ..., 0.78],  # B00001的图像embedding
        [0.45, 0.67, ..., -0.23],  # B00005的图像embedding
        [0.89, 0.12, ..., 0.56],   # B00007的图像embedding
        ...
    ])  # shape: (N_items, 768)
}

# 保存（注意：必须使用allow_pickle=True）
np.save('CDs.emb-ViT-L-14.npy', image_data, allow_pickle=True)

# ✓ 优点：
# - ASIN信息和embedding在同一文件中
# - 修改后的代码可以直接使用
# - 不需要额外的配套文件
```

### 格式2：纯数组 + 外部ASIN文件

如果无法修改原始NPY文件，可以创建外部ASIN文件：

#### 选项A：TXT文件（最简单）
```bash
# 创建 CDs.emb-ViT-L-14.asins.txt
# 每行一个ASIN，顺序与NPY文件中的向量一一对应
B00001
B00005
B00007
...
```

#### 选项B：JSON文件
```json
// CDs.emb-ViT-L-14.asins.json
[
  "B00001",
  "B00005",
  "B00007",
  ...
]
```

#### 选项C：配置文件
```json
// CDs.emb-ViT-L-14.config.json
{
  "asins": ["B00001", "B00005", "B00007", ...],
  "embedding_dim": 768,
  "model": "ViT-L-14",
  "created_at": "2024-01-01"
}
```

---

## 修改方法

### 方法1：从现有图像数据生成ASIN映射

如果您有生成图像embedding的原始代码，可以在生成时同时保存ASIN：

```python
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# 加载CLIP模型
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 准备数据
asins = []
embeddings = []

# 遍历所有商品图像
for asin, image_path in item_images.items():
    # 加载图像
    image = Image.open(image_path)

    # 编码
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # 保存
    asins.append(asin)
    embeddings.append(image_features.cpu().numpy()[0])

# 保存为字典格式（推荐）
image_data = {
    'asins': asins,
    'embeddings': np.array(embeddings)
}
np.save('CDs.emb-ViT-L-14.npy', image_data, allow_pickle=True)

print(f"Saved {len(asins)} image embeddings with ASIN mapping")
```

### 方法2：如果已有NPY文件，但有ASIN顺序信息

如果您知道NPY文件中向量的ASIN顺序（例如按ASIN字母序排序），可以重新打包：

```python
import numpy as np
import json

# 1. 加载现有的图像embedding
image_embs = np.load('CDs.emb-ViT-L-14.npy')  # shape: (N, 768)

# 2. 准备ASIN列表（按生成顺序）
# 例如：如果您的图像是按ASIN排序处理的
with open('sorted_asins.json', 'r') as f:
    asins = json.load(f)

# 或者从Amazon数据中提取
# asins = sorted(list(item_images.keys()))

# 3. 验证数量匹配
assert len(asins) == image_embs.shape[0], \
    f"ASIN count ({len(asins)}) != embedding count ({image_embs.shape[0]})"

# 4. 重新打包为字典格式
image_data = {
    'asins': asins,
    'embeddings': image_embs
}

# 5. 保存
np.save('CDs.emb-ViT-L-14_with_asins.npy', image_data, allow_pickle=True)
print(f"✓ Saved {len(asins)} embeddings with ASIN mapping")
```

### 方法3：创建外部ASIN文件（最快速）

如果无法修改NPY文件，只需创建配套的ASIN文件：

```python
import numpy as np

# 您的ASIN列表（与NPY文件中向量顺序一致）
asins = ['B00001', 'B00005', 'B00007', ...]

# 保存为TXT
with open('CDs.emb-ViT-L-14.asins.txt', 'w') as f:
    for asin in asins:
        f.write(f"{asin}\n")

# 或保存为JSON
import json
with open('CDs.emb-ViT-L-14.asins.json', 'w') as f:
    json.dump(asins, f)

print(f"✓ Saved {len(asins)} ASINs to external file")
```

---

## 验证步骤

### 步骤1：使用诊断脚本检查数据格式

```bash
python scripts/diagnose_image_data.py \
    --image_path /scratch/zl4789/MQL4GRec/data_process/MQL4GRec/CDs/CDs.emb-ViT-L-14.npy
```

**预期输出（成功）：**
```
======================================================================
IMAGE DATA DIAGNOSTIC REPORT
======================================================================
File: /scratch/zl4789/MQL4GRec/data_process/MQL4GRec/CDs/CDs.emb-ViT-L-14.npy
Exists: True
File size: 23.45 MB

✓ Successfully loaded data

Data type: <class 'numpy.ndarray'>
Data dtype: object

📦 Dictionary format detected
Keys: ['asins', 'embeddings']

  Key: 'asins'
    Type: <class 'list'>
    Length: 4523
    First 3 items: ['B00001', 'B00005', 'B00007']

  Key: 'embeddings'
    Type: <class 'numpy.ndarray'>
    Shape: (4523, 768)

✓ ASIN information found in 'asins' key
  Total ASINs: 4523
  Sample ASINs: ['B00001', 'B00005', 'B00007', ...]

✓ Embeddings found
  Shape: (4523, 768)
  Dtype: float32
  Mean: 0.0234
  Std: 0.4567

======================================================================
RECOMMENDATION:
======================================================================
✓ Your image data contains ASIN information.
  The modified code can use it directly for alignment.
======================================================================
```

### 步骤2：验证ASIN对齐

```bash
python scripts/validate_multimodal_alignment.py \
    --cache_dir cache/AmazonReviews2014/CDs_and_Vinyl \
    --image_path /scratch/zl4789/MQL4GRec/data_process/MQL4GRec/CDs/CDs.emb-ViT-L-14.npy \
    --verbose
```

**预期输出（成功）：**
```
======================================================================
MULTIMODAL ALIGNMENT VALIDATION
======================================================================

1. Text Data (from reviews)
   Total items: 5000
   Sample ASINs: ['B00001', 'B00005', 'B00007', 'B00009', 'B00010']

2. Image Data
   Total items: 4523
   Sample ASINs: ['B00001', 'B00005', 'B00007', 'B00009', 'B00010']

3. ASIN Overlap Analysis
   Common items (with both text & image): 4500 (90.0%)
   Text-only items (no image): 500 (10.0%)
   Image-only items (no text): 23 (0.5%)

4. Sample Alignment Verification
   Item ID    1: ASIN=B00001         | Has image: ✓
   Item ID   10: ASIN=B00010         | Has image: ✓
   Item ID  100: ASIN=B00123         | Has image: ✗
   Item ID  500: ASIN=B00567         | Has image: ✓

======================================================================
SUMMARY & RECOMMENDATIONS
======================================================================
✓ Good alignment: 90.0% of text items have images
  Missing images will be filled with zero vectors

⚠️  Note: 23 images have no corresponding text (will be ignored)
======================================================================
```

### 步骤3：运行训练测试

清除旧缓存并重新运行：

```bash
# 删除旧的缓存文件（重要！）
rm -f cache/AmazonReviews2014/CDs_and_Vinyl/processed/image_pca_*.npy
rm -f cache/AmazonReviews2014/CDs_and_Vinyl/processed/multimodal_*.npy

# 运行训练
CUDA_VISIBLE_DEVICES=0 python main.py --category=CDs_and_Vinyl
```

**检查日志输出：**
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
[TOKENIZER] ✓ Multimodal fusion complete: (5000, 128)
```

---

## 常见问题

### Q1: 如果我的NPY文件很大，重新保存会很慢吗？

A: 是的，但只需要做一次。推荐使用方法3（外部ASIN文件），这样不需要重新保存NPY文件。

```python
# 快速方案：只创建.asins.txt文件
with open('CDs.emb-ViT-L-14.asins.txt', 'w') as f:
    for asin in your_asin_list:
        f.write(f"{asin}\n")
```

### Q2: 我的图像数据按什么顺序排列的？

A: 常见的顺序有：
- **ASIN字母序**：sorted(asins)
- **处理顺序**：按文件夹遍历顺序
- **随机顺序**：需要查看生成代码

**如何确认：** 检查生成图像embedding的原始代码，看它是如何遍历商品的。

### Q3: 部分商品没有图像怎么办？

A: 修改后的代码已经处理了这种情况：
- **有图像**：使用实际的图像embedding
- **无图像**：填充零向量（`FILL_STRATEGY='zero'`）
- PCA训练时会排除零向量，避免影响模型

### Q4: 可以使用平均值填充吗？

A: 可以，修改`tokenizer.py`中的`FILL_STRATEGY`：

```python
# genrec/models/ActionPiece/tokenizer.py:193
FILL_STRATEGY = 'mean'  # 使用所有图像的平均值填充
```

但推荐使用`'zero'`，因为：
- 零向量明确表示"无信息"
- 平均值可能引入噪声
- PCA训练时会排除零向量

---

## 完整示例：转换现有数据

假设您有以下数据：
- 图像embedding：`CDs.emb-ViT-L-14.npy` (纯数组)
- ASIN列表文件：`cds_asins.txt`

### Step 1: 检查当前格式
```bash
python scripts/diagnose_image_data.py --image_path CDs.emb-ViT-L-14.npy
```

### Step 2: 转换为字典格式
```python
import numpy as np

# 加载现有数据
image_embs = np.load('CDs.emb-ViT-L-14.npy')
print(f"Loaded embeddings: {image_embs.shape}")

# 加载ASIN列表
with open('cds_asins.txt', 'r') as f:
    asins = [line.strip() for line in f]
print(f"Loaded ASINs: {len(asins)}")

# 验证数量
assert len(asins) == image_embs.shape[0], "Count mismatch!"

# 打包为字典
image_data = {
    'asins': asins,
    'embeddings': image_embs
}

# 保存
output_path = 'CDs.emb-ViT-L-14_aligned.npy'
np.save(output_path, image_data, allow_pickle=True)
print(f"✓ Saved to {output_path}")

# 验证
loaded = np.load(output_path, allow_pickle=True).item()
print(f"✓ Verification passed: {len(loaded['asins'])} items")
```

### Step 3: 验证对齐
```bash
python scripts/validate_multimodal_alignment.py \
    --cache_dir cache/AmazonReviews2014/CDs_and_Vinyl \
    --image_path CDs.emb-ViT-L-14_aligned.npy \
    --verbose
```

### Step 4: 更新路径并训练
```python
# 修改 genrec/models/ActionPiece/tokenizer.py:189
IMAGE_PATH_TEMPLATE = "/path/to/CDs.emb-ViT-L-14_aligned.npy"
```

```bash
# 运行训练
CUDA_VISIBLE_DEVICES=0 python main.py --category=CDs_and_Vinyl
```

---

## 总结

### ✓ 推荐方案（按优先级）

1. **字典式NPY** (最推荐)
   - 将ASIN和embedding打包在一起
   - 一次修改，永久使用
   - 代码可以直接识别

2. **外部TXT文件** (最快速)
   - 创建`{原文件名}.asins.txt`
   - 不需要修改NPY文件
   - 适合临时测试

3. **外部JSON文件** (最灵活)
   - 可以添加额外元数据
   - 适合复杂场景

### ✗ 不推荐的做法

- ❌ 直接截断到最小数量（丢失数据）
- ❌ 假设顺序一致（可能错误对齐）
- ❌ 手动调整id_mapping（破坏数据完整性）

---

## 需要帮助？

如果遇到问题，请提供以下信息：
1. 运行`scripts/diagnose_image_data.py`的输出
2. 运行`scripts/validate_multimodal_alignment.py`的输出
3. 图像embedding的生成代码（如果有）

根据这些信息，可以提供更具体的解决方案。
