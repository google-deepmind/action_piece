# 多模态融合中的ID对齐与Item数量更新指南

## 目录
1. [问题概述](#问题概述)
2. [当前实现的问题分析](#当前实现的问题分析)
3. [解决方案总览](#解决方案总览)
4. [方案一：基于ASIN的精确对齐（推荐）](#方案一基于asin的精确对齐推荐)
5. [方案二：保守的截断对齐（快速但可能丢失数据）](#方案二保守的截断对齐快速但可能丢失数据)
6. [方案三：生成缺失的图像Embedding](#方案三生成缺失的图像embedding)
7. [实施步骤详解](#实施步骤详解)
8. [代码实现示例](#代码实现示例)
9. [测试与验证](#测试与验证)

---

## 问题概述

### 当前问题
在多模态融合过程中，存在两个核心问题：

1. **ID对齐问题**
   - 文本embedding按`dataset.id_mapping['id2item']`顺序生成（基于评论数据）
   - 图像embedding按`.npy`文件中的顺序存储（可能是ASIN字典序或其他顺序）
   - **两者的item顺序不一致**，导致错误的配对

2. **Item数量不一致问题**
   - 评论数据包含 N_text 个商品（如 5000个）
   - 图像数据包含 N_image 个商品（如 4500个，因为有些商品没有图片）
   - 当前代码简单截断到 `min(N_text, N_image)`，**丢失了有效数据**

### 影响
- **错误的多模态表示**：商品A的文本特征 + 商品B的图像特征 → 无意义的融合
- **数据丢失**：有文本但无图像的商品被完全丢弃
- **模型性能下降**：错误的embedding导致推荐质量大幅下降

---

## 当前实现的问题分析

### 代码位置
`genrec/models/ActionPiece/tokenizer.py:114-240`

### 问题代码片段
```python
def _load_and_fuse_image_embeddings(self, dataset, text_embs):
    # 1. 加载图像embeddings
    image_embs = np.load(image_path)  # shape: (N_image, 768)

    # ❌ 问题1：简单截断，假设顺序一致
    min_items = min(text_embs.shape[0], image_embs.shape[0])
    text_embs = text_embs[:min_items]      # 丢弃后面的文本
    image_embs = image_embs[:min_items]    # 丢弃后面的图像

    # ❌ 问题2：没有验证ID对齐
    # text_embs[0] 对应 dataset.id_mapping['id2item'][1]
    # image_embs[0] 对应 ??? (未知ASIN)

    fused_embs = np.concatenate([text_embs, image_embs], axis=1)
    # ❌ 结果：可能是错误的配对！
```

### 文本Embedding生成逻辑
`genrec/models/ActionPiece/tokenizer.py:71-112`

```python
def _encode_sent_emb(self, dataset, output_path):
    meta_sentences = []
    # ✅ 文本按 id2item 顺序生成
    for i in range(1, dataset.n_items):  # 从1开始（0是PAD）
        item_asin = dataset.id_mapping['id2item'][i]
        meta_sentences.append(item2meta[item_asin])

    sent_embs = sent_emb_model.encode(meta_sentences)
    # 输出: sent_embs[0] 对应 id2item[1]
    #      sent_embs[1] 对应 id2item[2]
    #      ...
```

### 图像Embedding存储格式（假设）
假设图像`.npy`文件是按ASIN排序存储：
```python
# 图像embedding的可能存储格式
image_data = {
    'B00001': embedding_768d,
    'B00005': embedding_768d,  # 注意：B00002, B00003, B00004 缺失
    'B00007': embedding_768d,
    ...
}
# 保存为 .npy 时按ASIN排序或原始顺序
```

### 问题示例
```
评论数据 (按出现顺序):
  id2item[1] = 'B00005'
  id2item[2] = 'B00001'
  id2item[3] = 'B00007'
  id2item[4] = 'B00002'  ← 无图像
  id2item[5] = 'B00009'

图像数据 (按ASIN排序):
  image_embs[0] = 'B00001' 的embedding
  image_embs[1] = 'B00005' 的embedding
  image_embs[2] = 'B00007' 的embedding
  image_embs[3] = 'B00009' 的embedding

当前代码的错误配对:
  text_embs[0] (B00005) + image_embs[0] (B00001) ❌
  text_embs[1] (B00001) + image_embs[1] (B00005) ❌
  text_embs[2] (B00007) + image_embs[2] (B00007) ✅ (偶然正确)
```

---

## 解决方案总览

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **方案一：基于ASIN精确对齐** | 完全正确，保留所有数据 | 需要图像的ASIN元数据 | ⭐⭐⭐⭐⭐ |
| **方案二：保守截断** | 实现简单，快速 | 丢失大量数据 | ⭐⭐ |
| **方案三：生成缺失图像** | 不丢失文本数据 | 需要额外计算 | ⭐⭐⭐⭐ |

---

## 方案一：基于ASIN的精确对齐（推荐）

### 核心思想
使用商品ASIN作为唯一标识符，精确匹配文本和图像embedding。

### 前置条件
图像embedding文件需要包含ASIN信息，有以下几种可能格式：

#### 格式1：字典式NPY（推荐）
```python
# 保存时
image_data = {
    'asins': ['B00001', 'B00005', 'B00007', ...],
    'embeddings': np.array([[...], [...], ...])  # shape: (N, 768)
}
np.save('image_embs.npy', image_data)

# 加载时
image_data = np.load('image_embs.npy', allow_pickle=True).item()
asin2image_emb = dict(zip(image_data['asins'], image_data['embeddings']))
```

#### 完整流程
```
Step 1: 准备ASIN映射
    ├─ 文本侧: dataset.id_mapping['id2item'] → [ASIN1, ASIN2, ...]
    └─ 图像侧: 加载ASIN列表 → asin2image_emb

Step 2: 精确匹配
    ├─ 遍历 dataset.id_mapping['id2item']
    ├─ 对于每个ASIN，查找对应的图像embedding
    └─ 构建对齐后的embedding数组

Step 3: 处理缺失项
    ├─ 有文本有图像: 正常融合
    ├─ 有文本无图像: 使用零向量或平均向量填充
    └─ 更新 n_items 和 item2id 映射

Step 4: 融合
    ├─ 拼接对齐后的文本和图像embedding
    └─ 应用最终PCA
```

### 伪代码实现
```python
def _load_and_fuse_image_embeddings_aligned(self, dataset, text_embs):
    """精确对齐的多模态融合"""

    # 1. 加载图像embedding和ASIN映射
    image_data = load_image_data_with_asins(image_path)
    asin2image_emb = dict(zip(image_data['asins'], image_data['embeddings']))

    # 2. 为每个文本embedding找到对应的图像embedding
    aligned_image_embs = []
    valid_indices = []  # 记录有图像的item索引

    for i in range(1, dataset.n_items):
        asin = dataset.id_mapping['id2item'][i]

        if asin in asin2image_emb:
            # 找到匹配的图像
            aligned_image_embs.append(asin2image_emb[asin])
            valid_indices.append(i - 1)  # text_embs从0开始
        else:
            # 没有图像，使用零向量或跳过
            self.logger.warning(f'No image for item {asin}')
            # 选项1: 填充零向量
            aligned_image_embs.append(np.zeros(768))
            valid_indices.append(i - 1)
            # 选项2: 跳过该item（需要更新id_mapping）
            # continue

    aligned_image_embs = np.array(aligned_image_embs)

    # 3. 验证对齐
    assert aligned_image_embs.shape[0] == text_embs.shape[0], \
        f"Shape mismatch: text={text_embs.shape}, image={aligned_image_embs.shape}"

    # 4. 应用图像PCA
    image_pca = PCA(n_components=128, whiten=True)
    image_embs_reduced = image_pca.fit_transform(aligned_image_embs)

    # 5. 融合
    fused_embs = np.concatenate([text_embs, image_embs_reduced], axis=1)

    # 6. 最终PCA
    final_pca = PCA(n_components=128, whiten=True)
    final_embs = final_pca.fit_transform(fused_embs)

    return final_embs
```

---

## 方案二：保守的截断对齐（快速但可能丢失数据）

### 适用场景
- 图像数据没有ASIN信息
- 需要快速验证多模态效果
- 可以接受数据丢失

### 核心思想
只使用同时有文本和图像的item，并验证它们的顺序。

### 实现步骤

```
Step 1: 确定公共item集合
    ├─ 从图像文件推断有图像的ASIN列表
    └─ 与评论数据的ASIN列表求交集

Step 2: 重建ID映射
    ├─ 只保留公共item
    ├─ 重新分配 item_id (1, 2, 3, ...)
    └─ 更新 dataset.id_mapping

Step 3: 重新生成文本embedding
    ├─ 只为公共item生成文本embedding
    └─ 确保顺序与图像embedding一致

Step 4: 融合
    ├─ 此时文本和图像embedding完全对齐
    └─ 正常拼接和PCA
```

### 伪代码
```python
def _conservative_alignment(self, dataset, image_asins):
    """保守对齐：只使用公共item"""

    # 1. 找到公共ASIN
    text_asins = set(dataset.id_mapping['item2id'].keys()) - {'[PAD]'}
    image_asins = set(image_asins)
    common_asins = text_asins & image_asins

    self.logger.info(
        f'Items with text only: {len(text_asins - image_asins)}\n'
        f'Items with image only: {len(image_asins - text_asins)}\n'
        f'Common items: {len(common_asins)}'
    )

    # 2. 重建ID映射（只包含公共item）
    new_id_mapping = {
        'user2id': dataset.id_mapping['user2id'],  # 用户映射不变
        'item2id': {'[PAD]': 0},
        'id2user': dataset.id_mapping['id2user'],
        'id2item': ['[PAD]']
    }

    sorted_common_asins = sorted(common_asins)  # 确保顺序一致
    for asin in sorted_common_asins:
        new_id = len(new_id_mapping['id2item'])
        new_id_mapping['item2id'][asin] = new_id
        new_id_mapping['id2item'].append(asin)

    # 3. 更新dataset
    dataset.id_mapping = new_id_mapping

    # 4. 过滤用户序列（移除无图像的item）
    new_all_item_seqs = {}
    for user, item_seq in dataset.all_item_seqs.items():
        filtered_seq = [item for item in item_seq if item in common_asins]
        if len(filtered_seq) >= 3:  # 至少保留3个交互
            new_all_item_seqs[user] = filtered_seq

    dataset.all_item_seqs = new_all_item_seqs

    # 5. 重新生成文本embedding（按新的id2item顺序）
    # 6. 加载对应的图像embedding（按相同顺序）
    # 7. 融合
```

### 缺点
- 丢失了有文本但无图像的item（可能占20-30%）
- 丢失了部分用户交互（序列变短）
- 可能导致训练集变小

---

## 方案三：生成缺失的图像Embedding

### 核心思想
对于没有图像的item，使用替代方法生成图像embedding。

### 替代方法

#### 方法1：使用类别平均值
```python
# 计算同类别商品的图像embedding平均值
category2avg_image_emb = {}
for asin, image_emb in asin2image_emb.items():
    category = get_category(asin)
    if category not in category2avg_image_emb:
        category2avg_image_emb[category] = []
    category2avg_image_emb[category].append(image_emb)

for cat in category2avg_image_emb:
    category2avg_image_emb[cat] = np.mean(category2avg_image_emb[cat], axis=0)

# 对于缺失图像的item
if asin not in asin2image_emb:
    category = get_category(asin)
    image_emb = category2avg_image_emb.get(category, global_avg_image_emb)
```

#### 方法2：使用CLIP生成
```python
# 如果有商品标题，使用CLIP的文本编码器生成"伪图像"特征
from transformers import CLIPModel, CLIPProcessor

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def text_to_pseudo_image_emb(title):
    # 使用文本生成伪图像特征
    inputs = clip_processor(text=[title], return_tensors="pt", padding=True)
    text_features = clip_model.get_text_features(**inputs)
    return text_features.detach().numpy()[0]

# 对于缺失图像的item
if asin not in asin2image_emb:
    title = dataset.item2meta[asin]['title']
    image_emb = text_to_pseudo_image_emb(title)
```

#### 方法3：零向量填充 + Masking
```python
# 使用零向量填充，并添加mask标记
image_embs_with_mask = []
image_masks = []

for asin in id2item[1:]:
    if asin in asin2image_emb:
        image_embs_with_mask.append(asin2image_emb[asin])
        image_masks.append(1)  # 真实图像
    else:
        image_embs_with_mask.append(np.zeros(768))
        image_masks.append(0)  # 填充

# 在PCA或融合时考虑mask
# 方案A: 只用真实图像训练PCA
real_image_indices = [i for i, m in enumerate(image_masks) if m == 1]
image_pca.fit(image_embs_with_mask[real_image_indices])
image_embs_reduced = image_pca.transform(image_embs_with_mask)

# 方案B: 在模型中添加image_mask输入
```

---

## 实施步骤详解

### 步骤1：诊断当前数据
在修改代码前，先了解数据情况。

#### 1.1 检查图像数据格式
```python
# 创建诊断脚本: scripts/diagnose_image_data.py
import numpy as np

image_path = "/scratch/zl4789/MQL4GRec/data_process/MQL4GRec/CDs/CDs.emb-ViT-L-14.npy"

# 尝试加载
image_data = np.load(image_path, allow_pickle=True)

print(f"Type: {type(image_data)}")
print(f"Shape: {image_data.shape if hasattr(image_data, 'shape') else 'N/A'}")

# 如果是字典
if isinstance(image_data, dict) or (hasattr(image_data, 'item') and callable(image_data.item)):
    try:
        data_dict = image_data.item() if hasattr(image_data, 'item') else image_data
        print(f"Keys: {data_dict.keys()}")
        for key in data_dict:
            print(f"  {key}: {type(data_dict[key])}, shape: {data_dict[key].shape if hasattr(data_dict[key], 'shape') else 'N/A'}")
    except:
        pass

# 如果是数组
if isinstance(image_data, np.ndarray):
    print(f"Array shape: {image_data.shape}")
    print(f"First row type: {type(image_data[0])}")
```

运行：
```bash
python scripts/diagnose_image_data.py
```

#### 1.2 检查ASIN对齐
```python
# scripts/check_asin_alignment.py
import json
import numpy as np

# 加载评论数据的ASIN
cache_dir = "cache/AmazonReviews2014/CDs_and_Vinyl/processed"
with open(f"{cache_dir}/id_mapping.json") as f:
    id_mapping = json.load(f)

text_asins = set(id_mapping['id2item'][1:])  # 跳过PAD
print(f"Text ASINs count: {len(text_asins)}")
print(f"Sample text ASINs: {list(text_asins)[:5]}")

# 加载图像数据的ASIN（根据实际格式调整）
image_data = np.load(image_path, allow_pickle=True)
# 假设有ASIN信息
if 'asins' in image_data:
    image_asins = set(image_data['asins'])
elif 'asin' in image_data:
    image_asins = set(image_data['asin'])
else:
    print("WARNING: No ASIN info found in image data!")
    image_asins = set()

print(f"Image ASINs count: {len(image_asins)}")
print(f"Sample image ASINs: {list(image_asins)[:5]}")

# 检查重叠
common = text_asins & image_asins
print(f"\nOverlap: {len(common)} items")
print(f"Text only: {len(text_asins - image_asins)} items")
print(f"Image only: {len(image_asins - text_asins)} items")
```

### 步骤2：选择并实现对齐方案

根据步骤1的诊断结果选择方案：

| 诊断结果 | 推荐方案 |
|----------|----------|
| 图像数据有ASIN | 方案一（精确对齐） |
| 图像数据无ASIN，但有配套文件 | 方案一（加载配套文件） |
| 图像数据无ASIN，顺序未知 | 方案二（保守截断）或 联系数据提供方 |
| 缺失大量图像 | 方案三（生成缺失embedding） |

### 步骤3：修改tokenizer.py

#### 3.1 添加ASIN加载函数
```python
# genrec/models/ActionPiece/tokenizer.py

def _load_image_asins(self, image_base_path: str) -> list[str]:
    """加载图像embedding对应的ASIN列表

    尝试多种方法：
    1. 从.npy文件内部读取
    2. 从配套的.txt或.json文件读取
    3. 从配置文件读取
    """
    # 方法1: 从.npy内部
    try:
        image_data = np.load(image_base_path, allow_pickle=True).item()
        if 'asins' in image_data:
            return image_data['asins']
        if 'asin' in image_data:
            return image_data['asin']
    except:
        pass

    # 方法2: 配套文件
    asin_file = image_base_path.replace('.npy', '.asins.txt')
    if os.path.exists(asin_file):
        with open(asin_file) as f:
            return [line.strip() for line in f]

    # 方法3: JSON配套
    asin_json = image_base_path.replace('.npy', '.asins.json')
    if os.path.exists(asin_json):
        with open(asin_json) as f:
            return json.load(f)

    raise FileNotFoundError(
        f"Cannot find ASIN mapping for {image_base_path}. "
        f"Please provide one of: "
        f"1) ASIN info in .npy file, "
        f"2) {asin_file}, "
        f"3) {asin_json}"
    )
```

#### 3.2 重写融合函数
```python
def _load_and_fuse_image_embeddings(self, dataset, text_embs):
    """精确对齐的多模态融合"""

    # 配置
    IMAGE_PATH = self._get_image_path()
    IMAGE_PCA_DIM = 128
    FINAL_PCA_DIM = 128

    if not os.path.exists(IMAGE_PATH):
        self.logger.warning(f'Image file not found: {IMAGE_PATH}')
        return text_embs

    # 1. 加载图像数据和ASIN
    self.logger.info('Loading image embeddings and ASINs...')
    image_data = np.load(IMAGE_PATH, allow_pickle=True)

    # 尝试提取ASIN
    try:
        if isinstance(image_data, dict) or hasattr(image_data, 'item'):
            image_dict = image_data.item() if hasattr(image_data, 'item') else image_data
            image_embs_raw = image_dict['embeddings']
            image_asins = image_dict['asins']
        else:
            # 纯数组，需要外部ASIN
            image_embs_raw = image_data
            image_asins = self._load_image_asins(IMAGE_PATH)
    except Exception as e:
        self.logger.error(f'Failed to load image ASINs: {e}')
        return text_embs

    # 2. 构建ASIN到图像embedding的映射
    asin2image_emb = dict(zip(image_asins, image_embs_raw))
    self.logger.info(f'Loaded {len(asin2image_emb)} image embeddings')

    # 3. 按文本embedding顺序对齐图像
    aligned_image_embs = []
    missing_count = 0

    for i in range(1, dataset.n_items):  # 从1开始，跳过PAD
        asin = dataset.id_mapping['id2item'][i]

        if asin in asin2image_emb:
            aligned_image_embs.append(asin2image_emb[asin])
        else:
            # 使用零向量填充缺失
            aligned_image_embs.append(np.zeros(image_embs_raw.shape[1]))
            missing_count += 1

    aligned_image_embs = np.array(aligned_image_embs)

    self.logger.info(
        f'Aligned {aligned_image_embs.shape[0]} items, '
        f'{missing_count} missing images filled with zeros'
    )

    # 4. 验证对齐
    assert text_embs.shape[0] == aligned_image_embs.shape[0], \
        f'Shape mismatch: text={text_embs.shape}, image={aligned_image_embs.shape}'

    # 5. 图像PCA
    # 只用有图像的item训练PCA
    valid_mask = ~np.all(aligned_image_embs == 0, axis=1)
    if np.sum(valid_mask) > 0:
        image_pca = PCA(n_components=IMAGE_PCA_DIM, whiten=True)
        image_pca.fit(aligned_image_embs[valid_mask])
        image_embs_reduced = image_pca.transform(aligned_image_embs)
    else:
        self.logger.warning('No valid images for PCA!')
        image_embs_reduced = aligned_image_embs

    # 6. 融合
    fused_embs = np.concatenate([text_embs, image_embs_reduced], axis=1)

    # 7. 最终PCA
    final_pca = PCA(n_components=FINAL_PCA_DIM, whiten=True)
    final_embs = final_pca.fit_transform(fused_embs)

    self.logger.info(f'Final multimodal embeddings shape: {final_embs.shape}')

    return final_embs
```

### 步骤4：更新配置
```yaml
# genrec/models/ActionPiece/config.yaml

# 多模态配置
multimodal:
  enabled: true
  image_emb_path: "/scratch/zl4789/MQL4GRec/data_process/MQL4GRec/{category}/{category}.emb-ViT-L-14.npy"
  image_pca_dim: 128
  final_pca_dim: 128
  fill_missing: "zero"  # zero / category_mean / clip_text
```

### 步骤5：测试与验证

#### 5.1 单元测试
```python
# tests/test_multimodal_alignment.py

def test_asin_alignment():
    """测试ASIN对齐是否正确"""
    # 创建模拟数据
    dataset = create_mock_dataset()
    tokenizer = ActionPieceTokenizer(config, dataset)

    # 加载embedding
    text_embs = tokenizer._get_sent_embs(dataset)
    fused_embs = tokenizer._load_and_fuse_image_embeddings(dataset, text_embs)

    # 验证
    assert text_embs.shape[0] == fused_embs.shape[0]

    # 手动检查几个ASIN
    for i in [1, 10, 100]:
        asin = dataset.id_mapping['id2item'][i]
        # 验证fused_embs[i-1]确实对应该ASIN
        print(f"Item {i}: ASIN={asin}")
```

#### 5.2 可视化验证
```python
# scripts/visualize_alignment.py

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载文本和图像embedding
text_embs = np.load('cache/.../sentence-t5-base.sent_emb')
fused_embs = np.load('cache/.../multimodal_final_pca_128.npy')

# 选择几个已知的item
known_items = {
    'B001': 'Sony Headphones',
    'B002': 'Apple AirPods',
    'B003': 'Bose Speaker'
}

# t-SNE可视化
tsne = TSNE(n_components=2)
text_2d = tsne.fit_transform(text_embs[[id2item.index(asin) for asin in known_items]])
fused_2d = tsne.fit_transform(fused_embs[[id2item.index(asin) for asin in known_items]])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(text_2d[:, 0], text_2d[:, 1])
plt.title('Text-only embeddings')

plt.subplot(1, 2, 2)
plt.scatter(fused_2d[:, 0], fused_2d[:, 1])
plt.title('Multimodal embeddings')

plt.savefig('alignment_visualization.png')
```

---

## 代码实现示例

### 完整的精确对齐实现

```python
# genrec/models/ActionPiece/tokenizer.py

def _load_and_fuse_image_embeddings(
    self,
    dataset: AbstractDataset,
    text_embs: np.ndarray
) -> np.ndarray:
    """
    精确对齐文本和图像embedding，然后融合

    Args:
        dataset: 数据集对象，包含id_mapping
        text_embs: 文本embedding，shape=(N_items-1, 128)
                  text_embs[i] 对应 dataset.id_mapping['id2item'][i+1]

    Returns:
        融合后的embedding，shape=(N_items-1, 128)
    """

    # ========== 配置 ==========
    IMAGE_PATH_TEMPLATE = "/scratch/zl4789/MQL4GRec/data_process/MQL4GRec/{category}/{category}.emb-ViT-L-14.npy"
    USE_MULTIMODAL = self.config.get('use_multimodal', True)
    IMAGE_PCA_DIM = self.config.get('image_pca_dim', 128)
    FINAL_PCA_DIM = self.config.get('final_pca_dim', 128)
    FILL_STRATEGY = self.config.get('fill_missing_image', 'zero')  # zero / mean / clip

    if not USE_MULTIMODAL:
        self.logger.info('[TOKENIZER] Multimodal disabled, using text-only')
        return text_embs

    # ========== 1. 构建图像路径 ==========
    category = self.config.get('category', 'CDs_and_Vinyl')
    category_short = category.replace('_and_', '_')  # CDs_and_Vinyl → CDs_Vinyl
    image_path = IMAGE_PATH_TEMPLATE.format(category=category_short)

    if not os.path.exists(image_path):
        self.logger.warning(
            f'[TOKENIZER] Image embeddings not found at {image_path}. '
            'Falling back to text-only mode.'
        )
        return text_embs

    # ========== 2. 加载图像数据 ==========
    self.logger.info(f'[TOKENIZER] Loading image embeddings from {image_path}...')

    try:
        image_data = np.load(image_path, allow_pickle=True)

        # 尝试提取embedding和ASIN
        if isinstance(image_data, np.ndarray) and image_data.dtype == object:
            # 字典格式
            image_dict = image_data.item()
            image_embs_raw = np.array(image_dict['embeddings'])
            image_asins = list(image_dict['asins'])
            self.logger.info(
                f'[TOKENIZER] Loaded from dict: '
                f'{len(image_asins)} items, {image_embs_raw.shape[1]}D'
            )
        else:
            # 纯数组格式，需要外部ASIN
            image_embs_raw = image_data
            image_asins = self._load_external_asins(image_path)
            self.logger.info(
                f'[TOKENIZER] Loaded from array + external ASINs: '
                f'{len(image_asins)} items, {image_embs_raw.shape[1]}D'
            )

    except Exception as e:
        self.logger.error(f'[TOKENIZER] Failed to load image data: {e}')
        return text_embs

    # ========== 3. 构建ASIN映射 ==========
    asin2image_emb = {}
    for asin, emb in zip(image_asins, image_embs_raw):
        if isinstance(asin, bytes):
            asin = asin.decode('utf-8')
        asin2image_emb[str(asin)] = emb

    self.logger.info(f'[TOKENIZER] Built ASIN→Image mapping: {len(asin2image_emb)} items')

    # ========== 4. 精确对齐 ==========
    aligned_image_embs = []
    missing_asins = []
    stats = {'matched': 0, 'missing': 0}

    for i in range(1, dataset.n_items):  # 从1开始，跳过PAD
        asin = dataset.id_mapping['id2item'][i]
        asin_str = str(asin)

        if asin_str in asin2image_emb:
            # 找到匹配
            aligned_image_embs.append(asin2image_emb[asin_str])
            stats['matched'] += 1
        else:
            # 缺失处理
            if FILL_STRATEGY == 'zero':
                fill_emb = np.zeros(image_embs_raw.shape[1])
            elif FILL_STRATEGY == 'mean':
                fill_emb = np.mean(image_embs_raw, axis=0)
            else:  # 'clip' or others
                fill_emb = np.zeros(image_embs_raw.shape[1])

            aligned_image_embs.append(fill_emb)
            missing_asins.append(asin_str)
            stats['missing'] += 1

    aligned_image_embs = np.array(aligned_image_embs)

    self.logger.info(
        f'[TOKENIZER] Alignment complete: '
        f'{stats["matched"]} matched, {stats["missing"]} missing '
        f'({stats["missing"]*100/(stats["matched"]+stats["missing"]):.1f}%)'
    )

    if stats['missing'] > 0 and stats['missing'] <= 10:
        self.logger.debug(f'[TOKENIZER] Missing ASINs: {missing_asins}')

    # ========== 5. 验证对齐 ==========
    assert text_embs.shape[0] == aligned_image_embs.shape[0], \
        f'Alignment failed: text={text_embs.shape}, image={aligned_image_embs.shape}'

    # ========== 6. 图像PCA降维 ==========
    image_pca_cache = os.path.join(
        dataset.cache_dir, 'processed',
        f'image_pca_{IMAGE_PCA_DIM}_{FILL_STRATEGY}.npy'
    )

    if os.path.exists(image_pca_cache):
        self.logger.info('[TOKENIZER] Loading cached image PCA embeddings...')
        image_embs_reduced = np.load(image_pca_cache)
    else:
        self.logger.info(
            f'[TOKENIZER] Applying PCA to image embeddings: '
            f'{aligned_image_embs.shape[1]}D → {IMAGE_PCA_DIM}D'
        )

        # 只用非零向量训练PCA
        if FILL_STRATEGY == 'zero':
            non_zero_mask = ~np.all(aligned_image_embs == 0, axis=1)
            train_data = aligned_image_embs[non_zero_mask]
            self.logger.info(f'[TOKENIZER] Training PCA on {np.sum(non_zero_mask)} valid images')
        else:
            train_data = aligned_image_embs

        image_pca = PCA(n_components=IMAGE_PCA_DIM, whiten=True)
        image_pca.fit(train_data)
        image_embs_reduced = image_pca.transform(aligned_image_embs)

        # 缓存
        np.save(image_pca_cache, image_embs_reduced)
        self.logger.info(f'[TOKENIZER] Cached to {image_pca_cache}')

    # ========== 7. 融合 ==========
    self.logger.info('[TOKENIZER] Fusing text and image embeddings...')
    fused_embs = np.concatenate([text_embs, image_embs_reduced], axis=1)
    self.logger.info(
        f'[TOKENIZER] Fused shape: {fused_embs.shape} '
        f'(text:{text_embs.shape[1]}D + image:{image_embs_reduced.shape[1]}D)'
    )

    # ========== 8. 最终PCA ==========
    final_pca_cache = os.path.join(
        dataset.cache_dir, 'processed',
        f'multimodal_pca_{FINAL_PCA_DIM}_{FILL_STRATEGY}.npy'
    )

    if os.path.exists(final_pca_cache):
        self.logger.info('[TOKENIZER] Loading cached multimodal embeddings...')
        final_embs = np.load(final_pca_cache)
    else:
        self.logger.info(
            f'[TOKENIZER] Applying final PCA: '
            f'{fused_embs.shape[1]}D → {FINAL_PCA_DIM}D'
        )
        final_pca = PCA(n_components=FINAL_PCA_DIM, whiten=True)
        final_embs = final_pca.fit_transform(fused_embs)

        # 缓存
        np.save(final_pca_cache, final_embs)
        self.logger.info(f'[TOKENIZER] Cached to {final_pca_cache}')

    self.logger.info(
        f'[TOKENIZER] ✅ Multimodal fusion complete: {final_embs.shape}'
    )

    return final_embs


def _load_external_asins(self, image_npy_path: str) -> list[str]:
    """加载外部ASIN文件"""
    # 尝试 .txt 文件
    txt_path = image_npy_path.replace('.npy', '.asins.txt')
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            return [line.strip() for line in f if line.strip()]

    # 尝试 .json 文件
    json_path = image_npy_path.replace('.npy', '.asins.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)

    # 尝试从配置加载
    config_path = image_npy_path.replace('.npy', '.config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            if 'asins' in config:
                return config['asins']

    raise FileNotFoundError(
        f'Cannot find ASIN mapping for {image_npy_path}. '
        f'Tried: {txt_path}, {json_path}, {config_path}'
    )
```

---

## 测试与验证

### 验证脚本
```python
# scripts/validate_multimodal_alignment.py

import numpy as np
import json

def validate_alignment(dataset, final_embs):
    """验证对齐是否正确"""

    print("=" * 60)
    print("MULTIMODAL ALIGNMENT VALIDATION")
    print("=" * 60)

    # 1. 检查形状
    expected_shape = (dataset.n_items - 1, 128)
    actual_shape = final_embs.shape
    print(f"\n1. Shape Check:")
    print(f"   Expected: {expected_shape}")
    print(f"   Actual:   {actual_shape}")
    print(f"   ✅ PASS" if actual_shape == expected_shape else "   ❌ FAIL")

    # 2. 检查几个已知item
    print(f"\n2. Sample Item Check:")
    for i in [1, 10, 100]:
        if i < len(dataset.id_mapping['id2item']):
            asin = dataset.id_mapping['id2item'][i]
            emb = final_embs[i-1]
            print(f"   Item {i}: ASIN={asin}, emb_mean={emb.mean():.4f}, emb_std={emb.std():.4f}")

    # 3. 检查是否有异常值
    print(f"\n3. Anomaly Check:")
    zero_rows = np.all(final_embs == 0, axis=1).sum()
    nan_rows = np.any(np.isnan(final_embs), axis=1).sum()
    inf_rows = np.any(np.isinf(final_embs), axis=1).sum()

    print(f"   Zero vectors: {zero_rows}")
    print(f"   NaN vectors:  {nan_rows}")
    print(f"   Inf vectors:  {inf_rows}")
    print(f"   ✅ PASS" if (nan_rows == 0 and inf_rows == 0) else "   ❌ FAIL")

    # 4. 统计分布
    print(f"\n4. Distribution:")
    print(f"   Mean: {final_embs.mean():.4f}")
    print(f"   Std:  {final_embs.std():.4f}")
    print(f"   Min:  {final_embs.min():.4f}")
    print(f"   Max:  {final_embs.max():.4f}")

    print("\n" + "=" * 60)

# 使用
# validate_alignment(dataset, final_embs)
```

---

## 总结

### 推荐实施路径

1. **第一步：诊断**（1小时）
   - 运行诊断脚本，了解图像数据格式
   - 检查是否有ASIN信息
   - 统计缺失情况

2. **第二步：选择方案**（30分钟）
   - 如果有ASIN：方案一（精确对齐）
   - 如果无ASIN：联系数据提供方或使用方案二

3. **第三步：实现**（2-4小时）
   - 修改 `_load_and_fuse_image_embeddings` 函数
   - 添加 `_load_external_asins` 辅助函数
   - 更新配置文件

4. **第四步：测试**（1-2小时）
   - 运行验证脚本
   - 可视化检查
   - 小规模训练测试

5. **第五步：全量训练**
   - 清除旧的缓存文件
   - 重新运行完整pipeline
   - 监控日志输出

### 关键检查点

- ✅ 文本和图像embedding的item数量必须相同
- ✅ 每个位置的ASIN必须匹配（不能简单截断）
- ✅ 缺失图像需要合理填充（不能直接跳过）
- ✅ PCA训练时要排除填充的零向量
- ✅ 缓存文件要区分不同的填充策略

### 常见陷阱

1. **假设顺序一致**：文本和图像的顺序通常不同！
2. **简单截断**：会丢失大量数据
3. **忽略缺失项**：导致n_items不匹配
4. **缓存污染**：修改代码后要清除旧缓存
5. **ASIN编码问题**：注意bytes vs str的转换
