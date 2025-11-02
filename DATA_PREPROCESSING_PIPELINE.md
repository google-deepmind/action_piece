# ActionPiece 数据预处理Pipeline详解

## 目录
1. [概述](#概述)
2. [完整Pipeline流程图](#完整pipeline流程图)
3. [阶段一：原始数据加载](#阶段一原始数据加载)
4. [阶段二：语义Embedding生成（关键：128维向量）](#阶段二语义embedding生成关键128维向量)
5. [阶段三：特征量化与Hash](#阶段三特征量化与hash)
6. [阶段四：ActionPiece词表构建](#阶段四actionpiece词表构建)
7. [阶段五：Tokenization](#阶段五tokenization)
8. [多模态扩展（图像+文本融合）](#多模态扩展图像文本融合)
9. [关键代码路径](#关键代码路径)

---

## 概述

ActionPiece的数据预处理pipeline将原始的Amazon评论数据转换为适合T5模型训练的token序列。核心创新在于：
- **上下文感知的tokenization**：不是简单地对每个商品独立编码，而是考虑相邻商品的特征共现模式
- **语义特征提取**：使用Sentence-T5将商品元数据（标题、描述等）转换为128维语义向量
- **BPE-style合并**：通过频率驱动的特征合并构建层级词表

**128维向量的产生**：通过Sentence-T5 Base模型（768维）+ PCA降维到128维，最终用于Product Quantization生成离散特征ID。

---

## 完整Pipeline流程图

```
原始数据 (reviews + metadata)
    ↓
[阶段1] 数据加载与预处理
    ├─ 下载 Amazon Reviews 2014 数据
    ├─ 解析评论 (user, item, timestamp)
    ├─ 按时间排序构建用户行为序列
    └─ 提取商品元数据 (title, brand, description, etc.)
    ↓
[阶段2] 语义Embedding生成 ⭐ 128维向量产生
    ├─ 元数据拼接: "title price brand feature categories description"
    ├─ Sentence-T5 Base编码: 768维原始embedding
    ├─ PCA降维: 768维 → 128维 (sent_emb_pca=128)
    └─ (可选) 多模态融合: 128维文本 + 128维图像 → 256维 → 128维
    ↓
[阶段3] 特征量化与Hash
    ├─ Product Quantization (FAISS)
    │   ├─ 训练索引: OPQ + IVF1 + PQ (128维输入)
    │   ├─ 将128维向量量化为4个codebook ID (每个取值0-255)
    │   └─ 输出: item → (c0, c1, c2, c3) 的映射
    ├─ Hash冲突处理
    │   └─ 添加hash_bucket_id: (c0, c1, c2, c3, h) 避免相同语义ID的冲突
    └─ 最终item特征: 每个item表示为5维离散ID
    ↓
[阶段4] ActionPiece词表构建
    ├─ 初始化词表: 所有原子特征 (category_idx, feature_value)
    ├─ 统计特征对共现频率
    │   ├─ 同一item内: weight = 2/M
    │   └─ 相邻item间: weight = 1/(M1*M2)
    ├─ BPE-style迭代合并 (目标vocab_size=40000)
    │   └─ 每步选择频率最高的特征对合并为新token
    └─ 保存词表: actionpiece.json
    ↓
[阶段5] Tokenization
    ├─ 训练时: 随机排列特征顺序 (SPR正则化)
    ├─ 推理时: 按优先级贪心分段
    └─ 输出: input_ids, attention_mask, labels
    ↓
T5模型训练
```

---

## 阶段一：原始数据加载

### 1.1 数据下载
**位置**: `genrec/datasets/AmazonReviews2014/dataset.py:164-182`

```python
# 从Stanford SNAP下载两个文件
reviews_file = f"reviews_{category}_5.json.gz"  # 评论数据
meta_file = f"meta_{category}.json.gz"          # 商品元数据
```

### 1.2 评论数据处理
**位置**: `genrec/datasets/AmazonReviews2014/dataset.py:184-201`

```python
# 每条评论包含
{
    'reviewerID': 'A1234567',      # 用户ID
    'asin': 'B00ABC123',           # 商品ID
    'unixReviewTime': 1356048000   # Unix时间戳
}
```

### 1.3 构建用户行为序列
**位置**: `genrec/datasets/AmazonReviews2014/dataset.py:106-129`

```python
# 按用户分组，按时间排序
user_1 → [item_a, item_b, item_c, ...]  # 按时间顺序
user_2 → [item_x, item_y, ...]
```

### 1.4 数据划分 (Leave-One-Out)
**位置**: `genrec/dataset.py:117-141`

```python
# 假设用户有5个交互: [i1, i2, i3, i4, i5]
train: [i1, i2, i3]  # 用于训练
val:   [i1, i2, i3, i4]  # 预测i4
test:  [i1, i2, i3, i4, i5]  # 预测i5
```

---

## 阶段二：语义Embedding生成（关键：128维向量）

### 2.1 元数据拼接
**位置**: `genrec/datasets/AmazonReviews2014/dataset.py:329-356`

```python
# 从原始元数据提取6个字段
meta_sentence = ""
meta_sentence += title + " "
meta_sentence += price + " "
meta_sentence += brand + " "
meta_sentence += features + " "
meta_sentence += categories + " "
meta_sentence += description + " "

# 示例输出
"Sony WH-1000XM4 Wireless Headphones 349.99 Sony noise cancellation,
 30-hour battery Electronics > Audio > Headphones Premium wireless
 headphones with industry-leading noise cancellation..."
```

### 2.2 Sentence-T5 Base编码 ⭐ **第一步：768维**
**位置**: `genrec/models/ActionPiece/tokenizer.py:71-112`

**配置参数**:
```yaml
sent_emb_model: "sentence-transformers/sentence-t5-base"
sent_emb_batch_size: 512
sent_emb_dim: 768  # 原始输出维度
```

**代码实现**:
```python
# 加载预训练模型
sent_emb_model = SentenceTransformer('sentence-transformers/sentence-t5-base')

# 批量编码所有商品的元数据文本
meta_sentences = []  # N个商品的文本描述
for item_id in range(1, n_items):
    meta_sentences.append(item2meta[item_id])

# 编码：文本 → 768维向量
sent_embs = sent_emb_model.encode(
    meta_sentences,
    convert_to_numpy=True,
    batch_size=512,
    show_progress_bar=True
)
# 输出shape: (N_items, 768)
```

### 2.3 PCA降维 ⭐ **第二步：768维 → 128维**
**位置**: `genrec/models/ActionPiece/tokenizer.py:104-112`

**配置参数**:
```yaml
sent_emb_pca: 128  # PCA目标维度（-1表示不降维）
```

**代码实现**:
```python
from sklearn.decomposition import PCA

if config['sent_emb_pca'] > 0:  # 如果设置了PCA维度
    pca = PCA(n_components=128, whiten=True)
    sent_embs = pca.fit_transform(sent_embs)
    # 输出shape: (N_items, 128)

    # whiten=True的作用：
    # 1. 标准化主成分的方差为1
    # 2. 使不同维度的重要性均等化
    # 3. 提升后续Product Quantization的效果
```

**为什么是128维？**
1. **平衡效果与效率**：768维对于下游Product Quantization来说过于高维
2. **保留语义信息**：PCA保留最重要的128个主成分（通常能保留90%+方差）
3. **适配PQ设置**：128维可以均匀分配给4个codebook（每个32维）
4. **与T5模型对齐**：T5的d_model=128，便于embedding层初始化

### 2.4 缓存机制
**位置**: `genrec/models/ActionPiece/tokenizer.py:241-269`

```python
# 缓存路径
sent_emb_path = "cache/AmazonReviews2014/{category}/processed/sentence-t5-base.sent_emb"

# 首次运行：编码+降维+保存
# 后续运行：直接加载
if os.path.exists(sent_emb_path):
    sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, 128)
else:
    sent_embs = encode_and_pca(...)
    sent_embs.tofile(sent_emb_path)
```

---

## 阶段三：特征量化与Hash

### 3.1 Product Quantization (PQ)
**位置**: `genrec/models/ActionPiece/tokenizer.py:285-316`

**原理**：将连续的128维向量离散化为4个codebook索引

**配置参数**:
```yaml
pq_n_codebooks: 4      # 将128维分成4组
pq_codebook_size: 256  # 每个codebook有256个中心点 (2^8)
```

**FAISS索引构建**:
```python
# 128维向量分成4组，每组32维
# 每组有256个聚类中心（8 bits）
faiss_index = faiss.index_factory(
    128,  # 输入维度
    "OPQ4,IVF1,PQ4x8",  # 索引类型
    faiss.METRIC_INNER_PRODUCT
)

# OPQ4: Optimized Product Quantization with 4 codebooks
# IVF1: Inverted File Index (单个簇用于精确搜索)
# PQ4x8: 4个codebook，每个8 bits (256个中心)

# 只用训练集商品的embedding训练索引
training_embs = sent_embs[training_item_mask]  # shape: (M, 128)
faiss_index.train(training_embs)
faiss_index.add(sent_embs)  # 添加所有商品

# 提取量化后的code
codes = faiss_index.extract_codes()
# 输出shape: (N_items, 4)
# 每个商品 → [c0, c1, c2, c3]，其中 ci ∈ {0, 1, ..., 255}
```

**示例**:
```python
# Item: "Sony WH-1000XM4 Headphones"
128维向量: [0.12, -0.34, 0.56, ..., 0.78]
    ↓
PQ量化后: [123, 45, 201, 89]
    ↓ (4个codebook索引)
最终特征: (0, 123), (1, 45), (2, 201), (3, 89)
```

### 3.2 Hash冲突处理
**位置**: `genrec/models/ActionPiece/tokenizer.py:377-415`

**问题**：不同商品可能有相同的PQ code

**解决方案**：添加hash bucket ID
```python
n_hash_buckets = 128

# 对每个PQ code生成随机hash序列
feat2hash_ids = {}
for item, pq_code in item2feat.items():
    if pq_code not in feat2hash_ids:
        feat2hash_ids[pq_code] = np.random.permutation(128)

    # 根据该code已使用次数分配hash_id
    idx = count_of_this_code
    hash_id = feat2hash_ids[pq_code][idx]

    # 最终特征: (c0, c1, c2, c3, hash_id)
    item2hashed_feat[item] = (*pq_code, hash_id)

# 示例
Item A: PQ=[123,45,201,89] → (0,123), (1,45), (2,201), (3,89), (4,17)
Item B: PQ=[123,45,201,89] → (0,123), (1,45), (2,201), (3,89), (4,92)
# 通过不同的hash_id区分
```

### 3.3 最终Item特征表示
每个商品最终表示为5维离散特征：
```python
item_features = [
    (category_0, value_123),  # PQ codebook 0
    (category_1, value_45),   # PQ codebook 1
    (category_2, value_201),  # PQ codebook 2
    (category_3, value_89),   # PQ codebook 3
    (category_4, value_17)    # hash bucket
]
```

---

## 阶段四：ActionPiece词表构建

### 4.1 初始化词表
**位置**: `genrec/models/ActionPiece/core.py:132-157`

```python
# 从item特征提取所有原子特征
vocab = [(category_idx, feature_value)]

# 示例：如果有1000个不同的PQ code和128个hash bucket
# 初始vocab_size ≈ 4*256 + 128 = 1152
初始词表:
  token_0: [PAD]
  token_1: (0, 0)     # codebook_0, code_0
  token_2: (0, 1)     # codebook_0, code_1
  ...
  token_1024: (3, 255) # codebook_3, code_255
  token_1025: (4, 0)   # hash, bucket_0
  ...
```

### 4.2 BPE-Style迭代合并
**位置**: `genrec/models/ActionPiece/core.py:577-644`

**核心思想**：频繁共现的特征对应该被合并成新token

**权重计算**:
```python
# 同一item内的特征对
weight_within = 2 / M  # M是item的token数

# 相邻item间的特征对
weight_between = 1 / (M1 * M2)  # M1, M2是两个item的token数
```

**训练过程**:
```python
# 目标词表大小
target_vocab_size = 40000

while len(vocab) < target_vocab_size:
    # 1. 统计所有特征对的频率
    pair2count = count_all_pairs(user_sequences)

    # 2. 选择频率最高的特征对
    (token_i, token_j), max_count = find_max_pair()

    # 3. 创建新token
    new_token = len(vocab)
    vocab.append((-1, token_i, token_j))  # 合并规则
    priority.append(max_count)  # 记录优先级

    # 4. 更新所有包含该特征对的序列
    for sequence in sequences_with_pair:
        replace(token_i, token_j → new_token)
```

**示例**:
```python
# 迭代1: 发现 token_123 和 token_456 频繁共现
vocab[1153] = (-1, 123, 456)  # 新合并token
priority[1153] = 10542.3      # 共现频率

# 迭代2: token_1153 和 token_789 频繁共现
vocab[1154] = (-1, 1153, 789)  # 层级合并
priority[1154] = 8321.7

# 最终：token_1154 可以解码为 [token_123, token_456, token_789]
```

### 4.3 优先级与解码
**位置**: `genrec/models/ActionPiece/core.py:798-852`

```python
# 每个token有优先级（训练时的合并频率）
priority = [0, 0, ..., 10542.3, 8321.7, ...]

# 解码：递归展开合并token
def decode_token(token_id):
    rule = vocab[token_id]
    if rule[0] != -1:  # 原子特征
        return [rule]
    else:  # 合并token
        return decode_token(rule[1]) + decode_token(rule[2])

# 示例
decode_token(1154) → decode_token(1153) + decode_token(789)
                   → [token_123, token_456, token_789]
```

---

## 阶段五：Tokenization

### 5.1 训练时：Set Permutation Regularization (SPR)
**位置**: `genrec/models/ActionPiece/tokenizer.py:564-602`

**目的**：防止模型依赖特征的顺序

```python
def collate_fn_train(batch):
    for item_seq in batch:
        # 对每个item的特征进行随机排列
        shuffled_features = random_permutation(item_features)
        # 编码
        tokens = actionpiece.encode(shuffled_features, shuffle='feature')

# 示例
原始特征: [(0,123), (1,45), (2,201), (3,89), (4,17)]
随机排列: [(2,201), (4,17), (0,123), (1,45), (3,89)]
编码结果: [token_123, token_456, token_789]  # BPE-style合并
```

### 5.2 推理时：贪心最优分段
**位置**: `genrec/models/ActionPiece/core.py:697-796`

```python
def encode(item_seq, shuffle='none'):
    # 不进行随机排列
    # 每次选择优先级最高的可合并token对
    while can_merge():
        best_pair = find_highest_priority_pair()
        merge(best_pair)
    return token_seq

# 示例
item_features: [(0,123), (1,45), (2,201), (3,89), (4,17)]
优先级计算:
  - (token_123, token_456): priority=10542.3 ← 最高
  - (token_789, token_101): priority=8321.7
  ...
编码结果: [token_1153, token_789, ...]  # 使用高优先级合并token
```

### 5.3 最终输出格式
**位置**: `genrec/models/ActionPiece/tokenizer.py:564-602`

```python
# 单个batch的输出
{
    'input_ids': [
        [BOS, token_1, token_2, ..., token_n, EOS, PAD, PAD],
        [BOS, token_5, token_6, ..., EOS, PAD, PAD, PAD],
        ...
    ],  # shape: (batch_size, max_seq_len)

    'attention_mask': [
        [1, 1, 1, ..., 1, 1, 0, 0],
        [1, 1, 1, ..., 1, 0, 0, 0],
        ...
    ],  # 1表示有效token，0表示padding

    'labels': [
        [token_next1, token_next2, ..., EOS, -100, -100],
        [token_next3, token_next4, ..., EOS, -100, -100],
        ...
    ]  # -100是ignored_label，用于loss计算
}
```

---

## 多模态扩展（图像+文本融合）

### 6.1 图像Embedding加载
**位置**: `genrec/models/ActionPiece/tokenizer.py:114-240`

```python
# 硬编码配置（当前实现）
IMAGE_PATH_TEMPLATE = "/path/to/MQL4GRec/{category}/{category}.emb-ViT-L-14.npy"
USE_MULTIMODAL = True
IMAGE_PCA_DIM = 128
FINAL_PCA_DIM = 128

# 加载预计算的图像embedding (通常是CLIP ViT-L/14)
image_embs = np.load(image_path)  # shape: (N_items, 768)
```

### 6.2 图像PCA降维
```python
# 图像embedding也降到128维
image_pca = PCA(n_components=128, whiten=True)
image_embs_reduced = image_pca.fit_transform(image_embs)
# shape: (N_items, 128)
```

### 6.3 文本+图像融合
```python
# 拼接
fused_embs = np.concatenate([text_embs, image_embs_reduced], axis=1)
# shape: (N_items, 256)  # 128文本 + 128图像

# 再次PCA降维到128维
final_pca = PCA(n_components=128, whiten=True)
final_embs = final_pca.fit_transform(fused_embs)
# shape: (N_items, 128)  # 最终多模态embedding
```

### 6.4 多模态Pipeline对比
```
纯文本模式:
  元数据文本 → Sentence-T5 (768维) → PCA (128维) → PQ → 离散特征

多模态模式:
  元数据文本 → Sentence-T5 (768维) → PCA (128维) ┐
                                                  ├→ 拼接 (256维) → PCA (128维) → PQ → 离散特征
  商品图像 → CLIP ViT-L/14 (768维) → PCA (128维) ┘
```

---

## 关键代码路径

### Pipeline入口
```
genrec/pipeline.py:161-233
  ├─ Dataset初始化
  ├─ Tokenizer初始化 (触发所有预处理)
  ├─ Model初始化
  └─ Trainer训练循环
```

### 数据加载
```
genrec/datasets/AmazonReviews2014/dataset.py
  ├─ _download_raw (164-182): 下载原始数据
  ├─ _load_reviews (184-201): 解析评论
  ├─ _load_metadata (280-300): 解析元数据
  ├─ _extract_meta_sentences (329-356): 拼接元数据文本
  └─ _download_and_process_raw (400-430): 主流程
```

### 语义Embedding (⭐ 128维生成)
```
genrec/models/ActionPiece/tokenizer.py
  ├─ _encode_sent_emb (71-112):
  │   ├─ Sentence-T5编码 (768维)
  │   └─ PCA降维 (128维)
  ├─ _load_and_fuse_image_embeddings (114-240):
  │   ├─ 加载图像embedding
  │   ├─ 图像PCA (128维)
  │   └─ 多模态融合 (256→128维)
  └─ _get_sent_embs (241-269): 缓存管理
```

### Product Quantization
```
genrec/models/ActionPiece/tokenizer.py
  ├─ _sent_emb_to_sem_id (285-316):
  │   └─ FAISS索引训练 (128维 → 4个codebook ID)
  ├─ _get_hashed_feat (377-415):
  │   └─ 添加hash_bucket_id避免冲突
  └─ _get_item2feat (417-447): 完整流程
```

### ActionPiece词表构建
```
genrec/models/ActionPiece/core.py
  ├─ train (577-601): BPE训练主循环
  ├─ _train_step (603-644): 单步合并
  ├─ _count_pairs_inside_state (233-248): 同item内权重
  └─ _count_pairs_btw_states (250-258): 相邻item间权重
```

### Tokenization
```
genrec/models/ActionPiece/tokenizer.py
  ├─ collate_fn_train (564-602): 训练时SPR
  ├─ collate_fn_test (645-689): 推理时贪心分段
  └─ tokenize_function (475-493): 序列截断与处理
```

### 模型
```
genrec/models/ActionPiece/model.py
  ├─ forward (78-91): T5 Encoder-Decoder
  ├─ generate (93-155): Beam Search + Ensemble
  └─ beam_search (157-247): 自定义beam search实现
```

---

## 完整数据流示例

假设有一个用户的历史行为序列：

### 输入（原始数据）
```python
user_123 交互了 3 个商品 [item_A, item_B, item_C]

# 商品元数据
item_A: "Sony WH-1000XM4 Wireless Headphones 349.99 ..."
item_B: "Apple AirPods Pro 249.99 ..."
item_C: "Bose QuietComfort 35 II 299.99 ..."
```

### 阶段1：元数据编码 → 128维
```python
item_A: text → Sentence-T5 → [0.12, -0.34, 0.56, ..., 0.78]  # 768维
                           ↓ PCA
                           → [0.23, 0.45, ..., -0.12]  # 128维

item_B: text → ... → [0.15, 0.67, ..., 0.34]  # 128维
item_C: text → ... → [-0.43, 0.21, ..., 0.89]  # 128维
```

### 阶段2：Product Quantization → 离散特征
```python
item_A: [0.23, 0.45, ..., -0.12] → PQ → [123, 45, 201, 89] → hash → [123,45,201,89,17]
item_B: [0.15, 0.67, ..., 0.34] → PQ → [98, 156, 23, 211] → hash → [98,156,23,211,42]
item_C: [-0.43, 0.21, ..., 0.89] → PQ → [45, 201, 67, 134] → hash → [45,201,67,134,88]
```

### 阶段3：特征到Token
```python
# 每个商品的5维特征 → 初始token序列
item_A: [(0,123), (1,45), (2,201), (3,89), (4,17)]
        ↓ 查找rank映射
        [token_123, token_456, token_789, token_234, token_567]

item_B: [token_345, token_678, token_901, token_111, token_222]
item_C: [token_999, token_888, token_777, token_666, token_555]
```

### 阶段4：ActionPiece编码（BPE合并）
```python
# 训练时（随机排列）
[token_123, token_456, token_789, token_234, token_567]
    ↓ shuffle
[token_789, token_567, token_123, token_456, token_234]
    ↓ BPE合并
[token_10001, token_20002]  # 合并后的token

# 推理时（贪心最优）
[token_123, token_456, token_789, token_234, token_567]
    ↓ 选择最高优先级合并
[token_10001, token_789, token_567]
```

### 阶段5：最终输入T5
```python
# 完整序列
input_ids = [
    BOS,                           # 起始token
    token_10001, token_789, ...,   # item_A的编码
    context_token_1,                # item_A和item_B之间的上下文
    token_20002, token_456, ...,   # item_B的编码
    context_token_2,                # item_B和item_C之间的上下文
    token_30003, token_123, ...,   # item_C的编码
    EOS                            # 结束token
]

# 预测目标：下一个商品item_D的token表示
labels = [token_40004, token_555, ..., EOS]
```

---

## 总结：128维向量的完整生成路径

```
商品元数据文本
    ↓
Sentence-T5 Base模型
    ↓
768维embedding
    ↓
PCA (whiten=True)
    ↓
⭐ 128维语义向量 ⭐ ← 这是关键输出
    ↓
(可选) 与128维图像embedding融合 → 256维 → PCA → 128维
    ↓
FAISS Product Quantization (OPQ4,IVF1,PQ4x8)
    ↓
4个离散codebook ID (每个0-255)
    ↓
添加hash_bucket_id (0-127)
    ↓
5维离散特征表示
    ↓
ActionPiece BPE-style词表构建
    ↓
Token序列
    ↓
T5模型训练
```

**关键配置**:
- `sent_emb_model: sentence-transformers/sentence-t5-base`
- `sent_emb_dim: 768` (原始)
- `sent_emb_pca: 128` ⭐ (降维目标)
- `pq_n_codebooks: 4` (量化)
- `pq_codebook_size: 256`
- `n_hash_buckets: 128`
- `d_model: 128` (T5模型维度，与PCA维度对齐)

**为什么选择128维？**
1. **计算效率**: 768维 → 128维减少6倍存储和计算
2. **语义保留**: PCA保留最重要的主成分（通常90%+方差）
3. **量化友好**: 128 = 4 × 32，可均匀分配给4个PQ codebook
4. **模型对齐**: 与T5的d_model=128一致，便于embedding初始化
5. **实验验证**: 作者在ICML 2025论文中验证了128维的有效性
