# Amazon Reviews Dataset Processing Flow

This document provides a detailed explanation of the data processing pipeline for Amazon Reviews datasets (both 2014 and 2018 versions) in the ActionPiece project.

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [File Structure](#file-structure)
3. [Data Processing Pipeline](#data-processing-pipeline)
4. [Key Differences: 2014 vs 2018](#key-differences-2014-vs-2018)
5. [Detailed Function Breakdown](#detailed-function-breakdown)
6. [Configuration and Caching](#configuration-and-caching)

---

## Dataset Overview

The project supports two versions of Amazon Reviews datasets:

### Amazon Reviews 2014

- **Source**: Stanford SNAP (https://snap.stanford.edu/data/amazon/productGraph/)
- **Format**: Reviews in JSON, Metadata in Python dict format (single quotes)
- **Coverage**: ~492,799 records (e.g., CDs_and_Vinyl)
- **Key Fields**: `categories` (nested lists), `related` (dict with also_bought, etc.), `imUrl`

### Amazon Reviews 2018

- **Source**: UCSD McAuley Lab (https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/)
- **Format**: Both reviews and metadata in standard JSON (double quotes)
- **Coverage**: ~516,914 records (e.g., CDs_and_Vinyl)
- **Key Fields**: `category` (flat list), `also_buy`/`also_view` (top-level), `imageURL`/`imageURLHighRes`

---

## File Structure

```
genrec/datasets/
├── AmazonReviews2014/
│   ├── dataset.py              # 2014 dataset implementation
│   └── config.yaml             # Configuration for 2014 version
└── AmazonReviews2018/
    ├── dataset.py              # 2018 dataset implementation
    └── config.yaml             # Configuration for 2018 version

cache/
└── AmazonReviews{2014|2018}/
    └── {category}/
        ├── raw/                # Downloaded raw files
        │   ├── reviews_{category}_5.json.gz
        │   └── meta_{category}.json.gz
        └── processed/          # Processed data
            ├── all_item_seqs.json
            ├── id_mapping.json
            └── metadata.{mode}.json
```

---

## Data Processing Pipeline

The data processing pipeline consists of several stages:

```
┌─────────────────────────────────────────────────────────────┐
│                    1. INITIALIZATION                        │
│  - Load config                                              │
│  - Check category availability                              │
│  - Set cache directories                                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    2. DOWNLOAD RAW DATA                     │
│  - Download reviews file (reviews_{category}_5.json.gz)     │
│  - Download metadata file (meta_{category}.json.gz)         │
│  - Cache to: cache/{version}/{category}/raw/               │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  3. PROCESS REVIEWS                         │
│  Step 3a: Load Reviews                                      │
│    - Parse gzipped JSON lines                               │
│    - Extract (user, item, timestamp) tuples                 │
│                                                             │
│  Step 3b: Group by User & Sort                              │
│    - Group interactions by user ID                          │
│    - Sort each user's items by timestamp                    │
│                                                             │
│  Step 3c: Remap IDs                                         │
│    - Assign sequential IDs to users and items               │
│    - Create bidirectional mappings (ID ↔ raw token)         │
│    - Save: all_item_seqs.json, id_mapping.json             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  4. PROCESS METADATA                        │
│  Step 4a: Load Metadata                                     │
│    - Parse gzipped JSON lines                               │
│    - Filter by items present in reviews                     │
│                                                             │
│  Step 4b: Extract Meta Sentences (if mode = 'sentence')     │
│    - Process: title, brand, price, features, categories     │
│    - Convert to text sentences with clean_text()            │
│    - Save: metadata.{mode}.json                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  5. READY FOR TRAINING                      │
│  Outputs:                                                   │
│  - self.all_item_seqs: {user_id: [item_ids]}               │
│  - self.id_mapping: {user2id, item2id, id2user, id2item}   │
│  - self.item2meta: {item_asin: meta_text}                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Differences: 2014 vs 2018

### 1. Data Format

| Aspect | 2014 | 2018 |
|--------|------|------|
| **Reviews format** | Standard JSON | Standard JSON |
| **Metadata format** | Python dict (single quotes) | Standard JSON (double quotes) |
| **Parsing method** | `ast.literal_eval()` for metadata | `json.loads()` for both |

### 2. Field Mapping

| Field Type | 2014 | 2018 |
|------------|------|------|
| **Categories** | `categories` (nested list) | `category` (flat list) |
| **Related items** | `related` dict with `also_bought`, `bought_together`, `buy_after_viewing` | Top-level `also_buy`, `also_view` |
| **Images** | `imUrl` (single URL string) | `imageURL`, `imageURLHighRes` (arrays) |
| **Price** | Float number (e.g., 12.99) | String (e.g., "$12.99" or empty) |
| **Description** | String | List of strings (often empty) |
| **Brand** | String (sparse, ~10% coverage) | String (full coverage but often empty) |
| **Sales rank** | `salesRank` | `rank` (string or list) |
| **New in 2018** | - | `details` (dict), `similar_item`, `tech1`/`tech2`, `fit`, `date`, `main_cat` |

### 3. Data Type Handling

**2014 version (`_sent_process`):**
- `price`: Numeric (float) → convert to string
- `description`: Single string → apply `clean_text()`
- `categories`: Nested lists → flatten with commas

**2018 version (`_sent_process`):**
- `price`: String (may be empty or "$12.99") → apply `clean_text()`
- `description`: List of strings (often []) → iterate and clean each
- `category`: Flat list of strings → iterate and clean each

---

## Detailed Function Breakdown

### 1. `check_available_category(category: str)`

**Purpose**: Validates that the requested category exists in the dataset.

**Logic**:
```python
available_categories = ['Books', 'Electronics', 'Movies_and_TV', ...]
assert category in available_categories
```

**24 available categories** (shared across both 2014 and 2018):
- Books, Electronics, Movies_and_TV, CDs_and_Vinyl, Clothing_Shoes_and_Jewelry
- Home_and_Kitchen, Kindle_Store, Sports_and_Outdoors, Cell_Phones_and_Accessories
- Health_and_Personal_Care, Toys_and_Games, Video_Games, Tools_and_Home_Improvement
- Beauty, Apps_for_Android, Office_Products, Pet_Supplies, Automotive
- Grocery_and_Gourmet_Food, Patio_Lawn_and_Garden, Baby, Digital_Music
- Musical_Instruments, Amazon_Instant_Video

---

### 2. `parse_gz(path: str)`

**Purpose**: Parse gzipped data files line-by-line.

**2014 Implementation**:
```python
# Try JSON first (for reviews), fallback to ast.literal_eval (for metadata)
try:
    yield json.loads(line)
except json.JSONDecodeError:
    yield ast.literal_eval(line)
```

**2018 Implementation**:

```python
# Both reviews and metadata are standard JSON
yield json.loads(line)
```

**Why the difference?**
- 2014 metadata uses Python dict format: `{'asin': '0001501348', ...}`
- 2018 metadata uses standard JSON: `{"category": ["CDs & Vinyl"], ...}`

---

### 3. `get_item_seqs(reviews: Sequence[tuple[str, str, int]])`

**Purpose**: Group reviews by user and sort items chronologically.

**Input**: List of `(user_id, item_id, timestamp)` tuples

**Process**:
1. Group reviews by `user_id` using `collections.defaultdict(list)`
2. For each user, collect `(item, time)` pairs
3. Sort by timestamp: `item_time.sort(key=lambda x: x[1])`
4. Extract item sequence: `[item for item, _ in item_time]`

**Output**: `{user_id: [item1, item2, item3, ...]}`

**Example**:
```python
Input:
[
  ("U1", "I1", 1609459200),
  ("U1", "I2", 1609545600),
  ("U2", "I1", 1609632000),
]

Output:
{
  "U1": ["I1", "I2"],
  "U2": ["I1"]
}
```

---

### 4. `_download_raw(path: str, file_type: str)`

**Purpose**: Download raw data from online sources.

**URLs**:

**2014**:
```python
reviews: f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{category}_5.json.gz'
meta:    f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_{category}.json.gz'
```

**2018**:
```python
reviews: f'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/{category}_5.json.gz'
meta:    f'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_{category}.json.gz'
```

**Behavior**:
- Check if file already exists locally
- If not, download using `download_file()` utility
- Return local file path

---

### 5. `_load_reviews(path: str)`

**Purpose**: Load review interactions from gzipped file.

**Process**:
1. Parse each line using `parse_gz()`
2. Extract fields:
   - `reviewerID` → user ID
   - `asin` → item ID (Amazon Standard Identification Number)
   - `unixReviewTime` → timestamp
3. Convert timestamp to int and append to list

**Output**: `[(user1, item1, time1), (user2, item2, time2), ...]`

**Example review JSON**:
```json
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband...",
  "overall": 5.0,
  "summary": "Great CD",
  "unixReviewTime": 1362268800,
  "reviewTime": "03 3, 2013"
}
```

---

### 6. `_remap_ids(item_seqs: dict[str, list[str]])`

**Purpose**: Convert raw string IDs to sequential integer IDs.

**Why?**
- Raw IDs (e.g., "A2SUAM1J3GNN3B", "0000013714") are not suitable for embedding layers
- Sequential IDs (0, 1, 2, ...) enable efficient lookup in embedding tables
- ID 0 is reserved for padding `[PAD]`

**Process**:
1. Initialize mappings:
   ```python
   user2id = {}
   item2id = {}
   id2user = []
   id2item = []
   ```
2. For each user and their item sequence:
   - If user not seen before, assign next available ID
   - For each item in sequence:
     - If item not seen before, assign next available ID
     - Collect remapped item IDs

**Output**:
- `all_item_seqs`: `{raw_user_id: [raw_item_ids]}`
- `id_mapping`:
  - `user2id`: `{raw_user_id → int_user_id}`
  - `item2id`: `{raw_item_id → int_item_id}`
  - `id2user`: `[raw_user_ids]` (index = int_user_id)
  - `id2item`: `[raw_item_ids]` (index = int_item_id)

**Example**:
```python
Before remapping:
{
  "A2SUAM1J3GNN3B": ["0000013714", "0001501348"],
  "A3J8K5K5J2J3J2": ["0000013714"]
}

After remapping:
all_item_seqs = {
  "A2SUAM1J3GNN3B": ["0000013714", "0001501348"],
  "A3J8K5K5J2J3J2": ["0000013714"]
}

id_mapping = {
  "user2id": {"A2SUAM1J3GNN3B": 0, "A3J8K5K5J2J3J2": 1},
  "item2id": {"0000013714": 0, "0001501348": 1},
  "id2user": ["A2SUAM1J3GNN3B", "A3J8K5K5J2J3J2"],
  "id2item": ["0000013714", "0001501348"]
}
```

---

### 7. `_process_reviews(input_path: str, output_path: str)`

**Purpose**: Orchestrate the review processing pipeline.

**Steps**:
1. **Check cache**: If `all_item_seqs.json` and `id_mapping.json` exist, load and return
2. **Load reviews**: Parse gzipped file
3. **Group and sort**: Create user-item sequences
4. **Remap IDs**: Convert to integer IDs
5. **Save to cache**: Write JSON files

**Output files**:
- `all_item_seqs.json`: User → item sequence mapping
- `id_mapping.json`: ID conversion dictionaries

---

### 8. `_load_metadata(path: str, item2id: dict[str, int])`

**Purpose**: Load item metadata and filter by items present in reviews.

**Why filter?**
- The metadata file contains millions of items
- We only need metadata for items that appear in user reviews
- Filtering reduces memory usage and processing time

**Process**:
1. Extract item ASINs from `item2id.keys()`
2. For each metadata entry in gzipped file:
   - Check if `asin` is in our item set
   - If yes, store the full metadata dict
3. Return filtered metadata

**Output**: `{asin: {metadata_dict}}`

**Example metadata (2018)**:
```json
{
  "category": ["CDs & Vinyl", "Pop"],
  "tech1": "",
  "description": [],
  "fit": "",
  "title": "Turn It On Again: The Hits (The Tour Edition)",
  "also_buy": ["B000002UAE", "B00004ZA8C"],
  "tech2": "",
  "brand": "",
  "feature": ["CD (2007-06-26)"],
  "rank": "618,143 in CDs & Vinyl (See Top 100 in CDs & Vinyl)",
  "also_view": ["B000002UAE", "B00004ZA8C", "B00005NOKP"],
  "main_cat": "CDs & Vinyl",
  "similar_item": "",
  "date": "2007-06-26",
  "price": "",
  "asin": "B000OTRZ3W",
  "imageURL": ["https://images-na.ssl-images-amazon.com/..."],
  "imageURLHighRes": ["https://images-na.ssl-images-amazon.com/..."]
}
```

---

### 9. `_sent_process(raw)`

**Purpose**: Convert raw metadata values to clean text sentences.

**Handles multiple data types**:

#### 2018 Version Logic:

1. **None or empty** → return `""`
   ```python
   if raw is None or (isinstance(raw, (list, str)) and not raw):
       return ''
   ```

2. **Numeric values** → convert to string with period
   ```python
   if isinstance(raw, (int, float)):
       return str(raw) + '. '
   ```

3. **Nested lists** (backward compatibility with 2014 `categories`)
   ```python
   # Input: [["CDs & Vinyl", "Pop"], ["Music", "Rock"]]
   # Output: "CDs & Vinyl, Pop, Music, Rock. "
   elif isinstance(raw, list) and raw and isinstance(raw[0], list):
       sentence = ''
       for sublist in raw:
           for item in sublist:
               sentence += clean_text(str(item))[:-1] + ', '
       return sentence[:-2] + '. '
   ```

4. **Flat lists** (2018 `category`, `feature`, `description`)
   ```python
   # Input: ["CDs & Vinyl", "Pop"]
   # Output: "CDs & Vinyl. Pop. "
   elif isinstance(raw, list):
       sentence = ''
       for item in raw:
           sentence += clean_text(str(item))
       return sentence
   ```

5. **Strings** (2018 `title`, `brand`, `price`)
   ```python
   # Input: "Turn It On Again: The Hits"
   # Output: "Turn It On Again: The Hits. "
   else:
       return clean_text(str(raw))
   ```

**clean_text() utility**:
- Removes special characters
- Normalizes whitespace
- Adds period if missing

---

### 10. `_extract_meta_sentences(metadata: dict[str, Any])`

**Purpose**: Convert structured metadata to natural language sentences for each item.

**2014 Fields**:
```python
features_needed = [
    'title',       # String
    'price',       # Float
    'brand',       # String
    'feature',     # List of strings
    'categories',  # Nested list [[cat1, subcat1], [cat2, subcat2]]
    'description', # String
]
```

**2018 Fields**:
```python
features_needed = [
    'title',       # String
    'brand',       # String (often empty)
    'price',       # String like "$12.99" or ""
    'feature',     # List of strings
    'category',    # Flat list ["CDs & Vinyl", "Pop"]  ← Changed!
    'description', # List of strings (often [])        ← Changed!
]
```

**Process**:
1. For each item in metadata:
2. For each feature in `features_needed`:
   - Check if feature exists in metadata
   - Process value using `_sent_process()`
   - Concatenate to meta sentence
3. Store final sentence in `item2meta` dict

**Example output**:
```python
{
  "B000OTRZ3W": "Turn It On Again: The Hits (The Tour Edition). CD (2007-06-26). CDs & Vinyl. Pop. ",
  "B000002UAE": "...And Then There Were Three.... Genesis. $9.99. CD (1998-09-22). CDs & Vinyl. Rock. "
}
```

**Why extract sentences?**
- Models can use item metadata as additional features
- Natural language format allows text encoders (BERT, T5) to process
- Improves cold-start recommendations for new items

---

### 11. `_process_meta(input_path: str, output_path: str)`

**Purpose**: Process metadata according to specified mode.

**Three modes**:

1. **mode='none'**:
   - No metadata processing
   - Return `None`

2. **mode='raw'**:
   - Load metadata as-is
   - Return raw dictionaries
   - Used for custom feature extraction

3. **mode='sentence'**:
   - Load metadata
   - Extract text sentences using `_extract_meta_sentences()`
   - Return `{asin: sentence_string}`
   - Most commonly used mode

**Caching**:
- Saves processed metadata to `metadata.{mode}.json`
- Reuses cached file if available
- Different modes create different cache files

**Output**: `{asin: metadata}` or `None`

---

### 12. `_download_and_process_raw()`

**Purpose**: Main orchestration method that runs the entire pipeline.

**Flow**:
```python
# 1. Create directories
raw_data_path = cache_dir/raw
processed_data_path = cache_dir/processed

# 2. Download raw files (only on main process if distributed)
with self.accelerator.main_process_first():
    reviews_localpath = self._download_raw(path=raw_data_path, file_type='reviews')
    meta_localpath = self._download_raw(path=raw_data_path, file_type='meta')

# 3. Set random seed for reproducibility
np.random.seed(12345)

# 4. Process reviews
self.all_item_seqs, self.id_mapping = self._process_reviews(
    input_path=reviews_localpath,
    output_path=processed_data_path
)

# 5. Process metadata
self.item2meta = self._process_meta(
    input_path=meta_localpath,
    output_path=processed_data_path
)
```

**Important notes**:
- Uses `self.accelerator.main_process_first()` to prevent race conditions in distributed training
- Only main process downloads files; other processes wait
- All processes can safely read cached files

---

## Configuration and Caching

### Configuration Parameters

**Key config options**:
```yaml
dataset: AmazonReviews2018
category: CDs_and_Vinyl      # Which product category
metadata: sentence           # Processing mode: 'none', 'raw', 'sentence'
cache_dir: ./cache          # Where to cache processed data
```

### Cache Strategy

**Why cache?**
- Downloading takes time (files are 1-5GB)
- Parsing gzipped JSON is slow
- ID remapping requires full dataset scan
- Metadata extraction is computation-heavy

**What gets cached?**
1. **Raw downloads**: `cache/{version}/{category}/raw/`
   - `reviews_{category}_5.json.gz`
   - `meta_{category}.json.gz`

2. **Processed data**: `cache/{version}/{category}/processed/`
   - `all_item_seqs.json` (user sequences)
   - `id_mapping.json` (ID conversions)
   - `metadata.{mode}.json` (processed metadata)

**Cache invalidation**:
- Manual: Delete cached files to force reprocessing
- Automatic: Checks if files exist before recomputing

### Data Statistics

**Example for CDs_and_Vinyl**:

| Metric | 2014 | 2018 |
|--------|------|------|
| Reviews | ~1,097,592 | ~1,443,523 |
| Users | ~75,258 | ~103,144 |
| Items | ~486,360 | ~516,914 |
| Metadata entries | ~492,799 | ~516,914 |
| Sparsity | ~99.997% | ~99.997% |

---

## Usage Example

```python
from genrec.datasets.AmazonReviews2018.dataset import AmazonReviews2018

config = {
    'category': 'CDs_and_Vinyl',
    'metadata': 'sentence',
    'cache_dir': './cache',
    # ... other config params
}

# Initialize dataset (will download and process if needed)
dataset = AmazonReviews2018(config)

# Access processed data
user_sequences = dataset.all_item_seqs  # {user_id: [item_ids]}
id_mapping = dataset.id_mapping         # Conversion dicts
item_metadata = dataset.item2meta       # {asin: meta_sentence}

# Example: Get sequence for user "A2SUAM1J3GNN3B"
user_seq = dataset.all_item_seqs["A2SUAM1J3GNN3B"]
# Output: ["0000013714", "0001501348", ...]

# Get metadata for an item
meta_text = dataset.item2meta["0000013714"]
# Output: "Turn It On Again: The Hits. CD (2007-06-26). CDs & Vinyl. Pop. "
```

---

## Error Handling

**Common issues and solutions**:

1. **Parsing errors**:
   - Logged with line number and preview
   - Only first 10 errors shown to avoid spam
   - Continues processing despite errors

2. **Missing fields in metadata**:
   - Checks field existence before processing: `if feature in keys`
   - Handles empty values gracefully in `_sent_process()`

3. **Network issues during download**:
   - `download_file()` utility handles retries
   - Check internet connection and URLs

4. **Disk space**:
   - Ensure sufficient space for downloads (~5GB per category)
   - Cache can be cleared manually if needed

---

## Performance Considerations

**Optimization tips**:

1. **Use caching**: First run is slow, subsequent runs are instant
2. **Distributed training**: Only main process downloads files
3. **Metadata mode**:
   - Use `'none'` if metadata not needed (fastest)
   - Use `'raw'` for custom processing
   - Use `'sentence'` for text-based models

4. **Memory usage**:
   - Loading full metadata can use several GB
   - Filtering by `item2id` reduces memory footprint
   - Consider processing in batches for very large datasets

---

## References

- **2014 Dataset Paper**: "Image-based recommendations on styles and substitutes" (Julian McAuley et al., SIGIR 2015)
- **2018 Dataset Paper**: "Justifying recommendations using distantly-labeled reviews and fine-grained aspects" (Jianmo Ni et al., EMNLP 2019)
- **ActionPiece Paper**: "ActionPiece: Contextual Action Sequence Tokenization for Generative Recommendation" (ICML 2025 Spotlight)

---

## Appendix: Field Reference

### Review Fields (Both 2014 and 2018)

| Field | Type | Description |
|-------|------|-------------|
| `reviewerID` | string | Unique user ID |
| `asin` | string | Amazon Standard Identification Number (item ID) |
| `reviewerName` | string | User's display name |
| `helpful` | [int, int] | Helpfulness votes [helpful, total] |
| `reviewText` | string | Full review text |
| `overall` | float | Rating (1.0 - 5.0) |
| `summary` | string | Review title/summary |
| `unixReviewTime` | int | Unix timestamp |
| `reviewTime` | string | Human-readable date |

### Metadata Fields

#### 2014 Only

| Field | Type | Description |
|-------|------|-------------|
| `categories` | [[string]] | Nested category paths |
| `related` | dict | `{also_bought: [], bought_together: [], buy_after_viewing: []}` |
| `imUrl` | string | Single image URL |
| `salesRank` | dict | Sales rank by category |

#### 2018 Only

| Field | Type | Description |
|-------|------|-------------|
| `category` | [string] | Flat category list |
| `also_buy` | [string] | Related item ASINs (also bought) |
| `also_view` | [string] | Related item ASINs (also viewed) |
| `imageURL` | [string] | Array of standard image URLs |
| `imageURLHighRes` | [string] | Array of high-res image URLs |
| `details` | dict | Product details (SKU, dimensions, etc.) |
| `rank` | string/list | Sales rank information |
| `main_cat` | string | Main category |
| `similar_item` | string | Similar items |
| `tech1`/`tech2` | string | Technical specifications |
| `fit` | string | Fit information (clothing) |
| `date` | string | Release/publish date |

#### Shared Fields

| Field | Type 2014 | Type 2018 | Description |
|-------|-----------|-----------|-------------|
| `asin` | string | string | Item ID |
| `title` | string | string | Product title |
| `price` | float | string | Price (numeric in 2014, "$X.XX" in 2018) |
| `brand` | string | string | Brand name |
| `feature` | [string] | [string] | Product features/bullets |
| `description` | string | [string] | Description (single string in 2014, list in 2018) |

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
**Maintained by**: ActionPiece Team
