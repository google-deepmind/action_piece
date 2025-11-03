# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Dataset for Amazon Reviews 2023."""

import collections
import csv
import datetime as dt
import gzip
import json
import os
from typing import Any, Optional, Sequence

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text
from genrec.utils import download_file
import numpy as np
import tqdm


def parse_gz(path: str):
  """Parse a gzipped file and yield each line as a dict."""
  import ast

  with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as g_file:
    for line_num, line in enumerate(g_file, 1):
      line = line.strip()
      if not line:
        continue
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        try:
          yield ast.literal_eval(line)
        except (ValueError, SyntaxError) as exc:
          if line_num <= 10:
            print(f'Warning: Failed to parse line {line_num} in {path}: {exc}')
            preview = line[:100] if len(line) > 100 else line
            print(f'Line content: {preview}')
          continue


def get_item_seqs(
    reviews: Sequence[tuple[str, str, int]]
) -> dict[str, list[str]]:
  """Group the reviews by user and sort the items by time."""
  item_seqs = collections.defaultdict(list)
  for user, item, timestamp in reviews:
    item_seqs[user].append((item, timestamp))

  for user, item_time in item_seqs.items():
    item_time.sort(key=lambda x: x[1])
    item_seqs[user] = [item for item, _ in item_time]
  return item_seqs


def check_available_category(category: str):
  """Checks if the `self.category` is available in the dataset."""
  available_categories = [
      'All_Amazon',
      'All_Beauty',
      'Appliances',
      'Apps_and_Games',
      'Arts_Crafts_and_Sewing',
      'Automotive',
      'Baby_Products',
      'Beauty_and_Personal_Care',
      'Books',
      'CDs_and_Vinyl',
      'Cell_Phones_and_Accessories',
      'Clothing_Shoes_and_Jewelry',
      'Computers',
      'Digital_Music',
      'Electronics',
      'Gift_Cards',
      'Grocery_and_Gourmet_Food',
      'Handmade_Products',
      'Health_and_Household',
      'Home_and_Kitchen',
      'Industrial_and_Scientific',
      'Jewelry',
      'Kindle_Store',
      'Kitchen_and_Dining',
      'Luxury_Beauty',
      'Magazine_Subscriptions',
      'Movies_and_TV',
      'Musical_Instruments',
      'Office_Products',
      'Patio_Lawn_and_Garden',
      'Pet_Supplies',
      'Prime_Pantry',
      'Software',
      'Sports_and_Outdoors',
      'Tools_and_Home_Improvement',
      'Toys_and_Games',
      'Video_Games',
  ]
  assert category in available_categories, (
      f'Category "{category}" not available. '
      f'Available categories: {available_categories}'
  )


class AmazonReviews2023(AbstractDataset):
  """A class representing the Amazon Reviews 2023 dataset."""

  def __init__(self, config: dict[str, Any]):
    super().__init__(config)

    self.category = config['category']
    check_available_category(self.category)
    self.log(f'[DATASET] Amazon Reviews 2023 for category: {self.category}')

    self.cache_dir = os.path.join(
        config['cache_dir'], 'AmazonReviews2023', self.category
    )
    self._download_and_process_raw()

  def _parse_timestamp(self, record: dict[str, Any]) -> int:
    """Resolve the timestamp for a review record."""
    numeric_candidates = [
        record.get('unixReviewTime'),
        record.get('unix_review_time'),
        record.get('timestamp'),
        record.get('unix_time'),
    ]
    for value in numeric_candidates:
      if value is None:
        continue
      try:
        return int(value)
      except (TypeError, ValueError):
        continue

    date_candidates = [
        record.get('review_date'),
        record.get('reviewTime'),
        record.get('date'),
    ]
    for raw_value in date_candidates:
      if not raw_value:
        continue
      timestamp = self._parse_date_string(raw_value)
      if timestamp is not None:
        return timestamp
    return 0

  def _parse_date_string(self, raw_value: Any) -> Optional[int]:
    if isinstance(raw_value, (int, float)):
      return int(raw_value)
    if not isinstance(raw_value, str):
      return None

    raw_value = raw_value.strip()
    patterns = [
        '%Y-%m-%d',
        '%m %d, %Y',
        '%d %b %Y',
        '%b %d, %Y',
    ]
    for pattern in patterns:
      try:
        dt_object = dt.datetime.strptime(raw_value, pattern)
        return int(dt_object.timestamp())
      except ValueError:
        continue
    return None

  def _download_raw(self, path: str, file_type: str = 'reviews') -> str:
    """Downloads review or metadata files from the Amazon Reviews 2023 release."""
    if file_type == 'reviews':
      candidates = [
          (
              f'{self.category}.csv.gz',
              f'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/{self.category}.csv.gz',
          ),
          (
              f'{self.category}.jsonl.gz',
              f'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{self.category}.jsonl.gz',
          ),
      ]
    elif file_type == 'meta':
      candidates = [
          (
              f'meta_{self.category}.jsonl.gz',
              f'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{self.category}.jsonl.gz',
          ),
      ]
    else:
      raise ValueError(f'Unsupported file_type "{file_type}" for download.')

    for file_name, url in candidates:
      local_filepath = os.path.join(path, file_name)
      if os.path.exists(local_filepath):
        return local_filepath
      download_file(url, local_filepath)
      if os.path.exists(local_filepath):
        return local_filepath

    raise FileNotFoundError(
        f'Unable to download "{file_type}" data for category '
        f'{self.category}. Tried URLs: {[u for _, u in candidates]}'
    )

  def _load_reviews(self, path: str) -> list[tuple[str, str, int]]:
    """Load reviews from the raw gzipped JSON file."""
    self.log('[DATASET] Loading reviews...')
    reviews = []
    if path.endswith('.csv.gz'):
      with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
          user = (
              row.get('reviewerID')
              or row.get('reviewer_id')
              or row.get('customer_id')
              or row.get('user_id')
          )
          item = row.get('asin') or row.get('product_id') or row.get('item_id')
          if not user or not item:
            continue
          timestamp = self._parse_timestamp(row)
          reviews.append((str(user), str(item), int(timestamp)))
    else:
      for record in parse_gz(path):
        user = record.get('reviewerID') or record.get('reviewer_id')
        item = record.get('asin')
        if not user or not item:
          continue
        timestamp = self._parse_timestamp(record)
        reviews.append((str(user), str(item), int(timestamp)))
    return reviews

  def _remap_ids(
      self, item_seqs: dict[str, list[str]]
  ) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Remaps user and item IDs into contiguous integer ranges."""
    self.log('[DATASET] Remapping user and item IDs...')
    for user, items in item_seqs.items():
      if user not in self.id_mapping['user2id']:
        self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
        self.id_mapping['id2user'].append(user)
      iid_sequence = []
      for item in items:
        if item not in self.id_mapping['item2id']:
          self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
          self.id_mapping['id2item'].append(item)
        iid_sequence.append(item)
      self.all_item_seqs[user] = iid_sequence
    return self.all_item_seqs, self.id_mapping

  def _process_reviews(
      self, input_path: str, output_path: str
  ) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Process raw reviews and persist remapped sequences."""
    seq_file = os.path.join(output_path, 'all_item_seqs.json')
    id_mapping_file = os.path.join(output_path, 'id_mapping.json')

    if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
      self.log('[DATASET] Reviews have been processed...')
      with open(seq_file, 'r') as seq_handle:
        all_item_seqs = json.load(seq_handle)
      with open(id_mapping_file, 'r') as mapping_handle:
        id_mapping = json.load(mapping_handle)
      return all_item_seqs, id_mapping

    self.log('[DATASET] Processing reviews...')
    reviews = self._load_reviews(input_path)
    item_seqs = get_item_seqs(reviews)
    all_item_seqs, id_mapping = self._remap_ids(item_seqs)

    self.log('[DATASET] Saving mapping data...')
    with open(seq_file, 'w') as seq_handle:
      json.dump(all_item_seqs, seq_handle)
    with open(id_mapping_file, 'w') as mapping_handle:
      json.dump(id_mapping, mapping_handle)
    return all_item_seqs, id_mapping

  def _load_metadata(
      self, path: str, item2id: dict[str, int]
  ) -> dict[str, Any]:
    """Load metadata and filter by ASINs present in the review set."""
    self.log('[DATASET] Loading metadata...')
    filtered = {}
    item_asins = set(item2id.keys())
    for info in tqdm.tqdm(parse_gz(path)):
      asin = info.get('asin')
      if asin not in item_asins:
        continue
      filtered[str(asin)] = info
    return filtered

  def _sent_process(self, raw: Any) -> str:
    """Convert structured metadata fields into a flat sentence."""
    sentence = ''
    if isinstance(raw, float):
      sentence += f'{raw}.'
    elif isinstance(raw, (int, np.integer)):
      sentence += f'{raw}.'
    elif raw and isinstance(raw, list) and raw and isinstance(raw[0], list):
      for group in raw:
        for value in group:
          sentence += clean_text(value)[:-1]
          sentence += ', '
      sentence = sentence[:-2] if sentence.endswith(', ') else sentence
      sentence += '.'
    elif isinstance(raw, list):
      for value in raw:
        sentence += clean_text(value)
    elif isinstance(raw, str):
      sentence = clean_text(raw)
    else:
      sentence = str(raw)
    return sentence + ' '

  def _extract_meta_sentences(self, metadata: dict[str, Any]) -> dict[str, str]:
    """Extract textual representations from metadata."""
    self.log('[DATASET] Extracting meta sentences...')
    item2meta = {}
    features_needed = [
        'title',
        'product_title',
        'subtitle',
        'brand',
        'manufacturer',
        'feature',
        'bullet_points',
        'categories',
        'keywords',
        'description',
        'product_description',
        'product_overview',
        'editorial_reviews',
        'ingredients',
    ]

    for item, meta in tqdm.tqdm(metadata.items()):
      meta_sentence = ''
      keys = set(meta.keys())
      for feature in features_needed:
        if feature in keys:
          meta_sentence += self._sent_process(meta[feature])
      item2meta[item] = meta_sentence
    return item2meta

  def _process_meta(
      self, input_path: str, output_path: str
  ) -> Optional[dict[str, Any]]:
    """Process metadata into the desired representation."""
    process_mode = self.config['metadata']
    meta_file = os.path.join(output_path, f'metadata.{process_mode}.json')

    if os.path.exists(meta_file):
      self.log('[DATASET] Metadata has been processed...')
      with open(meta_file, 'r') as meta_handle:
        return json.load(meta_handle)

    self.log(f'[DATASET] Processing metadata, mode: {process_mode}')

    if process_mode == 'none':
      return None

    item2raw_meta = self._load_metadata(path=input_path, item2id=self.item2id)
    if process_mode == 'raw':
      item2meta = item2raw_meta
    elif process_mode == 'sentence':
      item2meta = self._extract_meta_sentences(metadata=item2raw_meta)
    else:
      raise NotImplementedError('Metadata processing type not implemented.')

    with open(meta_file, 'w') as meta_handle:
      json.dump(item2meta, meta_handle)
    return item2meta

  def _download_and_process_raw(self) -> None:
    """Download (if needed) and preprocess reviews plus metadata."""
    raw_data_path = os.path.join(self.cache_dir, 'raw')
    os.makedirs(raw_data_path, exist_ok=True)

    with self.accelerator.main_process_first():
      reviews_localpath = self._download_raw(
          path=raw_data_path, file_type='reviews'
      )
      meta_localpath = self._download_raw(path=raw_data_path, file_type='meta')

    np.random.seed(12345)

    processed_data_path = os.path.join(self.cache_dir, 'processed')
    os.makedirs(processed_data_path, exist_ok=True)

    self.all_item_seqs, self.id_mapping = self._process_reviews(
        input_path=reviews_localpath, output_path=processed_data_path
    )

    self.item2meta = self._process_meta(
        input_path=meta_localpath, output_path=processed_data_path
    )
