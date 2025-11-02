#!/usr/bin/env python3
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

"""Validation script to check multimodal ASIN alignment."""

import argparse
import json
import numpy as np
import os


def validate_alignment(
    cache_dir: str, image_path: str, verbose: bool = False
):
  """Validate that text and image embeddings are properly aligned by ASIN.

  Args:
      cache_dir: Path to the dataset cache directory
      image_path: Path to the image embedding .npy file
      verbose: Whether to print detailed information
  """
  print("=" * 70)
  print("MULTIMODAL ALIGNMENT VALIDATION")
  print("=" * 70)

  # 1. Load text dataset's ASIN mapping
  id_mapping_path = os.path.join(cache_dir, 'processed', 'id_mapping.json')
  if not os.path.exists(id_mapping_path):
    print(f"\n❌ ERROR: id_mapping.json not found at {id_mapping_path}")
    return

  with open(id_mapping_path, 'r') as f:
    id_mapping = json.load(f)

  text_asins = set(id_mapping['id2item'][1:])  # Skip PAD at index 0
  n_text_items = len(text_asins)

  print(f"\n1. Text Data (from reviews)")
  print(f"   Total items: {n_text_items}")
  if verbose:
    print(f"   Sample ASINs: {list(text_asins)[:5]}")

  # 2. Load image data
  if not os.path.exists(image_path):
    print(f"\n❌ ERROR: Image file not found at {image_path}")
    return

  try:
    image_data = np.load(image_path, allow_pickle=True)

    # Extract ASINs
    if isinstance(image_data, np.ndarray) and image_data.dtype == object:
      image_dict = image_data.item()
      if 'asins' in image_dict:
        image_asins = set(image_dict['asins'])
      elif 'asin' in image_dict:
        image_asins = set(image_dict['asin'])
      else:
        print("\n❌ ERROR: No ASIN info found in image data dictionary")
        return
    else:
      # Try external files
      for ext in ['.asins.txt', '.asins.json']:
        ext_path = image_path.replace('.npy', ext)
        if os.path.exists(ext_path):
          if ext == '.asins.txt':
            with open(ext_path, 'r') as f:
              image_asins = set(line.strip() for line in f if line.strip())
          else:
            with open(ext_path, 'r') as f:
              image_asins = set(json.load(f))
          break
      else:
        print("\n❌ ERROR: No ASIN info found for image data")
        return

    n_image_items = len(image_asins)

    print(f"\n2. Image Data")
    print(f"   Total items: {n_image_items}")
    if verbose:
      print(f"   Sample ASINs: {list(image_asins)[:5]}")

  except Exception as e:
    print(f"\n❌ ERROR: Failed to load image data: {e}")
    return

  # 3. Analyze overlap
  common_asins = text_asins & image_asins
  text_only = text_asins - image_asins
  image_only = image_asins - text_asins

  print(f"\n3. ASIN Overlap Analysis")
  print(f"   Common items (with both text & image): {len(common_asins)}")
  print(
      f"   Text-only items (no image): {len(text_only)} "
      f"({len(text_only)*100/n_text_items:.1f}%)"
  )
  print(
      f"   Image-only items (no text): {len(image_only)} "
      f"({len(image_only)*100/n_image_items:.1f}%)"
  )

  if verbose and len(text_only) > 0:
    print(f"\n   Sample text-only ASINs: {list(text_only)[:10]}")
  if verbose and len(image_only) > 0:
    print(f"   Sample image-only ASINs: {list(image_only)[:10]}")

  # 4. Check alignment for specific items
  print(f"\n4. Sample Alignment Verification")
  sample_ids = [1, 10, 100, 500] if n_text_items >= 500 else [1, 10, 50]

  for item_id in sample_ids:
    if item_id < len(id_mapping['id2item']):
      asin = id_mapping['id2item'][item_id]
      has_image = asin in image_asins
      status = "✓" if has_image else "✗"
      print(
          f"   Item ID {item_id:4d}: ASIN={asin:15s} | Has image: {status}"
      )

  # 5. Validate multimodal embeddings if they exist
  multimodal_paths = [
      os.path.join(
          cache_dir, 'processed', 'multimodal_final_pca_128_zero.npy'
      ),
      os.path.join(
          cache_dir, 'processed', 'multimodal_final_pca_128_mean.npy'
      ),
  ]

  for mm_path in multimodal_paths:
    if os.path.exists(mm_path):
      print(f"\n5. Multimodal Embeddings Validation")
      print(f"   Found: {mm_path}")

      mm_embs = np.load(mm_path)
      expected_shape = (n_text_items, 128)
      actual_shape = mm_embs.shape

      print(f"   Expected shape: {expected_shape}")
      print(f"   Actual shape:   {actual_shape}")

      if actual_shape == expected_shape:
        print(f"   ✓ Shape matches!")
      else:
        print(f"   ✗ Shape mismatch!")

      # Check for anomalies
      zero_rows = np.all(mm_embs == 0, axis=1).sum()
      nan_rows = np.any(np.isnan(mm_embs), axis=1).sum()
      inf_rows = np.any(np.isinf(mm_embs), axis=1).sum()

      print(f"\n   Anomaly Check:")
      print(f"     Zero vectors: {zero_rows}")
      print(f"     NaN vectors:  {nan_rows}")
      print(f"     Inf vectors:  {inf_rows}")

      if nan_rows == 0 and inf_rows == 0:
        print(f"     ✓ No anomalies detected")
      else:
        print(f"     ✗ Anomalies detected!")

      # Statistics
      print(f"\n   Statistics:")
      print(f"     Mean: {mm_embs.mean():.4f}")
      print(f"     Std:  {mm_embs.std():.4f}")
      print(f"     Min:  {mm_embs.min():.4f}")
      print(f"     Max:  {mm_embs.max():.4f}")

      break

  # 6. Summary and recommendations
  print(f"\n" + "=" * 70)
  print("SUMMARY & RECOMMENDATIONS")
  print("=" * 70)

  if len(common_asins) == n_text_items:
    print("✓ Perfect alignment: All text items have images")
  elif len(common_asins) >= n_text_items * 0.8:
    print(
        f"✓ Good alignment: {len(common_asins)*100/n_text_items:.1f}% "
        "of text items have images"
    )
    print("  Missing images will be filled with zero vectors")
  elif len(common_asins) >= n_text_items * 0.5:
    print(
        f"⚠️  Moderate alignment: {len(common_asins)*100/n_text_items:.1f}% "
        "of text items have images"
    )
    print("  Consider improving image coverage")
  else:
    print(
        f"❌ Poor alignment: Only {len(common_asins)*100/n_text_items:.1f}% "
        "of text items have images"
    )
    print("  Multimodal performance may be degraded")

  if len(image_only) > 0:
    print(
        f"\n⚠️  Note: {len(image_only)} images have no corresponding "
        "text (will be ignored)"
    )

  print("=" * 70)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Validate multimodal ASIN alignment'
  )
  parser.add_argument(
      '--cache_dir',
      type=str,
      default='cache/AmazonReviews2014/CDs_and_Vinyl',
      help='Path to dataset cache directory',
  )
  parser.add_argument(
      '--image_path',
      type=str,
      default='/scratch/zl4789/MQL4GRec/data_process/MQL4GRec/CDs/CDs.emb-ViT-L-14.npy',
      help='Path to image embedding .npy file',
  )
  parser.add_argument(
      '--verbose', action='store_true', help='Print detailed information'
  )
  args = parser.parse_args()

  validate_alignment(args.cache_dir, args.image_path, args.verbose)
