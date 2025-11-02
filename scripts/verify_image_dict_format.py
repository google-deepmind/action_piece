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

"""Verify that image data is in the correct dictionary format."""

import argparse
import numpy as np


def verify_image_dict_format(image_path: str) -> bool:
  """Verify image data is in dictionary format with 'asins' and 'embeddings'.

  Args:
      image_path: Path to the .npy image file

  Returns:
      True if format is correct, False otherwise
  """
  print("=" * 70)
  print("IMAGE DICTIONARY FORMAT VERIFICATION")
  print("=" * 70)
  print(f"\nFile: {image_path}")

  try:
    # Load data
    image_data = np.load(image_path, allow_pickle=True)
    print("✓ File loaded successfully")

    # Check if it's an object array (dictionary)
    if not (isinstance(image_data, np.ndarray) and image_data.dtype == object):
      print(f"\n❌ ERROR: Not in dictionary format")
      print(f"   Expected: numpy array with dtype=object")
      print(f"   Got: {type(image_data)} with dtype={image_data.dtype}")
      return False

    # Extract dictionary
    image_dict = image_data.item()
    print(f"✓ Dictionary extracted")

    # Check keys
    required_keys = {'asins', 'embeddings'}
    found_keys = set(image_dict.keys())

    print(f"\nKeys found: {list(found_keys)}")

    if not required_keys.issubset(found_keys):
      missing = required_keys - found_keys
      print(f"\n❌ ERROR: Missing required keys: {missing}")
      return False

    print(f"✓ Required keys present: {list(required_keys)}")

    # Check asins
    asins = image_dict['asins']
    print(f"\n'asins' field:")
    print(f"  Type: {type(asins)}")
    print(f"  Length: {len(asins)}")
    print(f"  Sample (first 5): {asins[:5]}")

    # Check embeddings
    embeddings = image_dict['embeddings']
    print(f"\n'embeddings' field:")
    print(f"  Type: {type(embeddings)}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")

    # Check consistency
    if len(asins) != embeddings.shape[0]:
      print(f"\n❌ ERROR: Count mismatch!")
      print(f"   ASINs: {len(asins)}")
      print(f"   Embeddings: {embeddings.shape[0]}")
      return False

    print(f"\n✓ Counts match: {len(asins)} items")

    # Check for duplicates
    unique_asins = set(asins)
    if len(unique_asins) < len(asins):
      duplicates = len(asins) - len(unique_asins)
      print(f"\n⚠️  WARNING: {duplicates} duplicate ASINs found")
    else:
      print(f"✓ All ASINs are unique")

    # Statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std:  {embeddings.std():.4f}")
    print(f"  Min:  {embeddings.min():.4f}")
    print(f"  Max:  {embeddings.max():.4f}")

    print("\n" + "=" * 70)
    print("✅ FORMAT VERIFICATION PASSED")
    print("=" * 70)
    print("\nYour image data is correctly formatted and ready to use!")
    return True

  except Exception as e:
    print(f"\n❌ ERROR: {e}")
    return False


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Verify image data dictionary format'
  )
  parser.add_argument(
      '--image_path',
      type=str,
      required=True,
      help='Path to the .npy image file',
  )
  args = parser.parse_args()

  success = verify_image_dict_format(args.image_path)
  exit(0 if success else 1)
