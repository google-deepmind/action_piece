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

"""Diagnostic script to inspect image embedding data format."""

import argparse
import numpy as np
import os


def diagnose_image_data(image_path: str):
  """Diagnose the format and content of image embedding data.

  Args:
      image_path: Path to the .npy image embedding file
  """
  print("=" * 70)
  print("IMAGE DATA DIAGNOSTIC REPORT")
  print("=" * 70)
  print(f"\nFile: {image_path}")
  print(f"Exists: {os.path.exists(image_path)}")

  if not os.path.exists(image_path):
    print("\n❌ ERROR: File not found!")
    return

  print(f"File size: {os.path.getsize(image_path) / 1024 / 1024:.2f} MB")

  # Try loading the data
  try:
    image_data = np.load(image_path, allow_pickle=True)
    print(f"\n✓ Successfully loaded data")
  except Exception as e:
    print(f"\n❌ ERROR: Failed to load data: {e}")
    return

  # Check data type
  print(f"\nData type: {type(image_data)}")
  print(f"Data dtype: {image_data.dtype}")

  # Check if it's a dictionary
  if isinstance(image_data, np.ndarray) and image_data.dtype == object:
    print("\n📦 Dictionary format detected")
    try:
      data_dict = image_data.item()
      print(f"Keys: {list(data_dict.keys())}")

      for key in data_dict.keys():
        value = data_dict[key]
        print(f"\n  Key: '{key}'")
        print(f"    Type: {type(value)}")
        if hasattr(value, 'shape'):
          print(f"    Shape: {value.shape}")
        elif isinstance(value, list):
          print(f"    Length: {len(value)}")
          print(f"    First 3 items: {value[:3]}")

      # Check if ASIN info exists
      if 'asins' in data_dict:
        print(f"\n✓ ASIN information found in 'asins' key")
        print(f"  Total ASINs: {len(data_dict['asins'])}")
        print(f"  Sample ASINs: {data_dict['asins'][:5]}")
      elif 'asin' in data_dict:
        print(f"\n✓ ASIN information found in 'asin' key")
        print(f"  Total ASINs: {len(data_dict['asin'])}")
        print(f"  Sample ASINs: {data_dict['asin'][:5]}")
      else:
        print(f"\n⚠️  WARNING: No ASIN information found in dictionary")
        print(f"  Available keys: {list(data_dict.keys())}")

      # Check embeddings
      if 'embeddings' in data_dict:
        print(f"\n✓ Embeddings found")
        embs = data_dict['embeddings']
        print(f"  Shape: {embs.shape}")
        print(f"  Dtype: {embs.dtype}")
        print(f"  Mean: {embs.mean():.4f}")
        print(f"  Std: {embs.std():.4f}")

    except Exception as e:
      print(f"\n❌ ERROR: Failed to parse dictionary: {e}")

  # Check if it's a pure array
  elif isinstance(image_data, np.ndarray):
    print("\n📊 Pure array format detected")
    print(f"Shape: {image_data.shape}")
    print(f"Dtype: {image_data.dtype}")

    if len(image_data.shape) == 2:
      print(f"\n✓ 2D array (likely embeddings)")
      print(f"  Number of items: {image_data.shape[0]}")
      print(f"  Embedding dimension: {image_data.shape[1]}")
      print(f"  Mean: {image_data.mean():.4f}")
      print(f"  Std: {image_data.std():.4f}")
    else:
      print(f"\n⚠️  WARNING: Unexpected shape {image_data.shape}")

    # Check for external ASIN files
    print("\n🔍 Checking for external ASIN files...")
    base_path = image_path.replace('.npy', '')

    asin_files = {
        '.asins.txt': f'{base_path}.asins.txt',
        '.asins.json': f'{base_path}.asins.json',
        '.config.json': f'{base_path}.config.json',
    }

    found_external = False
    for name, path in asin_files.items():
      if os.path.exists(path):
        print(f"  ✓ Found: {path}")
        found_external = True
      else:
        print(f"  ✗ Not found: {name}")

    if not found_external:
      print(
          f"\n⚠️  WARNING: No external ASIN files found. "
          "You need to create one!"
      )

  else:
    print(f"\n⚠️  WARNING: Unexpected data type: {type(image_data)}")

  print("\n" + "=" * 70)
  print("RECOMMENDATION:")
  print("=" * 70)

  # Check if we can align
  if isinstance(image_data, np.ndarray) and image_data.dtype == object:
    data_dict = image_data.item()
    if 'asins' in data_dict or 'asin' in data_dict:
      print("✓ Your image data contains ASIN information.")
      print("  The modified code can use it directly for alignment.")
    else:
      print("⚠️  Your image data is in dictionary format but lacks ASIN info.")
      print("  Please add 'asins' key to the dictionary.")
  else:
    print("⚠️  Your image data is in pure array format.")
    print("  You need to provide ASIN information in one of these ways:")
    print("  1. Convert to dictionary format with 'asins' and 'embeddings' keys")
    print("  2. Create an external .asins.txt or .asins.json file")

  print("=" * 70)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Diagnose image embedding data format'
  )
  parser.add_argument(
      '--image_path',
      type=str,
      required=True,
      help='Path to the .npy image embedding file',
  )
  args = parser.parse_args()

  diagnose_image_data(args.image_path)
