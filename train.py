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

"""Train ActionPiece model (assumes vocabulary is already built)."""

import argparse
import os
import sys

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' to enable logging.")


def parse_args():
  parser = argparse.ArgumentParser(
      description='Train ActionPiece model (vocabulary must be pre-built)'
  )
  parser.add_argument(
      '--model', type=str, default='ActionPiece', help='Model name'
  )
  parser.add_argument(
      '--dataset', type=str, default='AmazonReviews2014', help='Dataset name'
  )
  # WandB arguments
  parser.add_argument(
      '--use_wandb', action='store_true', help='Enable WandB logging'
  )
  parser.add_argument(
      '--wandb_project', type=str, default='actionpiece', help='WandB project name'
  )
  parser.add_argument(
      '--wandb_entity', type=str, default=None, help='WandB entity (team/username)'
  )
  parser.add_argument(
      '--wandb_name', type=str, default=None, help='WandB run name'
  )
  parser.add_argument(
      '--wandb_tags', type=str, default=None, help='WandB tags (comma-separated)'
  )
  parser.add_argument(
      '--wandb_notes', type=str, default=None, help='WandB run notes'
  )
  # Check vocabulary flag
  parser.add_argument(
      '--skip_vocab_check', action='store_true',
      help='Skip vocabulary existence check (not recommended)'
  )
  return parser.parse_known_args()


def clean_empty_args(unparsed_args):
  """Clean empty command line arguments."""
  cleaned_args = []
  for arg in unparsed_args:
    if '=' in arg:
      key, value = arg.split('=', 1)
      if value and value.strip() and not value.startswith('${'):
        cleaned_args.append(arg)
      else:
        print(f"Warning: Skipping empty argument: {arg}")
    else:
      cleaned_args.append(arg)
  return cleaned_args


def check_vocabulary_exists(category: str, cache_dir: str = 'cache') -> tuple[bool, str]:
  """Check if vocabulary file exists for the given category.

  Args:
      category: Dataset category (e.g., 'CDs_and_Vinyl')
      cache_dir: Cache directory path

  Returns:
      Tuple of (exists: bool, vocab_path: str)
  """
  vocab_path = os.path.join(
      cache_dir, 'AmazonReviews2014', category, 'processed/actionpiece.json'
  )
  return os.path.exists(vocab_path), vocab_path


if __name__ == '__main__':
  args, unparsed_args = parse_args()

  # Clean empty arguments
  print("Unparsed args before cleaning:", unparsed_args)
  unparsed_args = clean_empty_args(unparsed_args)
  print("Unparsed args after cleaning:", unparsed_args)

  # Parse command line configs
  command_line_configs = parse_command_line_args(unparsed_args)
  print("Parsed config:", command_line_configs)

  # Check if vocabulary exists (unless skip_vocab_check is set)
  if not args.skip_vocab_check:
    category = command_line_configs.get('category', None)
    cache_dir = command_line_configs.get('cache_dir', 'cache')

    if category is None:
      print("\n" + "=" * 70)
      print("ERROR: --category argument is required!")
      print("=" * 70)
      print("\nUsage:")
      print("  python train.py --category=CDs_and_Vinyl [other args...]")
      print("\nExample:")
      print("  python train.py --category=CDs_and_Vinyl --lr=0.001 --d_model=256")
      print("=" * 70)
      sys.exit(1)

    vocab_exists, vocab_path = check_vocabulary_exists(category, cache_dir)

    if not vocab_exists:
      print("\n" + "=" * 70)
      print("ERROR: Vocabulary not found!")
      print("=" * 70)
      print(f"Category: {category}")
      print(f"Expected vocabulary at: {vocab_path}")
      print("\nPlease build the vocabulary first using:")
      print(f"  python build_vocab.py --category={category}")
      print("\nOr run the original pipeline that builds vocabulary automatically:")
      print(f"  python main.py --category={category}")
      print("=" * 70)
      sys.exit(1)

    print(f"\n✓ Vocabulary found at: {vocab_path}")

  # Add WandB configuration
  wandb_configs = {
      'use_wandb': args.use_wandb and WANDB_AVAILABLE,
      'wandb_project': args.wandb_project,
      'wandb_entity': args.wandb_entity,
      'wandb_name': args.wandb_name,
      'wandb_notes': args.wandb_notes,
  }

  # Handle wandb_tags
  if args.wandb_tags:
      wandb_configs['wandb_tags'] = [tag.strip() for tag in args.wandb_tags.split(',')]
  else:
      wandb_configs['wandb_tags'] = []

  # Merge configurations
  command_line_configs.update(wandb_configs)

  # Check WandB availability
  if args.use_wandb and not WANDB_AVAILABLE:
      print("Warning: --use_wandb specified but wandb is not installed. Disabling WandB logging.")
      command_line_configs['use_wandb'] = False

  # Initialize WandB if needed
  if command_line_configs.get('use_wandb', False):
      try:
          api_key = wandb.api.api_key
          if not api_key:
              print("Please login to WandB first: wandb login")
              sys.exit(1)
      except Exception:
          print("Please login to WandB first: wandb login")
          sys.exit(1)

  # Create and run pipeline
  print("\n" + "=" * 70)
  print("Starting Training Pipeline")
  print("=" * 70)

  pipeline = Pipeline(
      model_name=args.model,
      dataset_name=args.dataset,
      config_dict=command_line_configs,
  )
  pipeline.run()

  print("\n" + "=" * 70)
  print("Training Complete!")
  print("=" * 70)
