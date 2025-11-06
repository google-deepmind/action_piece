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

"""Build ActionPiece vocabulary only (no training)."""

import argparse
import logging
import os
import sys

from genrec import utils
from genrec.utils import parse_command_line_args


def parse_args():
  parser = argparse.ArgumentParser(
      description='Build ActionPiece vocabulary (no training)'
  )
  parser.add_argument(
      '--model', type=str, default='ActionPiece', help='Model name'
  )
  parser.add_argument(
      '--dataset', type=str, default='AmazonReviews2014', help='Dataset name'
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


def build_vocabulary(model_name: str, dataset_name: str, config_dict: dict):
  """Build ActionPiece vocabulary without training.

  Args:
      model_name: Name of the model (default: 'ActionPiece')
      dataset_name: Name of the dataset (default: 'AmazonReviews2014')
      config_dict: Configuration dictionary from command line

  Returns:
      Path to the saved vocabulary file
  """
  # Get configuration
  config = utils.get_config(
      model_name=model_name,
      dataset_name=dataset_name,
      config_file=None,
      config_dict=config_dict,
  )

  # Initialize seed and logger
  utils.init_seed(config['rand_seed'], config['reproducibility'])
  utils.init_logger(config)
  logger = logging.getLogger()

  logger.info('=' * 60)
  logger.info('Building ActionPiece Vocabulary')
  logger.info('=' * 60)

  # Load dataset
  logger.info(f'Loading dataset: {dataset_name}')
  dataset_class = utils.get_dataset(dataset_name)
  raw_dataset = dataset_class(config)
  logger.info(str(raw_dataset))

  # Split dataset
  logger.info('Splitting dataset...')
  split_datasets = raw_dataset.split()

  # Initialize tokenizer (this will build or load the vocabulary)
  logger.info(f'Initializing tokenizer for model: {model_name}')
  tokenizer_class = utils.get_tokenizer(model_name)
  tokenizer = tokenizer_class(config, raw_dataset)

  # Get vocabulary path
  vocab_path = os.path.join(
      raw_dataset.cache_dir, 'processed/actionpiece.json'
  )

  # Report results
  logger.info('=' * 60)
  logger.info('Vocabulary Construction Complete!')
  logger.info('=' * 60)
  logger.info(f'Vocabulary saved to: {vocab_path}')
  logger.info(f'Vocabulary size: {tokenizer.actionpiece.vocab_size}')
  logger.info(f'Number of categories: {tokenizer.actionpiece.n_categories}')
  logger.info(f'Number of initial features: {tokenizer.actionpiece.n_init_feats}')
  logger.info('=' * 60)

  return vocab_path


if __name__ == '__main__':
  args, unparsed_args = parse_args()

  # Clean empty arguments
  print("Unparsed args before cleaning:", unparsed_args)
  unparsed_args = clean_empty_args(unparsed_args)
  print("Unparsed args after cleaning:", unparsed_args)

  # Parse command line configs
  command_line_configs = parse_command_line_args(unparsed_args)
  print("Parsed config:", command_line_configs)

  # Build vocabulary
  vocab_path = build_vocabulary(
      model_name=args.model,
      dataset_name=args.dataset,
      config_dict=command_line_configs,
  )

  print(f'\n✓ Vocabulary built successfully!')
  print(f'✓ Saved to: {vocab_path}')
  print(f'\nYou can now train the model using:')
  print(f'  python train.py --category=<your_category> [other args...]')
