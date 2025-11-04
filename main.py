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

"""Main file for ActionPiece."""

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
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', type=str, default='ActionPiece', help='Model name'
  )
  parser.add_argument(
      '--dataset', type=str, default='AmazonReviews2014', help='Dataset name'
  )
  # 添加 WandB 相关参数
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
  return parser.parse_known_args()


def clean_empty_args(unparsed_args):
  """清理空的命令行参数"""
  cleaned_args = []
  for arg in unparsed_args:
    if '=' in arg:
      key, value = arg.split('=', 1)
      # 如果值为空或者只是环境变量占位符，跳过此参数
      if value and value.strip() and not value.startswith('${'):
        cleaned_args.append(arg)
      else:
        print(f"Warning: Skipping empty argument: {arg}")
    else:
      cleaned_args.append(arg)
  return cleaned_args


if __name__ == '__main__':
  args, unparsed_args = parse_args()
  
  # 清理空的命令行参数
  print("Unparsed args before cleaning:", unparsed_args)
  unparsed_args = clean_empty_args(unparsed_args)
  print("Unparsed args after cleaning:", unparsed_args)
  
  command_line_configs = parse_command_line_args(unparsed_args)
  print("Parsed config:", command_line_configs)
  
  # 添加 WandB 配置到 command_line_configs
  wandb_configs = {
      'use_wandb': args.use_wandb and WANDB_AVAILABLE,
      'wandb_project': args.wandb_project,
      'wandb_entity': args.wandb_entity,
      'wandb_name': args.wandb_name,
      'wandb_notes': args.wandb_notes,
  }
  
  # 处理 wandb_tags
  if args.wandb_tags:
      wandb_configs['wandb_tags'] = [tag.strip() for tag in args.wandb_tags.split(',')]
  else:
      wandb_configs['wandb_tags'] = []
  
  # 合并配置
  command_line_configs.update(wandb_configs)
  
  # 如果启用 WandB 但没有安装，给出警告
  if args.use_wandb and not WANDB_AVAILABLE:
      print("Warning: --use_wandb specified but wandb is not installed. Disabling WandB logging.")
      command_line_configs['use_wandb'] = False
  
  # 初始化 WandB（如果需要）
  if command_line_configs.get('use_wandb', False):
      # 检查是否已经登录
      try:
          # 尝试获取 API key 来检查登录状态
          api_key = wandb.api.api_key
          if not api_key:
              print("Please login to WandB first: wandb login")
              sys.exit(1)
      except Exception:
          print("Please login to WandB first: wandb login")
          sys.exit(1)

  pipeline = Pipeline(
      model_name=args.model,
      dataset_name=args.dataset,
      config_dict=command_line_configs,
  )
  pipeline.run()