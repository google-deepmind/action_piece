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

"""Utils for GenRec."""

import datetime
import hashlib
import html
import importlib
import logging
import os
import random
import re
import sys
from typing import Any, Optional, Union

import accelerate.utils
import datasets.utils.logging
# 移除这行导入，改为在函数内部延迟导入
# from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
# from genrec.trainer import Trainer  # 改为延迟导入
import numpy as np
import requests
import torch
import urllib3
import yaml

# Disable SSL warnings for academic dataset downloads
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def init_seed(seed, reproducibility):
  r"""Init random seed for random functions in numpy, torch, cuda and cudnn.

  Args:
      seed (int): random seed
      reproducibility (bool): Whether to require reproducibility
  """

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  accelerate.utils.set_seed(seed)
  if reproducibility:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
  else:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_local_time():
  """Get current time.

  Returns:
      str: current time
  """
  cur = datetime.datetime.now()
  cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
  return cur


def get_command_line_args_str():
  r"""Get command line arguments as a string.

  Returns:
      str: Command line arguments as a string.
  """
  filtered_args = []
  for arg in sys.argv:
    filter_flag = False
    for flag in [
        '--model',
        '--dataset',
        '--category',
        '--my_log_dir',
        '--tensorboard_log_dir',
        '--ckpt_dir',
    ]:
      if arg.startswith(flag):
        filter_flag = True
        break
    if arg.startswith('--cache_dir'):
      filtered_args.append(f'--cache_dir={os.path.basename(arg.split("=")[1])}')
    elif not filter_flag:
      filtered_args.append(arg)
  return '_'.join(filtered_args).replace('/', '|')


def get_file_name(config: dict[str, Any], suffix: str = '') -> str:
  """Generates a unique file name based on the given configuration and suffix.

  Args:
      config (dict): The configuration dictionary.
      suffix (str): The suffix to append to the file name.

  Returns:
      str: The unique file name.
  """
  config_str = ''.join(
      str(value) for key, value in config.items() if key != 'accelerator'
  )
  md5 = hashlib.md5(config_str.encode()).hexdigest()[:6]
  command_line_args = get_command_line_args_str()
  logfilename = f'{config["run_id"]}-{command_line_args}-{config["run_local_time"]}-{md5}-{suffix}'
  return logfilename


def init_logger(config: dict[str, Any]):
  """Initializes the logger for the given configuration."""

  log_root = config['log_dir']
  os.makedirs(log_root, exist_ok=True)
  dataset_name = os.path.join(log_root, config['dataset'])
  os.makedirs(dataset_name, exist_ok=True)
  model_name = os.path.join(dataset_name, config['model'])
  os.makedirs(model_name, exist_ok=True)

  logfilename = get_file_name(config, suffix='.log')
  logfilepath = os.path.join(
      log_root, config['dataset'], config['model'], logfilename
  )

  filefmt = '%(asctime)-15s %(levelname)s  %(message)s'
  filedatefmt = '%a %d %b %Y %H:%M:%S'
  fileformatter = logging.Formatter(filefmt, filedatefmt)

  fh = logging.FileHandler(logfilepath)
  fh.setLevel(logging.INFO)
  fh.setFormatter(fileformatter)

  sh = logging.StreamHandler()
  sh.setLevel(logging.INFO)

  logging.basicConfig(level=logging.INFO, handlers=[sh, fh])

  if not config['accelerator'].is_main_process:
    datasets.utils.logging.disable_progress_bar()


def log(message, accelerator, logger, level='info'):
  """Logs a message to the logger.

  Args:
      message (str): The message to log.
      accelerator (Accelerator): The accelerator object.
      logger (logging.Logger): The logger object.
      level (str): The log level ('info', 'error', 'warning', 'debug').
  """
  if accelerator.is_main_process:
    try:
      # 兼容 Python 3.10 和更早版本
      level_mapping = {
          'DEBUG': logging.DEBUG,
          'INFO': logging.INFO,
          'WARNING': logging.WARNING,
          'ERROR': logging.ERROR,
          'CRITICAL': logging.CRITICAL
      }
      level_num = level_mapping.get(level.upper())
      if level_num is None:
        raise ValueError(f'Invalid log level: {level}')
    except KeyError as exc:
      raise ValueError(f'Invalid log level: {level}') from exc

    logger.log(level_num, message)


def get_tokenizer(model_name: str):
  """Retrieves the tokenizer for a given model name.

  Args:
      model_name (str): The model name.

  Returns:
      AbstractTokenizer: The tokenizer for the given model name.

  Raises:
      ValueError: If the tokenizer is not found.
  """

  module_name = f'genrec.models.{model_name}.tokenizer'
  try:
    module = importlib.import_module(module_name)
    tokenizer_class = getattr(module, f'{model_name}Tokenizer')
    return tokenizer_class
  except Exception as exc:
    raise ValueError(f'Tokenizer for model "{model_name}" not found.') from exc

def get_model(model_name: Union[str, Any]) -> Any:
  """Retrieves the model class based on the provided model name.

  Args:
      model_name (Union[str, Any]): The name of the model or an
        instance of the model class.

  Returns:
      Any: The model class corresponding to the provided model name.

  Raises:
      ValueError: If the model name is not found.
  """
  if hasattr(model_name, '__class__') and hasattr(model_name.__class__, '__name__'):
    # 检查是否是模型实例，但避免直接类型检查
    if 'AbstractModel' in str(type(model_name)):
      return model_name

  try:
    model_class = getattr(importlib.import_module('genrec.models'), model_name)
  except Exception as exc:
    raise ValueError(f'Model "{model_name}" not found.') from exc
  return model_class


def get_dataset(dataset_name: Union[str, Any]) -> Any:
  """Get the dataset object based on the dataset name or directly return the dataset object if it is already provided.

  Args:
      dataset_name (Union[str, AbstractDataset]): The name of the dataset or the
        dataset object itself.

  Returns:
      AbstractDataset: The dataset object.

  Raises:
      ValueError: If the dataset name is not found.
  """
  # 延迟导入以避免循环导入
  from genrec.dataset import AbstractDataset
  
  if isinstance(dataset_name, AbstractDataset):
    return dataset_name

  # 首先尝试从 genrec.datasets 模块中获取
  try:
    dataset_module = importlib.import_module('genrec.datasets')
    dataset_class = getattr(dataset_module, dataset_name)
    return dataset_class
  except (ImportError, AttributeError):
    # 如果上面失败，尝试从具体的子模块中获取
    try:
      if dataset_name == 'AmazonReviews2014':
        from genrec.datasets.AmazonReviews2014.dataset import AmazonReviews2014
        return AmazonReviews2014
      if dataset_name == 'AmazonReviews2018':
        from genrec.datasets.AmazonReviews2018.dataset import AmazonReviews2018
        return AmazonReviews2018

      else:
        # 可以在这里添加其他数据集的映射
        raise ValueError(f'Dataset "{dataset_name}" not found.')
    except ImportError as exc:
      raise ValueError(f'Dataset "{dataset_name}" not found.') from exc


def get_trainer(model_name: Union[str, Any]):
  """Returns the trainer class based on the given model name.

  Args:
      model_name (Union[str, Any]): The name of the model or an
        instance of the AbstractModel class.

  Returns:
      trainer_class: The trainer class corresponding to the given model name. If
      the model name is not found, the default Trainer class is returned.
  """
  # 延迟导入
  from genrec.trainer import Trainer
  
  if isinstance(model_name, str):
    try:
      trainer_module = importlib.import_module(f'genrec.models.{model_name}.trainer')
      trainer_class = getattr(trainer_module, f'{model_name}Trainer')
      return trainer_class
    except (ImportError, AttributeError):
      # 如果找不到专用的训练器，使用默认的训练器
      return Trainer

  return Trainer


def get_total_steps(config, train_dataloader):
  """Calculate the total number of steps for training based on the given configuration and dataloader.

  Args:
      config (dict): The configuration dictionary containing the training
        parameters.
      train_dataloader (DataLoader): The dataloader for the training dataset.

  Returns:
      int: The total number of steps for training.
  """
  if config['steps'] is not None:
    return config['steps']
  else:
    return len(train_dataloader) * config['epochs']


def _convert_value(value: str) -> Any:
  """Convert a string value to its appropriate type.

  Args:
      value (str): The string value to convert.

  Returns:
      Any: The converted value.
  """
  if value.lower() == 'true':
    return True
  if value.lower() == 'false':
    return False
  try:
    return int(value)
  except ValueError:
    pass
  try:
    return float(value)
  except ValueError:
    pass
  # 修复：只有当字符串明确包含 [ 和 ] 时才尝试解析为列表
  if value.strip().startswith('[') and value.strip().endswith(']'):
    try:
      # 更安全的列表解析
      inner = value.strip()[1:-1].strip()
      if not inner:  # 空列表
        return []
      # 按逗号分割并清理
      items = [item.strip().strip('"\'') for item in inner.split(',')]
      # 尝试转换每个元素
      converted_items = []
      for item in items:
        try:
          # 尝试转换为数字
          if '.' in item:
            converted_items.append(float(item))
          else:
            converted_items.append(int(item))
        except ValueError:
          # 保持为字符串
          converted_items.append(item)
      return converted_items
    except (ValueError, TypeError):
      pass
  return value


def deep_update(base_dict: dict[Any, Any], update_dict: dict[Any, Any]) -> dict[Any, Any]:
  """Recursively update a nested dictionary.

  Args:
      base_dict (dict): The base dictionary to update.
      update_dict (dict): The dictionary containing updates.

  Returns:
      dict: The updated dictionary.
  """
  for key, value in update_dict.items():
    if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
      # Recursively merge nested dictionaries
      deep_update(base_dict[key], value)
    else:
      # Overwrite or add the value
      base_dict[key] = value
  return base_dict


def convert_config_dict(config: dict[Any, Any]) -> dict[Any, Any]:
  """Convert the values in a dictionary to their appropriate types.

  Recursively processes nested dictionaries.

  Args:
      config (dict): The dictionary containing the configuration values.

  Returns:
      dict: The dictionary with the converted values.
  """
  logger = logging.getLogger()
  for key, v in config.items():
    if isinstance(v, dict):
      # Recursively convert nested dictionaries
      config[key] = convert_config_dict(v)
    elif isinstance(v, str):
      try:
        config[key] = _convert_value(v)
      except (ValueError, TypeError):
        logger.warning('Could not convert value "%s" for key "%s".', v, key)
  return config


def get_config(
    model_name: Union[str, Any],
    dataset_name: Union[str, Any],
    config_file: Union[str, list[str], None],
    config_dict: Optional[dict[str, Any]],
) -> dict[str, Any]:
  """Get the configuration for a model and dataset.

  Overwrite rule: config_dict > config_file > model config.yaml > dataset
  config.yaml > default.yaml

  Args:
      model_name (Union[str, Any]): The name of the model or an
        instance of the model class.
      dataset_name (Union[str, Any]): The name of the dataset or an
        instance of the dataset class.
      config_file (Union[str, list[str], None]): The path to additional
        configuration file(s) or a list of paths to multiple additional
        configuration files. If None, default configurations will be used.
      config_dict (Optional[dict[str, Any]]): A dictionary containing additional
        configuration options. These options will override the ones loaded from
        the configuration file(s).

  Returns:
      dict: The final configuration dictionary.

  Raises:
      FileNotFoundError: If any of the specified configuration files cannot be
      found.

  Note:
      - If `model_name` is a string, the function will attempt to load the
      model's configuration file located at
      `genrec/models/{model_name}/config.yaml`.
      - If `dataset_name` is a string, the function will attempt to load the
      dataset's configuration file located at
      `genrec/datasets/{dataset_name}/config.yaml`.
      - The function will merge the configurations from all the specified
      configuration files and the `config_dict` parameter.
  """
  final_config = {}
  logger = logging.getLogger()

  # Load default configs
  current_path = os.path.dirname(os.path.realpath(__file__))
  config_file_list = [os.path.join(current_path, 'default.yaml')]

  if isinstance(dataset_name, str):
    config_file_list.append(
        os.path.join(current_path, f'datasets/{dataset_name}/config.yaml')
    )
    final_config['dataset'] = dataset_name
  else:
    logger.info(
        'Custom dataset, '
        'whose config should be manually loaded and passed '
        'via "config_file" or "config_dict".'
    )
    final_config['dataset'] = dataset_name.__class__.__name__

  if isinstance(model_name, str):
    config_file_list.append(
        os.path.join(current_path, f'models/{model_name}/config.yaml')
    )
    final_config['model'] = model_name
  else:
    logger.info(
        'Custom model, '
        'whose config should be manually loaded and passed '
        'via "config_file" or "config_dict".'
    )
    final_config['model'] = model_name.__class__.__name__

  if config_file:
    if isinstance(config_file, str):
      config_file = [config_file]
    config_file_list.extend(config_file)

  for file in config_file_list:
    cur_config = yaml.safe_load(open(file, 'r'))
    if cur_config is not None:
      deep_update(final_config, cur_config)

  if config_dict:
    deep_update(final_config, config_dict)

  final_config['run_local_time'] = get_local_time()

  final_config = convert_config_dict(final_config)
  return final_config


def parse_command_line_args(unparsed: list[str]) -> dict[str, Any]:
  """Parses command line arguments and returns a dictionary of key-value pairs.

  Supports nested configuration via dot notation (e.g., --multimodal.enable=true).

  Args:
      unparsed (list[str]): A list of command line arguments in the format
        '--key=value' or '--parent.child=value'.

  Returns:
      dict: A dictionary containing the parsed key-value pairs with nested
        structure for dot-separated keys.

  Example:
      >>> parse_command_line_args(['--name=John', '--age=25',
      '--is_student=True', '--multimodal.enable=false'])
      {'name': 'John', 'age': 25, 'is_student': True,
       'multimodal': {'enable': False}}
  """
  args = {}
  for text_arg in unparsed:
    if '=' not in text_arg:
      raise ValueError(
          f"Invalid command line argument: {text_arg}, please add '=' to"
          ' separate key and value.'
      )
    key, value = text_arg.split('=', 1)
    key = key[len('--') :]
    try:
      value = _convert_value(value)
    except (ValueError, TypeError):
      pass

    # Handle nested configuration (e.g., multimodal.enable -> {'multimodal': {'enable': ...}})
    if '.' in key:
      keys = key.split('.')
      current = args
      for i, k in enumerate(keys[:-1]):
        if k not in current:
          current[k] = {}
        elif not isinstance(current[k], dict):
          # Key conflict: existing non-dict value
          raise ValueError(
              f"Cannot set nested key '{key}': '{k}' already has a non-dict value"
          )
        current = current[k]
      current[keys[-1]] = value
    else:
      args[key] = value
  return args


def download_file(url: str, path: str) -> None:
  """Downloads a file from the given URL and saves it to the specified path.

  Args:
      url (str): The URL of the file to download.
      path (str): The path where the downloaded file will be saved.
  """
  logger = logging.getLogger()
  response = requests.get(url, verify=False)
  if response.status_code == 200:
    with open(path, 'wb') as f:
      f.write(response.content)
    logger.info('Downloaded %s', os.path.basename(path))
  else:
    logger.error('Failed to download %s', os.path.basename(path))


def list_to_str(l: Union[list[Any], str], remove_blank=False) -> str:
  """Converts a list or a string to a string representation.

  Args:
      l (Union[list, str]): The input list or string.
      remove_blank (bool): Whether to remove blank spaces from the string.

  Returns:
      str: The string representation of the input.
  """
  if isinstance(l, list):
    ret = ', '.join(map(str, l))
  else:
    ret = l
  if remove_blank:
    ret = ret.replace(' ', '')
  return ret


def clean_text(raw_text: str) -> str:
  """Cleans the raw text by removing HTML tags, special characters, and extra spaces.

  Args:
      raw_text (str): The raw text to be cleaned.

  Returns:
      str: The cleaned text.
  """
  text = list_to_str(raw_text)
  text = html.unescape(text)
  text = text.strip()
  text = re.sub(r'<[^>]+>', '', text)
  text = re.sub(r'[\n\t]', ' ', text)
  text = re.sub(r' +', ' ', text)
  text = re.sub(r'[^\x00-\x7F]', ' ', text)
  return text


def init_device():
  """Set the visible devices for training. Supports multiple GPUs.

  Returns:
      torch.device: The device to use for training.
  """
  use_ddp = (
      True if os.environ.get('WORLD_SIZE') else False
  )  # Check if DDP is enabled
  if torch.cuda.is_available():
    return torch.device('cuda'), use_ddp
  else:
    return torch.device('cpu'), use_ddp


def config_for_log(config: dict[str, Any]) -> dict[str, Any]:
  """Prepares the configuration dictionary for logging by removing unnecessary keys and converting list values to strings.

  Args:
      config (dict): The configuration dictionary.

  Returns:
      dict: The configuration dictionary prepared for logging.
  """
  config = config.copy()
  config.pop('device', None)
  config.pop('accelerator', None)
  for k, v in config.items():
    if isinstance(v, list):
      config[k] = str(v)
  return config
