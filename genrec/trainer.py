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

"""Trainer for ActionPiece.

This module defines the Trainer class, which handles the training process for an
ActionPiece model. It includes methods for fitting the model, evaluating it, and
managing resources.
"""

import collections
import logging
import os
from typing import Any

from genrec.evaluator import Evaluator
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
import numpy as np
import torch
from torch import optim
from torch.nn import utils
import tqdm
from transformers import optimization
import hashlib
import sys

# WandB 延迟导入
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def get_command_line_args_str():
  """Get command line arguments as a string.

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

def config_for_log(config: dict[str, Any]) -> dict[str, Any]:
  """Prepares the configuration dictionary for logging by removing unnecessary keys and converting complex values to TensorBoard-compatible types.

  Args:
      config (dict): The configuration dictionary.

  Returns:
      dict: The configuration dictionary prepared for logging (flattened, with only int, float, str, bool types).
  """
  def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten nested dictionaries with dot notation."""
    items = []
    for k, v in d.items():
      new_key = f'{parent_key}{sep}{k}' if parent_key else k
      if isinstance(v, dict):
        items.extend(flatten_dict(v, new_key, sep=sep).items())
      else:
        items.append((new_key, v))
    return dict(items)

  # First flatten the config
  config = flatten_dict(config)

  # Remove unnecessary keys
  config.pop('device', None)
  config.pop('accelerator', None)

  # Convert all values to TensorBoard-compatible types
  cleaned_config = {}
  for k, v in config.items():
    if isinstance(v, (int, float, str, bool)):
      cleaned_config[k] = v
    elif isinstance(v, (list, tuple)):
      cleaned_config[k] = str(v)
    elif v is None:
      cleaned_config[k] = 'None'
    elif isinstance(v, torch.Tensor):
      cleaned_config[k] = v
    else:
      # Convert any other type to string
      cleaned_config[k] = str(v)

  return cleaned_config

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

get_scheduler = optimization.get_scheduler
tqdm = tqdm.tqdm
AdamW = optim.AdamW
clip_grad_norm_ = utils.clip_grad_norm_
getLogger = logging.getLogger
OrderedDict = collections.OrderedDict


class Trainer:
  """A class that handles the training process for a model.

  Attributes:
      config (dict): The configuration parameters for training.
      model (AbstractModel): The model to be trained.
      evaluator (Evaluator): The evaluator used for evaluating the model.
      logger (Logger): The logger used for logging training progress.
      project_dir (str): The directory path for saving tensorboard logs.
      saved_model_ckpt (str): The file path for saving the trained model
        checkpoint.
      accelerator: The accelerator used for training.

  Methods:
      fit(train_dataloader, val_dataloader): Trains the model using the provided
        training and validation dataloaders.
      evaluate(dataloader, split='test'): Evaluate the model on the given
        dataloader.
      end(): Ends the training process and releases any used resources.
  """

  def __init__(self, config: dict[Any, Any], model: AbstractModel,
               tokenizer: AbstractTokenizer):
    """Initializes the Trainer with the given configuration, model, and tokenizer.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the
          data.
    """
    self.config = config
    self.model = model
    self.accelerator = config['accelerator']
    self.evaluator = Evaluator(config, tokenizer)
    self.logger = getLogger()

    self.saved_model_ckpt = os.path.join(
        self.config['ckpt_dir'], get_file_name(self.config, suffix='.pth')
    )
    os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)
    
    # 初始化 WandB
    self.use_wandb = config.get('use_wandb', False) and WANDB_AVAILABLE
    if self.use_wandb and self.accelerator.is_main_process:
        self._init_wandb()

  def _init_wandb(self):
    """初始化 WandB"""
    try:
        # 准备运行名称
        run_name = self.config.get('wandb_name')
        if not run_name:
            run_name = get_file_name(self.config, suffix='')
        
        # 准备配置
        wandb_config = config_for_log(self.config.copy())
        
        # 准备标签
        tags = self.config.get('wandb_tags', [])
        if self.config.get('category'):
            tags.append(self.config['category'])
        if self.config.get('dataset'):
            tags.append(self.config['dataset'])
        
        # 准备注释
        notes = self.config.get('wandb_notes')
        if not notes:
            notes = f"ActionPiece training on {self.config.get('category', 'unknown')} dataset"
        
        # 初始化 WandB
        run = wandb.init(
            project=self.config.get('wandb_project', 'actionpiece'),
            entity=self.config.get('wandb_entity'),
            name=run_name,
            config=wandb_config,
            tags=tags,
            notes=notes,
            save_code=True,
            reinit=True
        )
        
        # 获取并打印WandB链接
        wandb_url = run.get_url()
        project_name = self.config.get('wandb_project', 'actionpiece')
        category = self.config.get('category', 'unknown')
        
        # 美观的输出
        print("\n" + "🚀 " + "="*77)
        print("   WEIGHTS & BIASES DASHBOARD READY")
        print("   " + "-"*50)
        print(f"   Dataset:  {category}")
        print(f"   Project:  {project_name}")
        print(f"   Run:      {run_name}")
        print(f"   URL:      {wandb_url}")
        print("   " + "-"*50)
        print("   📊 Click the link above to monitor training progress!")
        print("="*80 + "\n")
        
        # 监控模型
        wandb.watch(self.model, log='all', log_freq=100)
        
        self.log(f"WandB initialized: {wandb_url}")
        self.wandb_url = wandb_url
        
    except Exception as e:
        self.log(f"Failed to initialize WandB: {e}", level='warning')
        self.use_wandb = False
        self.wandb_url = None

  def fit(self, train_dataloader, val_dataloader):
    """Trains the model using the provided training and validation dataloaders.

    Args:
        train_dataloader: The dataloader for training data.
        val_dataloader: The dataloader for validation data.
    """
    # WandB 提醒
    if self.use_wandb and self.accelerator.is_main_process and hasattr(self, 'wandb_url'):
        print("🎯 Starting training... Track progress at: " + self.wandb_url)
    
    optimizer = AdamW(
        self.model.parameters(),
        lr=self.config['lr'],
        weight_decay=self.config['weight_decay'],
    )

    total_n_steps = get_total_steps(self.config, train_dataloader)
    if total_n_steps == 0:
      self.log('No training steps needed.')
      return

    scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=self.config['warmup_steps'],
        num_training_steps=total_n_steps,
    )

    self.model, optimizer, train_dataloader, val_dataloader, scheduler = (
        self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )
    )
    self.accelerator.init_trackers(
        project_name=get_file_name(self.config, suffix=''),
        config=config_for_log(self.config),
        init_kwargs={'tensorboard': {'flush_secs': 60}},
    )

    n_epochs = np.ceil(
        total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)
    ).astype(int)
    best_epoch = 0
    best_val_score = -1
    
    global_step = 0

    for epoch in range(n_epochs):
      # Training
      self.model.train()
      total_loss = 0.0
      epoch_losses = []
      
      train_progress_bar = tqdm(
          train_dataloader,
          total=len(train_dataloader),
          desc=f'Training - [Epoch {epoch + 1}]',
      )
      
      for step, batch in enumerate(train_progress_bar):
        optimizer.zero_grad()
        outputs = self.model(batch)
        loss = outputs.loss
        self.accelerator.backward(loss)
        
        if self.config['max_grad_norm'] is not None:
          clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
        
        optimizer.step()
        scheduler.step()
        
        loss_item = loss.item()
        total_loss += loss_item
        epoch_losses.append(loss_item)
        global_step += 1
        
        # 获取当前学习率
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.config['lr']
        
        # WandB 步级记录
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.log({
                'train/loss_step': loss_item,
                'train/learning_rate': current_lr,
                'train/epoch': epoch + 1,
                'train/global_step': global_step,
            }, step=global_step)
        
        # 更新进度条
        train_progress_bar.set_postfix({
            'loss': f'{loss_item:.4f}',
            'avg_loss': f'{total_loss / (step + 1):.4f}',
            'lr': f'{current_lr:.2e}'
        })

      # Epoch 级别统计
      avg_train_loss = total_loss / len(train_dataloader)
      train_loss_std = np.std(epoch_losses)
      
      # 记录到 accelerator
      self.accelerator.log(
          {'Loss/train_loss': avg_train_loss},
          step=epoch + 1,
      )
      
      # WandB epoch 级记录
      if self.use_wandb and self.accelerator.is_main_process:
          wandb.log({
              'train/loss_epoch': avg_train_loss,
              'train/loss_std': train_loss_std,
              'epoch': epoch + 1
          }, step=global_step)
      
      self.log(
          f'[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} ± {train_loss_std:.4f}'
      )

      # Evaluation
      if (epoch + 1) % self.config['eval_interval'] == 0:
        all_results = self.evaluate(val_dataloader, split='val')
        if self.accelerator.is_main_process:
          for key in all_results:
            self.accelerator.log(
                {f'Val_Metric/{key}': all_results[key]}, step=epoch + 1
            )
          
          # WandB 验证指标记录
          if self.use_wandb:
              wandb_metrics = {}
              for key, value in all_results.items():
                  wandb_metrics[f'val/{key}'] = value
              wandb_metrics['epoch'] = epoch + 1
              wandb.log(wandb_metrics, step=global_step)
          
          self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')

        val_score = all_results[self.config['val_metric']]
        if val_score > best_val_score:
          best_val_score = val_score
          best_epoch = epoch + 1
          
          # WandB 最佳结果记录
          if self.use_wandb and self.accelerator.is_main_process:
              wandb.log({
                  'best/val_score': best_val_score,
                  'best/epoch': best_epoch
              }, step=global_step)
          
          if self.accelerator.is_main_process:
            if self.config['use_ddp']:
              unwrapped_model = self.accelerator.unwrap_model(self.model)
              torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
            else:
              torch.save(self.model.state_dict(), self.saved_model_ckpt)
            self.log(
                f'[Epoch {epoch + 1}] Saved model checkpoint to'
                f' {self.saved_model_ckpt}'
            )

        if (
            self.config['patience'] is not None
            and epoch + 1 - best_epoch >= self.config['patience']
        ):
          self.log(f'Early stopping at epoch {epoch + 1}')
          if self.use_wandb and self.accelerator.is_main_process:
              wandb.log({'training/early_stopped': True}, step=global_step)
          break

    self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')
    
    # WandB 最终结果记录
    if self.use_wandb and self.accelerator.is_main_process:
        wandb.summary['best_epoch'] = best_epoch
        wandb.summary['best_val_score'] = best_val_score
        wandb.summary['total_epochs'] = epoch + 1

  def evaluate(self, dataloader, split='test'):
    """Evaluates the model on the given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
        split (str, optional): The split name. Defaults to 'test'.

    Returns:
        collections.OrderedDict: A dictionary containing the evaluation results.
    """
    self.model.eval()

    all_results = collections.defaultdict(list)
    val_progress_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f'Eval - {split}',
    )
    for batch in val_progress_bar:
      with torch.no_grad():
        batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
        if self.config[
            'use_ddp'
        ]:
          preds = self.model.module.generate(
              batch, n_return_sequences=self.evaluator.maxk
          )
          all_preds, all_labels = self.accelerator.gather_for_metrics(
              (preds, batch['labels'])
          )
          results = self.evaluator.calculate_metrics(all_preds, all_labels)
        else:
          preds = self.model.generate(
              batch, n_return_sequences=self.evaluator.maxk
          )
          results = self.evaluator.calculate_metrics(preds, batch['labels'])
        for key, value in results.items():
          all_results[key].append(value)

    output_results = OrderedDict()
    for metric in self.config['metrics']:
      for k in self.config['topk']:
        key = f'{metric}@{k}'
        output_results[key] = torch.cat(all_results[key]).mean().item()
    return output_results

  def end(self):
    """Ends the training process and releases any used resources."""
    self.accelerator.end_training()
    # 结束 WandB
    if self.use_wandb and self.accelerator.is_main_process:
        try:
            wandb.finish()
        except Exception as e:
            self.log(f"Error finishing WandB: {e}", level='warning')

  def log(self, message, level='info'):
    return log(message, self.config['accelerator'], self.logger, level=level)