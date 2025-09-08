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

"""Dataset classes for ActionPiece."""

import logging

from typing import Any
import datasets as datasets_lib


class AbstractDataset:
  """Abstract base class for datasets.

  This class provides a basic structure for loading, processing, and splitting
  datasets used in recommendation tasks. It includes methods for managing user
  and item mappings, handling item sequences, and splitting the data into
  training, validation, and test sets.

  Attributes:
      config (dict): Configuration parameters for the dataset.
      logger: Logger for the dataset.
      all_item_seqs (dict): Dictionary storing item sequences for each user.
      id_mapping (dict): Dictionary containing mappings between users/items and
        their IDs.
      item2meta (dict): Dictionary storing metadata for each item.
      split_data (dict): Dictionary storing the split datasets.
      n_users (int): The number of users in the dataset.
      n_items (int): The number of items in the dataset.
      n_interactions (int): The total number of interactions in the dataset.
      avg_item_seq_len (float): The average length of item sequences.
      user2id (dict): The user-to-id mapping.
      item2id (dict): The item-to-id mapping.
  """

  def __init__(self, config: dict[str, Any]):
    self.config = config
    self.logger = logging.getLogger()

    self.all_item_seqs = {}
    self.id_mapping = {
        'user2id': {'[PAD]': 0},
        'item2id': {'[PAD]': 0},
        'id2user': ['[PAD]'],
        'id2item': ['[PAD]'],
    }
    self.item2meta = None
    self.split_data = None

  def __str__(self) -> str:
    return (
        f'[Dataset] {self.__class__.__name__}\n'
        f'\tNumber of users: {self.n_users}\n'
        f'\tNumber of items: {self.n_items}\n'
        f'\tNumber of interactions: {self.n_interactions}\n'
        f'\tAverage item sequence length: {self.avg_item_seq_len}'
    )

  @property
  def accelerator(self) -> Any:
    """Returns the accelerator instance."""
    return self.config['accelerator']

  @property
  def n_users(self) -> int:
    """Returns the number of users in the dataset."""
    return len(self.user2id)

  @property
  def n_items(self) -> int:
    """Returns the total number of items in the dataset."""
    return len(self.item2id)

  @property
  def n_interactions(self) -> int:
    """Returns the total number of interactions in the dataset."""
    return sum(len(seq) for seq in self.all_item_seqs.values())

  @property
  def avg_item_seq_len(self) -> float:
    """Returns the average length of item sequences in the dataset."""
    return self.n_interactions / self.n_users

  @property
  def user2id(self) -> dict[str, int]:
    """Returns the user-to-id mapping."""
    return self.id_mapping['user2id']

  @property
  def item2id(self) -> dict[str, int]:
    """Returns the item-to-id mapping."""
    return self.id_mapping['item2id']

  def _download_and_process_raw(self):
    """This method should be implemented in the subclass.

    It is responsible for downloading and processing the raw data.
    """
    raise NotImplementedError(
        'This method should be implemented in the subclass'
    )

  def _leave_one_out(self) -> dict[str, datasets_lib.Dataset]:
    """Splits the dataset into train, validation, and test sets using the leave-one-out strategy.

    Returns:
        dict: A dictionary containing the train, validation, and test datasets.
              Each dataset is represented as a dictionary with 'user' and
              'item_seq' keys.
              The 'user' key contains a list of users, and the 'item_seq' key
              contains a list of item sequences.
    """
    datasets = {
        'train': {'user': [], 'item_seq': []},
        'val': {'user': [], 'item_seq': []},
        'test': {'user': [], 'item_seq': []},
    }
    for user, sequence in self.all_item_seqs.items():
      datasets['test']['user'].append(user)
      datasets['test']['item_seq'].append(sequence)
      if len(sequence) > 1:
        datasets['val']['user'].append(user)
        datasets['val']['item_seq'].append(sequence[:-1])
      if len(sequence) > 2:
        datasets['train']['user'].append(user)
        datasets['train']['item_seq'].append(sequence[:-2])
    return {k: datasets_lib.Dataset.from_dict(v) for k, v in datasets.items()}

  def split(self) -> dict[str, datasets_lib.Dataset]:
    """Split the dataset into train, validation, and test sets based on the specified split strategy."""
    if self.split_data is not None:
      return self.split_data

    split_strategy = self.config['split']
    if split_strategy in ['leave_one_out', 'last_out']:
      datasets = self._leave_one_out()
    else:
      raise NotImplementedError(
          f'Split strategy [{split_strategy}] not implemented.'
      )

    self.split_data = datasets
    return self.split_data

  def log(self, message: str, level: str = 'info') -> None:
    """Logs a message with the specified level."""
    from genrec.utils import log as log_lib
    return log_lib(
        message, self.config['accelerator'], self.logger, level=level
    )
