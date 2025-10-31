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

"""Core ActionPiece tokenizer."""

import collections
import json
import queue
from typing import Any, List

from genrec.models.ActionPiece.utils import LinkedListState
import numpy as np
import tqdm

tqdm = tqdm.tqdm
PriorityQueue = queue.PriorityQueue


def diff_cnt(cnt1, cnt2):
  """Minus the second pair2cnt from the first pair2cnt.

  Args:
      cnt1 (dict): The first pair2cnt.
      cnt2 (dict): The second pair2cnt.

  Returns:
      dict: A duplication, not inplace.
  """
  return {k: v - cnt2.get(k, 0) for k, v in cnt1.items()}


def add_cnt_inplace(cnt1, cnt2):
  """Add pair2cnt inplace."""
  for k, v in cnt2.items():
    cnt1[k] += v


class ActionPieceCore:
  """The core ActionPiece tokenizer.

  Note that this class can be initialized in three ways:
  1. From state2feat, which is a dict mapping state (str) to
      features (list[int]). These features are used as the initial vocabulary.
  2. From metadata, which is a dict containing the metadata of a constructed
  ActionPiece.
      The metadata is saved by the save() method.
  3. (highly recommended) using the from_pretrained() method.
      Input the path to the metadata file, and it will load the ActionPiece
      from the metadata.

  When counting the pairs of tokens.
  The weights inside a state is 2 / M, where M is the number of tokens in the
  state.
  The weights between states is 1 / (M1 * M2), where M1 and M2 are the number
  of tokens in the two states.

  Attributes:
      vocab (list): The vocabulary of ActionPiece (same as token2feat).
      rank (dict): The rank of tokens (same as feat2token). The tokens are
        defined as the index (rank) in the vocabulary. Key is a tuple of either:
        a. (category_idx, feature_idx), indicating the token is an initial
        feature. b. (-1, token_idx1, token_idx2), indicating a merging rule of
        two previous tokens. Value is the token index in the vocabulary.
      vocab_size: The size of the vocabulary.
      token2feat: The mapping from token to features.
      feat2token: The mapping from features to token.
      token2all_feat: The mapping from token to the most basic features.
      state2feat: The initial vocabulary of ActionPiece.
      cur_corpus: The current corpus of linked lists.
      head_id2pair_cnt: The mapping from head_id to the pair counts in the
        corresponding linked list.
      pair2head_ids: The mapping from pair of tokens to the head ids of the
        linked lists that contain the pair.
      metadata: The metadata of a learned ActionPiece.
      priority: The priority of each token.
      n_categories: The number of categories of the features.
      pq: The priority queue to find the maximum appeared pair in O(logH).
      n_init_feats: The number of initial features.
      eps: A small number to avoid numerical issues.
      all_pair2cnt: The mapping from pair of tokens to the total count of the
        pair in all the sequences.
  """

  def __init__(self, state2feat=None, metadata=None):
    self.state2feat = state2feat
    self.metadata = metadata
    self.token2all_feat = {}

    if self.state2feat is not None:
      self.n_categories, self.token2feat, self.feat2token, self.priority = (
          self._init_from_state2feat(state2feat)
      )
      self.n_init_feats = len(self.token2feat)
    elif metadata is not None:
      (
          self.n_categories,
          self.n_init_feats,
          self.token2feat,
          self.feat2token,
          self.priority,
      ) = self._init_from_metadata(metadata)
    else:
      raise ValueError(
          'Check that one of state2feat and metadata is None.'
      )
    self.eps = 1e-12

  @property
  def vocab(self):
    return self.token2feat

  @property
  def rank(self):
    return self.feat2token

  @property
  def vocab_size(self):
    return len(self.token2feat)

  def _init_from_state2feat(self, state2feat: dict[str, List[int]]):
    """Only initialize using the most basic features from state2feat.

    ActionPiece initialized by this method has not been trained.

    Args:
        state2feat (Dict[str, List[int]]): The initial vocabulary of
          ActionPiece.

    Returns:
        tuple: A tuple containing:
          - n_categories (int): The number of categories of the features.
          - vocab (list): The vocabulary of ActionPiece (same as token2feat).
          - rank (dict): The rank of tokens (same as feat2token).
          - priority (list): The priority of each token.
    """
    vocab = [(-1, -1)]  # The first token is the padding token
    rank = {(-1, -1): 0}
    priority = [0]
    feats = np.array(list(state2feat.values()))
    for i in range(feats.shape[1]):
      for j in np.unique(feats[:, i]).tolist():
        rank[(i, j)] = len(vocab)
        vocab.append((i, j))
        priority.append(0)
    return feats.shape[1], vocab, rank, priority

  def _init_from_metadata(self, metadata: dict[str, Any]):
    """Initialize ActionPiece from the metadata of a trained ActionPiece."""
    n_categories = metadata['n_categories']
    n_init_feats = metadata['n_init_feats']
    token2feat = [tuple(_) for _ in metadata['token2feat']]
    feat2token = {feat: token for token, feat in enumerate(token2feat)}
    priority = [float(_) for _ in metadata['priority']]
    return n_categories, n_init_feats, token2feat, feat2token, priority

  def save(self, save_path):
    """Save ActionPiece to a metadata file.

    Args:
        save_path (str): The path to the metadata file.
    """
    data = {
        'n_categories': self.n_categories,
        'n_init_feats': self.n_init_feats,
        'token2feat': self.token2feat,
        'priority': self.priority,
    }
    with open(save_path, 'w') as f:
      json.dump(data, f)

  @classmethod
  def from_pretrained(cls, save_path, vocab_size=None):
    """Initialize ActionPiece from a saved file of a pretrained ActionPiece.

    Args:
        save_path (str): The path to the metadata file.
        vocab_size (int): The target vocab size. If not None, the vocab will be
          truncated or padded to the target size.

    Returns:
        actionpiece (ActionPieceCore):
            The initialized ActionPiece.
    """
    with open(save_path, 'r') as f:
      metadata = json.load(f)
    if vocab_size is not None:
      assert vocab_size >= metadata['n_init_feats'], (
          f'The target vocab size ({vocab_size}) must be larger than the'
          f' initial vocab size ({metadata["n_init_feats"]})'
      )
      assert vocab_size <= len(metadata['token2feat']), (
          f'The target vocab size ({vocab_size}) must be smaller than the'
          f' number of tokens ({len(metadata["token2feat"])})'
      )
      metadata['token2feat'] = metadata['token2feat'][:vocab_size]
      metadata['priority'] = metadata['priority'][:vocab_size]
    actionpiece = cls(metadata=metadata)
    return actionpiece

  def _construct_linked_list(self, head_id, state_seq):
    """Construct the linked list for a single state sequence.

    Args:
        head_id (int): The head id of the linked list.
        state_seq (list[list[int]]): The state sequence. state_seq[i] is a
          state. state_seq[i][j] is the j-th feature/token of the state.

    Returns:
        LinkedListState: The head of the constructed linked list.
    """
    state_seq = state_seq.tolist()
    head = LinkedListState(state_seq[0], head_id, context=False)
    tail = head
    for state in state_seq[1:]:
      # Append context slot
      tail = tail.append(LinkedListState([], head_id, context=True))
      # Append regular state
      tail = tail.append(LinkedListState(state, head_id, context=False))
    return head

  def _count_pairs_inside_state(self, state):
    """Count the pairs of tokens inside a single state.

    Combination of 2 out of M tokens. Self pairs are not included.

    Args:
        state (list[int]): The list of tokens in the state.

    Returns:
        pair2cnt (dict): The dictionary of pairs of tokens and their counts.
    """
    pair2cnt = collections.defaultdict(float)
    for p, tk1 in enumerate(state):
      for tk2 in state[p+1:]:
        pair2cnt[(min(tk1, tk2), max(tk1, tk2))] += 2 / len(state)
    return pair2cnt

  def _count_pairs_btw_states(self, state1, state2):
    """Iterate all the pairs of tokens between two states."""
    pair2cnt = collections.defaultdict(float)
    for tk1 in state1:
      for tk2 in state2:
        pair2cnt[(min(tk1, tk2), max(tk1, tk2))] += 1 / (
            len(state1) * len(state2)
        )
    return pair2cnt

  def _count_pairs_in_list(self, head):
    """Count the pairs of tokens in a single linked list."""
    pair2cnt = collections.defaultdict(float)
    cur_node = head
    while cur_node:
      # Count the pairs inside a state, iterate combination of 2 out of M
      add_cnt_inplace(pair2cnt, self._count_pairs_inside_state(cur_node.state))
      if not cur_node.next:
        # The last node, no need to count the pairs between states
        break
      # If the next context slot is not empty, count the pairs between
      # the next context slot and the ajacent regular states.
      if cur_node.next.state:
        # Count the pairs between the next context slot and the current
        # regular state.
        add_cnt_inplace(
            pair2cnt,
            self._count_pairs_btw_states(cur_node.state, cur_node.next.state),
        )
        # Count the pairs between the next context slot and the next
        # regular state.
        add_cnt_inplace(
            pair2cnt,
            self._count_pairs_btw_states(
                cur_node.next.state, cur_node.next.next.state
            ),
        )
      # Otherwise, count the pairs between the next regular state
      # and the current regular state.
      else:
        add_cnt_inplace(
            pair2cnt,
            self._count_pairs_btw_states(
                cur_node.state, cur_node.next.next.state
            ),
        )
      cur_node = cur_node.next.next
    return pair2cnt

  def _build(self, token_corpus):
    """Build the data structures for the training process.

    Args:
        token_corpus: list[np.ndarray] token_corpus[i] is a state sequence
          (np.ndarray of shape (*, n_categories)). token_corpus[i][j] is a
          state. token_corpus[i][j][k] is the k-th feature/token of the state.
    """

    # Construct the linked list for each sequence
    self.cur_corpus = [
        self._construct_linked_list(i, state_seq)
        for i, state_seq in enumerate(token_corpus)
    ]
    # Count the pairs of tokens in each sequence
    self.head_id2pair_cnt = []

    # For each pair of tokens, find the sequences that contain the pair.
    # This can be seen as an inverted index.
    self.pair2head_ids = collections.defaultdict(set)
    # Maintain the total count of each pair of tokens in all the sequences.
    self.all_pair2cnt = collections.defaultdict(float)
    for head in self.cur_corpus:
      head_id = head.head_id
      # Count the pairs of tokens in a single sequence
      pair2cnt = self._count_pairs_in_list(head)
      # Inplace update the total count of each pair of tokens
      # in all the sequences.
      add_cnt_inplace(self.all_pair2cnt, pair2cnt)
      # For each pair that appears in the sequence, add the head id
      # to the inverted index.
      for pair in pair2cnt:
        self.pair2head_ids[pair].add(head_id)
      # Maintain the pair2cnt of the current sequence.
      self.head_id2pair_cnt.append(pair2cnt)

    # Build the priority queue to find the maximum appeared pair in O(logH).
    self.pq = PriorityQueue()
    for (tk1, tk2), cnt in self.all_pair2cnt.items():
      # Note that in Python, the priority queue is a min heap.
      # Thus, we need to negate the count to make it a max heap.
      self.pq.put((-cnt, (tk1, tk2)))

  def _outdated(self, pair, priority):
    """The priority queue (heap) will be lazy updated, meaning that.

        we always insert latest <pair, cnt> into the heap,
        but never delete the outdated pairs.

    This function checks if the pair is outdated.

    Each time we get the pair with maximum appearance, we will check if
    the appearance count is the same as what we maintain in `all_pair2cnt`.

    If they are the same, it means the pair is not outdated.
    If they are different, it means the pair is outdated.

    Args:
        pair (tuple[int, int]): The pair of tokens.
        priority (float): The priority of the pair.

    Returns:
        bool: True if the pair is outdated, False otherwise.
    """
    return abs(priority - self.all_pair2cnt[pair]) > self.eps

  def _merge_empty_nodes(self, head):
    """Merge empty nodes in the linked list.

    Args:
        head (LinkedListState): The head of the linked list.

    Returns:
        head (LinkedListState): The head of the updated linked list.
    """
    updated_flag = True
    while updated_flag:
      updated_flag = False
      # Update the empty regular state to the previous context slot
      cur_node = head
      while cur_node:
        if cur_node.context:
          cur_node = cur_node.next
          continue
        if not cur_node.state and cur_node.prev:
          # Cannot be the first node
          if cur_node.prev.context and cur_node.prev.state:
            cur_node.state = cur_node.prev.state
            cur_node.prev.state = []
            updated_flag = True
        cur_node = cur_node.next
      # Remove the consequtive pair of
      # empty regular state and empty context slot
      cur_node = head
      while cur_node:
        if cur_node.context:
          cur_node = cur_node.next
          continue
        if not cur_node.next:
          break
        if not cur_node.state and not cur_node.next.state:
          if not cur_node.prev:
            # Head should be removed
            head = cur_node.next.next
            cur_node.next.next.prev = None
          else:
            cur_node.prev.next = cur_node.next.next
            cur_node.next.next.prev = cur_node.prev
          updated_flag = True
          cur_node = cur_node.next.next
        else:
          cur_node = cur_node.next
    return head

  def _merge_inside_regular_state(self, node, rule, new_token):
    """Merge the tokens inside a regular state."""
    if rule[0] == rule[1]:
      # One state never has the same token twice
      return
    if rule[0] in node.state and rule[1] in node.state:
      # new_token always appears first, i.e., coarse-to-fine.
      node.state = [new_token] + [
          state for state in node.state if state not in rule
      ]

  def _merge_state_context(self, state_node, context_node, rule, new_token):
    """Merge the tokens between a regular state and a context slot.

    The merged token will be inserted into the context slot.

    Args:
        state_node (LinkedListState): The regular state node.
        context_node (LinkedListState): The context slot node.
        rule (tuple[int, int]): The merging rule.
        new_token (int): The new token to be inserted.
    """
    assert len(context_node.state) == 1
    if rule[0] == context_node.state[0]:
      if rule[1] in state_node.state:
        state_node.state = [_ for _ in state_node.state if _ != rule[1]]
        context_node.state = [new_token]
    elif rule[1] == context_node.state[0]:
      if rule[0] in state_node.state:
        state_node.state = [_ for _ in state_node.state if _ != rule[0]]
        context_node.state = [new_token]

  def _merge_two_states(self, node1, node2, rule, new_token):
    """Merge the tokens between two regular states.

    The merged token will be inserted into the context slot
        between the two states (where node1.next.next == node2)

    Args:
        node1 (LinkedListState): The first regular state.
        node2 (LinkedListState): The second regular state.
        rule (tuple[int, int]): The merging rule.
        new_token (int): The new token to be inserted.
    """
    # Only possible to merge if the context slot is empty
    assert not node1.next.state
    if rule[0] in node1.state and rule[1] in node2.state:
      # Update regular states
      node1.state = [item for item in node1.state if item != rule[0]]
      node2.state = [item for item in node2.state if item != rule[1]]
      # Update context slot
      node1.next.state = [new_token]
    elif rule[1] in node1.state and rule[0] in node2.state:
      # Update regular states
      node1.state = [item for item in node1.state if item != rule[1]]
      node2.state = [item for item in node2.state if item != rule[0]]
      # Update context slot
      node2.prev.state = [new_token]

  def _merge_single_rule(self, head, rule, new_token):
    """Merge the tokens in the linked list according to the new merging rule.

    Args:
        head (LinkedListState): The head of the linked list.
        rule (tuple[int, int]): The new merging rule.
        new_token (int): The new token to be inserted.

    Returns:
        head (LinkedListState): The head of the updated linked list.
    """
    # Make a copy of the old linked list.
    # All the changes will be made on the new linked list immediately.
    new_link = head.copy_link()
    cur_node = new_link
    while cur_node:
      assert not cur_node.context, 'cur_node should be a regular state'
      # Regular state, check inside
      self._merge_inside_regular_state(cur_node, rule, new_token)
      if not cur_node.next:
        break  # The last node
      if cur_node.next.state:  # Token in context slot
        # Check (regular state, context slot)
        self._merge_state_context(cur_node, cur_node.next, rule, new_token)
        # Check (context slot, regular state)
        self._merge_state_context(
            cur_node.next.next, cur_node.next, rule, new_token
        )
      else:
        # Check (regular state, regular state)
        self._merge_two_states(cur_node, cur_node.next.next, rule, new_token)

      # Move to the next regular state
      cur_node = cur_node.next.next
    return self._merge_empty_nodes(new_link)

  def _update_pair2head_ids(self, diff_pair2cnt, head_id):
    """Update the inverted index of the pair2head_ids based on the diff of the pair counting.

    Args:
        diff_pair2cnt (dict): The difference of pair counting.
        head_id (int): The head id of the linked list.
    """
    for pair in diff_pair2cnt:
      if (
          diff_pair2cnt[pair] > 0
          and abs(self.head_id2pair_cnt[head_id][pair]) < self.eps
      ):
        # New pair after merging
        assert (
            head_id not in self.pair2head_ids[pair]
        ), f'head_id {head_id} already in pair2head_ids[{pair}]'
        self.pair2head_ids[pair].add(head_id)
      elif (
          diff_pair2cnt[pair] < 0
          and abs(self.head_id2pair_cnt[head_id][pair] + diff_pair2cnt[pair])
          < self.eps
      ):
        # Disappear pair after merging
        assert (
            head_id in self.pair2head_ids[pair]
        ), f'head_id {head_id} not in pair2head_ids[{pair}]'
        self.pair2head_ids[pair].remove(head_id)

  def _update_pq(self, diff):
    """Update the priority queue using the lazy update strategy.

    We always insert the latest <pair, cnt> into the heap,
    but never delete the outdated pairs.

    We also maintain the total count of each pair in all the sequences,
    which is used to check if one pair got from the heap is outdated.

    Args:
        diff (dict): The difference of pair counting.
    """
    for pair in diff:
      if abs(diff[pair]) < self.eps:
        # No change, thus no need to update
        continue
      self.all_pair2cnt[pair] += diff[pair]
      # Note that in Python, the priority queue is a min heap.
      # Thus, we need to negate the count to make it a max heap.
      self.pq.put((-self.all_pair2cnt[pair], pair))

  def _get_token_corpus(self, state_corpus):
    """Get the token corpus from the state corpus.

    Args:
        state_corpus (list[list[str]]): The state corpus to get the token
          corpus.

    Returns:
        token_corpus (list[np.ndarray]):
            The token corpus.
    """
    token_corpus = []
    for state_seq in state_corpus:
      token_seq = [
          [self.feat2token[it] for it in enumerate(self.state2feat[state])]
          for state in state_seq
      ]
      token_corpus.append(np.array(token_seq))
    return token_corpus

  def train(
      self,
      state_corpus,
      target_vocab_size: int,
  ):
    """Train the ActionPiece tokenizer.

    Args:
        state_corpus (list[list[str]]): The state corpus to train the
          ActionPiece.
        target_vocab_size (int): The target vocabulary size.
    """
    token_corpus = self._get_token_corpus(state_corpus)
    # Build the data structures for the training process
    self._build(token_corpus)

    progress_bar = tqdm(range(target_vocab_size - self.n_init_feats))
    while len(self.vocab) < target_vocab_size:
      # Train for one step, the vocab size will be increased by 1
      self._train_step()
      progress_bar.set_description(
          f'[Vocab size: {len(self.vocab)} / {target_vocab_size}] '
      )
      progress_bar.update(1)
    progress_bar.close()

  def _train_step(self):
    """The difference is additionally recording priority scores here."""
    priority, tk1, tk2 = None, None, None
    while not self.pq.empty():
      # Get the pair with maximum appearance
      # If the pair is outdated, just ignore.
      # Will repeat until the fetched pair is not outdated.
      priority, (tk1, tk2) = self.pq.get()
      if not self._outdated((tk1, tk2), -priority):
        break

    # Add the new token to the vocabulary
    new_rule = (-1, tk1, tk2)
    new_token = len(self.vocab)
    self.rank[new_rule] = new_token
    self.vocab.append(new_rule)
    self.priority.append(-priority)

    # Update data structures.
    # Only update the sequences that contain the token pair to merge.
    # These heads point to the sequences that need to be updated.
    head_to_update = self.pair2head_ids[(tk1, tk2)].copy()

    # Count the diff of pairs in all the sequence to be updated.
    all_diff = collections.defaultdict(int)
    for head_id in head_to_update:
      # Update the linked list according to the new merging rule.
      self.cur_corpus[head_id] = self._merge_single_rule(
          self.cur_corpus[head_id], rule=(tk1, tk2), new_token=new_token
      )
      # Count the pairs in the updated sequence.
      new_pair2cnt = self._count_pairs_in_list(self.cur_corpus[head_id])
      # Count the diff of pair counting between the updated sequence and
      # the old sequence.
      diff_pair2cnt = diff_cnt(new_pair2cnt, self.head_id2pair_cnt[head_id])
      # Update the inverted index based on how the counting changes.
      self._update_pair2head_ids(diff_pair2cnt, head_id)
      self.head_id2pair_cnt[head_id] = new_pair2cnt
      # Update the total counting of pairs.
      add_cnt_inplace(all_diff, diff_pair2cnt)
    # Update the priority queue of the updated pair appearances.
    self._update_pq(all_diff)

  def _random_walk_augmentation(self, state_seq: np.ndarray):
    """Random walk augmentation, flatten the state sequence into a sequence of initial tokens.

    Args:
        state_seq (np.ndarray): The state sequence to augment, shape (N,
          n_categories).

    Returns:
        aug_state_seq (list[int]):
            The augmented state sequence, shape (N * n_categories).

    Note:
        each random walk will cover all features,
            i.e. length(random walk seq) == n_categories.
    """
    aug_state_seq = []
    for seq in state_seq:
      aug_state_seq.extend(np.random.permutation(seq).tolist())
    return aug_state_seq

  def _encode(self, seq):
    """Encode a flattened feature sequence into a token sequence.

    The encoding process is just like BPE encoding. Usually the input seq is the
    output of _random_walk_augmentation().

    Args:
        seq (list[int]): The feature sequence to encode.

    Returns:
        enc_seq (list[int]):
            The encoded token sequence.
    """
    while True:
      min_idx = None
      min_rank = float('inf')
      for i, (tk1, tk2) in enumerate(zip(seq[:-1], seq[1:])):
        tk1, tk2 = min(tk1, tk2), max(tk1, tk2)
        cur_rank = self.rank.get((-1, tk1, tk2))
        if cur_rank is not None and cur_rank < min_rank:
          min_idx = i
          min_rank = cur_rank
      if min_idx is None:
        break
      seq = seq[:min_idx] + [min_rank] + seq[min_idx + 2 :]
    return seq

  def encode_fast(self, state_seq):
    aug_state_seq = self._random_walk_augmentation(state_seq)
    return self._encode(aug_state_seq)

  def encode(self, state_seq, shuffle='feature'):
    """Encode the state sequence into a list of tokens.

    Args:
        state_seq (np.ndarray): The state sequence.
        shuffle (str): The shuffle strategy. 'feature': random walk
          augmentation. 'token': enuemrate all the pairs of tokens, merge, and
          shuffle inside the state. 'none': enuemrate all the pairs of tokens,
          merge.

    Returns:
        encoded_seq (list[int]):
            The encoded state sequence.
    """

    def _count_inside_ll(node, updates):
      """Count the best pair of tokens inside a single state."""
      best_priority, node_to_update, rule_to_update = updates
      for i, tk1 in enumerate(node.state):
        for tk2 in node.state[i + 1 :]:
          cur_rule = (-1, min(tk1, tk2), max(tk1, tk2))
          if cur_rule not in self.rank:
            continue
          score = self.priority[self.rank[cur_rule]] * 2 / len(node.state)
          if best_priority is None or score > best_priority:
            best_priority = score
            node_to_update = (node,)
            rule_to_update = cur_rule
      return (best_priority, node_to_update, rule_to_update)

    def _count_two_states_ll(node1, node2, updates):
      """Count the best pair of tokens between two states."""
      best_priority, node_to_update, rule_to_update = updates
      for tk1 in node1.state:
        for tk2 in node2.state:
          cur_rule = (-1, min(tk1, tk2), max(tk1, tk2))
          if cur_rule not in self.rank:
            continue
          score = self.priority[self.rank[cur_rule]] / (
              len(node1.state) * len(node2.state)
          )
          if best_priority is None or score > best_priority:
            best_priority = score
            node_to_update = (node1, node2)
            rule_to_update = cur_rule
      return (best_priority, node_to_update, rule_to_update)

    if shuffle == 'feature':
      return self.encode_fast(state_seq)
    else:
      head = self._construct_linked_list(head_id=-1, state_seq=state_seq)  # type: ignore
      while True:
        # best_priority, node_to_update, rule_to_update
        cur_updates = (None, None, None)
        cur_node = head
        while cur_node:
          cur_updates = _count_inside_ll(cur_node, cur_updates)
          if not cur_node.next:
            break
          if cur_node.next.state:
            cur_updates = _count_two_states_ll(
                cur_node, cur_node.next, cur_updates
            )
            cur_updates = _count_two_states_ll(
                cur_node.next.next, cur_node.next, cur_updates
            )
          else:
            cur_updates = _count_two_states_ll(
                cur_node, cur_node.next.next, cur_updates
            )
          cur_node = cur_node.next.next
        if cur_updates[0] is None:
          break
        _, node_to_update, rule_to_update = cur_updates
        if len(node_to_update) == 1:
          self._merge_inside_regular_state(
              node_to_update[0],
              (rule_to_update[1], rule_to_update[2]),
              new_token=self.rank[rule_to_update],
          )
        else:
          if node_to_update[1].context:
            self._merge_state_context(
                node_to_update[0],
                node_to_update[1],
                (rule_to_update[1], rule_to_update[2]),
                new_token=self.rank[rule_to_update],
            )
          else:
            self._merge_two_states(
                node_to_update[0],
                node_to_update[1],
                (rule_to_update[1], rule_to_update[2]),
                new_token=self.rank[rule_to_update],
            )
        head = self._merge_empty_nodes(head)
      if shuffle == 'token':
        return head.to_shuffled_list()
      elif shuffle == 'none':
        return head.tolist()

  def _decode_single_token(self, token):
    """Decode a single token into the most basic features.

    Args:
        token (int): The token to decode.

    Returns:
        all_feat (list[tuple[int, int]]):
            The most basic features of the token.
    """
    if token in self.token2all_feat:
      return self.token2all_feat[token]
    decoded = self.vocab[token]
    if decoded[0] == -1:
      assert len(decoded) == 3, f'Invalid token: {token}'
      all_feat = self._decode_single_token(
          decoded[1]
      ) + self._decode_single_token(decoded[2])
    else:
      all_feat = [decoded]
    self.token2all_feat[token] = all_feat
    return all_feat

  def decode_single_state(self, token_seq):
    """Decode a sequence of tokens into the most basic features.

    The function assumes the token sequence is a valid single state.

    Args:
        token_seq (list[int]): The token sequence to decode.

    Returns:
        if None:
            The token sequence is not a valid single state.
        else:
            state (list[tuple[int, int]]):
                The most basic features of the state.
                Note that the features are sorted by the category index.
    """
    cur_state = {}
    for token in token_seq:
      if token == 0:
        return None
      if token >= len(self.vocab):
        print(f'Invalid token: {token}')
        return None
      feats = self._decode_single_token(token)
      for pos, f in feats:
        if pos in cur_state:
          return None
        cur_state[pos] = f
    for i in range(self.n_categories):
      if i not in cur_state:
        return None
    return [(i, cur_state[i]) for i in range(self.n_categories)]
