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

"""ActionPiece model for GenRec."""

import collections
from typing import Any

from genrec.model import AbstractModel
import numpy as np
import torch
import transformers

T5ForConditionalGeneration = transformers.T5ForConditionalGeneration
T5Config = transformers.T5Config


class ActionPiece(AbstractModel):
  """ActionPiece model for GenRec.

  This class implements the ActionPiece model, which is based on the T5
  architecture. It includes methods for forward passes, generation, and beam
  search.
  """

  def __init__(self, config, dataset, tokenizer):
    super().__init__(config, dataset, tokenizer)

    t5config = T5Config(
        num_layers=config['num_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        d_kv=config['d_kv'],
        dropout_rate=config['dropout_rate'],
        activation_function=config['activation_function'],
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.padding_token,
        eos_token_id=tokenizer.eos_token,
        decoder_start_token_id=0,
        feed_forward_proj=config['feed_forward_proj'],
        n_positions=tokenizer.max_token_seq_len,
    )

    self.t5 = T5ForConditionalGeneration(config=t5config)
    self.n_inference_ensemble = config['n_inference_ensemble']

  @property
  def n_parameters(self) -> str:
    """Calculates the number of trainable parameters in the model.

    Returns:
        str: A string containing the number of embedding parameters,
        non-embedding parameters, and total trainable parameters.
    """
    num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
    total_params = num_params(self.parameters())
    emb_params = num_params(self.t5.get_input_embeddings().parameters())
    return (
        f'#Embedding parameters: {emb_params}\n'
        f'#Non-embedding parameters: {total_params - emb_params}\n'
        f'#Total trainable parameters: {total_params}\n'
    )

  def forward(self, batch: dict[Any, Any]) -> torch.Tensor:
    """Forward pass of the model. Returns the output logits and the loss value.

    Args:
        batch (dict): A dictionary containing the input data for the model.

    Returns:
        outputs (ModelOutput):
            The output of the model, which includes:
            - loss (torch.Tensor)
            - logits (torch.Tensor)
    """
    outputs = self.t5(**batch)
    return outputs

  def generate(self, batch: dict[Any, Any], n_return_sequences: int = 1):
    """Ensembles the outputs of multiple random walk augmented inputs by their ranking scores (nDCG).

    Args:
        batch: The input batch.
        n_return_sequences: The number of sequences to return.

    Returns:
        The output sequences.
    """
    n_ensemble = 1
    if self.n_inference_ensemble != -1:
      if batch['input_ids'].shape[0] != batch['labels'].shape[0]:
        assert batch['input_ids'].shape[0] % self.n_inference_ensemble == 0
        n_ensemble = self.n_inference_ensemble
    batch_size = batch['input_ids'].shape[0] // n_ensemble

    outputs = self.beam_search(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        max_length=self.tokenizer.actionpiece.n_categories + 1,
        num_beams=self.config['num_beams'],
        num_return_sequences=n_return_sequences,
        return_score=False,
    )

    # decode the output states
    decoded_outputs = []
    for output in outputs.cpu()[:, 1:].tolist():
      if self.tokenizer.eos_token in output:
        idx = output.index(self.tokenizer.eos_token)
        output = output[:idx]
      else:
        output = output[: self.tokenizer.actionpiece.n_categories]
      # The output is valid when it can be decoded to a single state,
      # otherwise set to -1
      decoded_output = self.tokenizer.actionpiece.decode_single_state(output)
      if decoded_output is None:
        decoded_outputs.append([-1] * self.tokenizer.actionpiece.n_categories)
      else:
        decoded_outputs.append(
            [self.tokenizer.actionpiece.rank[_] for _ in decoded_output]
        )
    decoded_outputs = torch.LongTensor(decoded_outputs).reshape(
        batch_size, n_ensemble, n_return_sequences, -1
    )

    final_outputs = torch.full(
        (batch_size, n_return_sequences, decoded_outputs.shape[-1]),
        -1,
        dtype=torch.long,
    )
    for bid in range(batch_size):
      pred2score = collections.defaultdict(float)
      for i in range(n_ensemble):
        for j in range(n_return_sequences):
          pred = tuple(decoded_outputs[bid, i, j].tolist())
          if pred[0] != -1:
            pred2score[pred] += 1 / np.log2(j + 2)
      all_scores = [(pred, score) for pred, score in pred2score.items()]
      all_scores.sort(key=lambda x: x[1], reverse=True)
      for j in range(min(n_return_sequences, len(all_scores))):
        final_outputs[bid, j] = torch.LongTensor(all_scores[j][0])
    return final_outputs.to(batch['labels'].device)

  def beam_search(
      self,
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
      max_length: int = 6,
      num_beams: int = 1,
      num_return_sequences: int = 1,
      return_score: bool = False,
  ):
    """Adapted from huggingface's implementation.

    https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

    Perform beam search to generate sequences using the specified model.

    This implementation does not include stopping conditions based on
    end-of-sequence (EOS) tokens. Instead, the
    sequence generation is controlled solely by the `max_length` parameter.

    In scenarios where the generation should explicitly detect and respond
    to EOS tokens to terminate the sequence early, this function would need
    modifications. In the current setup, setting `max_length` to a suitable
    fixed value (e.g., 6) can serve the purpose by limiting the maximum sequence
    length.

    Args:
      input_ids (torch.Tensor): Tensor of input ids.
      attention_mask (torch.Tensor): Tensor representing the attention mask.
      max_length (int): Maximum length of the sequence to be generated; controls
        when to stop extending the sequence.
      num_beams (int): Number of beams for beam search.
      num_return_sequences (int): Number of sequences to return.
      return_score (bool): If True, returns a tuple of (sequences, scores) where
        'scores' are the average log likelihood of the returned sequences.

    Returns:
      torch.Tensor: The final decoder input ids from the beam search, or a tuple
        of (decoder_input_ids, scores) if 'return_score' is True.

    Example usage:
      Assuming the model, input_ids, and attention_mask are predefined:
      sequences = beam_search(model, input_ids, attention_mask, max_length=6,
      num_beams=5, num_return_sequences=5)
    """

    batch_size = input_ids.shape[0]

    # Prepare beam search inputs
    (
        input_ids,
        attention_mask,
        decoder_input_ids,
        beam_scores,
        beam_idx_offset,
    ) = self.prepare_beam_search_inputs(
        input_ids, attention_mask, batch_size, num_beams
    )
    # Store encoder_outputs to prevent running full forward path repeatedly
    with torch.no_grad():
      encoder_outputs = self.t5.get_encoder()(
          input_ids=input_ids, attention_mask=attention_mask, return_dict=True
      )

    # Beam search loop
    while decoder_input_ids.shape[1] < max_length:
      with torch.no_grad():
        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

      decoder_input_ids, beam_scores = self.beam_search_step(
          outputs.logits,
          decoder_input_ids,
          beam_scores,
          beam_idx_offset,
          batch_size,
          num_beams,
      )

    # (batch_size * num_beams, ) -> (batch_size * num_return_sequences, )
    selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
    selection_mask[:, :num_return_sequences] = True

    if return_score:
      return decoder_input_ids[selection_mask.view(-1), :], beam_scores[
          selection_mask.view(-1)
      ] / (decoder_input_ids.shape[1] - 1)

    return decoder_input_ids[selection_mask.view(-1), :]

  def prepare_beam_search_inputs(
      self,
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
      batch_size: int,
      num_beams: int,
  ):
    """Adapted from huggingface's implementation.

    https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

    Prepares and duplicates the input data for beam search decoding.

    This function initializes decoder input IDs and beam scores, creates an
    offset for beam indices,
    and expands the input_ids and attention_mask tensors to accommodate the
    specified number of beams for each instance in the batch.

    Args:
      input_ids (torch.Tensor): The input IDs tensor of shape (batch_size,
        sequence_length) used for the encoder part of the model.
      attention_mask (torch.Tensor): The attention mask tensor of shape
        (batch_size, sequence_length) indicating to the model which tokens
        should be attended to.
      batch_size (int): The number of instances per batch in the input data.
      num_beams (int): The number of beams to use in beam search. This expands
        the input data and scores accordingly.

    Returns:
      input_ids (torch.Tensor): The expanded input IDs tensor to match the
        number of beams, shape (batch_size * num_beams, sequence_length).
      attention_mask (torch.Tensor): The expanded attention mask tensor to match
        the number of beams, shape (batch_size * num_beams, sequence_length).
      initial_decoder_input_ids (torch.Tensor): The initialized decoder input
        IDs for each beam, shape (batch_size * num_beams, 1).
      initial_beam_scores (torch.Tensor): The initialized scores for each beam,
        flattened to a single dimension, shape (batch_size * num_beams,).
      beam_idx_offset (torch.Tensor): An offset for each beam index to assist in
        reordering beams during the search, shape (batch_size * num_beams,).

    Each input sequence is replicated 'num_beams' times to provide separate
    candidate paths in beam search. Beam scores are initialized with 0 for the
    first beam and a very low number (-1e9) for others to ensure the first token
    of each sequence is chosen from the first beam.
    """

    decoder_input_ids = torch.ones(
        (batch_size * num_beams, 1), device=self.t5.device, dtype=torch.long
    )
    initial_decoder_input_ids = (
        decoder_input_ids * self.t5.config.decoder_start_token_id
    )

    beam_scores = torch.zeros(
        (batch_size, num_beams), dtype=torch.float, device=input_ids.device
    )
    # Set a low score for all but the first beam to ensure the first beam is
    # selected initially
    beam_scores[:, 1:] = -1e9
    initial_beam_scores = beam_scores.view((batch_size * num_beams,))

    beam_idx_offset = (
        torch.arange(batch_size, device=self.t5.device).repeat_interleave(
            num_beams
        )
        * num_beams
    )

    input_ids = input_ids.repeat_interleave(num_beams, dim=0)
    attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

    return (
        input_ids,
        attention_mask,
        initial_decoder_input_ids,
        initial_beam_scores,
        beam_idx_offset,
    )

  def beam_search_step(
      self,
      logits,
      decoder_input_ids,
      beam_scores,
      beam_idx_offset,
      batch_size,
      num_beams,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapted from huggingface's implementation.

    https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

    Executes one step of beam search, calculating the next set of input IDs
    based on logits from a model.

    This function expands the current beam, calculates scores for all possible
    next tokens, selects the top tokens for each beam, and prepares the input
    IDs for the next iteration of the model. It utilizes logits output by the
    model to determine the most likely next tokens and updates the beam scores.

    Args:
      logits (torch.Tensor): Logits returned from the model, shape (batch_size *
        num_beams, sequence_length, vocab_size).
      decoder_input_ids (torch.Tensor): Current decoder input IDs, shape
        (batch_size * num_beams, current_sequence_length).
      beam_scores (torch.Tensor): Current scores for each beam, shape
        (batch_size * num_beams,).
      beam_idx_offset (torch.Tensor): Index offsets for each beam to handle
        batches correctly, shape (batch_size * num_beams,).
      batch_size (int): Number of sequences being processed in a batch.
      num_beams (int): Number of beams used in the beam search.

    Returns:
      decoder_input_ids (torch.Tensor): Updated decoder input IDs after adding
        the next tokens, shape (batch_size * num_beams, current_sequence_length
        + 1).
      beam_scores (torch.Tensor): Updated scores for each beam, shape
        (batch_size * num_beams,).

    The function selects the top `2 * num_beams` tokens from the logits based on
    their scores, reshapes and adjusts them based on the existing beam scores,
    and determines the next tokens to add to each beam path. The updated paths
    are then returned for use in the next iteration of the beam search.
    """
    assert batch_size * num_beams == logits.shape[0]

    vocab_size = logits.shape[-1]
    next_token_logits = logits[:, -1, :]
    next_token_scores = torch.log_softmax(
        next_token_logits, dim=-1
    )  # Calculate log softmax over the last dimension

    next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
        next_token_scores
    )
    next_token_scores = next_token_scores.view(
        batch_size, num_beams * vocab_size
    )
    next_token_scores, next_tokens = torch.topk(
        next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
    )

    next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
    next_tokens = next_tokens % vocab_size

    beam_scores = next_token_scores[:, :num_beams].reshape(-1)
    beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
    beam_idx = next_indices[:, :num_beams].reshape(-1)

    # beam_idx_offset: beam_idx contains sequence indicies relative to each
    # individual batch. We need to offset the indicies to retrieve the correct
    # sequence in the corresponding batch for example, when batch_size = 2,
    # beam_size = 3, beam_idx_offset = [0, 0, 0, 3, 3, 3]
    decoder_input_ids = torch.cat(
        [
            decoder_input_ids[beam_idx + beam_idx_offset, :],
            beam_next_tokens.unsqueeze(-1),
        ],
        dim=-1,
    )

    return decoder_input_ids, beam_scores
