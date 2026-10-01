"""Transformer-XL-style finite memory core for Sample Factory 2.1.x."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from sample_factory.model.core import ModelCore, default_make_core_func
from sample_factory.algo.utils.context import global_model_factory


class TransformerMemoryCore(ModelCore):
    """Attend over the current visual token and a fixed window of past tokens.

    Sample Factory exposes recurrent state as a flat tensor. We store a
    right-aligned window of projected visual tokens plus one valid-length value.
    The stored state is detached by the rollout buffer, like an RNN hidden state;
    gradients flow through all tokens within each recurrence chunk.
    """

    def __init__(self, cfg, input_size: int):
        super().__init__(cfg)
        self.model_dim = int(cfg.transformer_dim)
        self.context_len = int(cfg.transformer_context)
        expected_state_size = self.context_len * self.model_dim + 1
        if int(cfg.rnn_size) != expected_state_size:
            raise ValueError(
                f"Transformer state requires rnn_size={expected_state_size}, got {cfg.rnn_size}"
            )
        if self.model_dim % int(cfg.transformer_heads) != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads")

        self.input_projection = nn.Linear(input_size, self.model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.context_len, self.model_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(cfg.transformer_heads),
            dim_feedforward=int(cfg.transformer_ff_dim),
            dropout=float(cfg.transformer_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(cfg.transformer_layers),
            norm=nn.LayerNorm(self.model_dim),
        )
        self.output_projection = nn.Linear(self.model_dim, input_size)
        self.core_output_size = input_size
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def _step(self, x: torch.Tensor, state: torch.Tensor):
        batch = x.shape[0]
        memory = state[:, :-1].reshape(batch, self.context_len, self.model_dim)
        lengths = state[:, -1].round().long().clamp(0, self.context_len)
        token = self.input_projection(x).unsqueeze(1)
        memory = torch.cat((memory[:, 1:], token), dim=1)
        lengths = torch.clamp(lengths + 1, max=self.context_len)

        positions = torch.arange(self.context_len, device=x.device).unsqueeze(0)
        padding_mask = positions < (self.context_len - lengths).unsqueeze(1)
        encoded = self.transformer(memory + self.position_embedding, src_key_padding_mask=padding_mask)
        output = self.output_projection(encoded[:, -1])
        new_state = torch.cat((memory.reshape(batch, -1), lengths.to(memory.dtype).unsqueeze(1)), dim=1)
        return output, new_state

    def _packed_forward(self, packed: PackedSequence, state: torch.Tensor):
        # PackedSequence data is ordered by decreasing active batch size. Match
        # PyTorch RNN semantics by sorting the initial states the same way.
        if packed.sorted_indices is not None:
            state = state.index_select(0, packed.sorted_indices.to(state.device))

        outputs = []
        offset = 0
        for batch_size_tensor in packed.batch_sizes:
            batch_size = int(batch_size_tensor.item())
            x_t = packed.data[offset : offset + batch_size]
            active_state = state[:batch_size]
            y_t, next_state = self._step(x_t, active_state)
            outputs.append(y_t)
            if batch_size < state.shape[0]:
                state = torch.cat((next_state, state[batch_size:]), dim=0)
            else:
                state = next_state
            offset += batch_size

        if packed.unsorted_indices is not None:
            final_state = state.index_select(0, packed.unsorted_indices.to(state.device))
        else:
            final_state = state
        output_packed = PackedSequence(
            torch.cat(outputs, dim=0),
            packed.batch_sizes,
            packed.sorted_indices,
            packed.unsorted_indices,
        )
        return output_packed, final_state

    def forward(self, head_output, rnn_states):
        if isinstance(head_output, PackedSequence):
            return self._packed_forward(head_output, rnn_states)
        return self._step(head_output, rnn_states)


def make_temporal_core(cfg, input_size: int):
    if getattr(cfg, "memory", None) == "transformer":
        return TransformerMemoryCore(cfg, input_size)
    return default_make_core_func(cfg, input_size)


def register_transformer_core():
    global_model_factory().register_model_core_factory(make_temporal_core)
