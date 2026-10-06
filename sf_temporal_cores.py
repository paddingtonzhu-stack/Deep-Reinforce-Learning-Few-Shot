"""Optional temporal cores for matched Sample Factory APPO experiments.

The standard CNN and GRU use Sample Factory's built-in cores. This module adds
the Transformer, GTrXL, and GRU-attention alternatives.
"""
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


class GRUGatingUnit(nn.Module):
    """GRU-style residual gate used by GTrXL.

    A positive ``identity_bias`` initially favors the residual stream, which is
    the identity-map initialization described for stable Transformer RL agents.
    """

    def __init__(self, dim: int, identity_bias: float):
        super().__init__()
        self.reset_update = nn.Linear(2 * dim, 2 * dim)
        self.candidate = nn.Linear(2 * dim, dim)
        self.identity_bias = float(identity_bias)

    def forward(self, residual: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        reset_logits, update_logits = self.reset_update(
            torch.cat((residual, update), dim=-1)
        ).chunk(2, dim=-1)
        reset = torch.sigmoid(reset_logits)
        # Subtracting a positive bias makes the initial update gate small and
        # therefore preserves the residual pathway at initialization.
        update_gate = torch.sigmoid(update_logits - self.identity_bias)
        candidate = torch.tanh(
            self.candidate(torch.cat((reset * residual, update), dim=-1))
        )
        return (1.0 - update_gate) * residual + update_gate * candidate


class GTrXLBlock(nn.Module):
    """Pre-normalized attention/MLP block with gated residual connections."""

    def __init__(self, cfg):
        super().__init__()
        dim = int(cfg.transformer_dim)
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=int(cfg.transformer_heads),
            dropout=float(cfg.transformer_dropout),
            batch_first=True,
        )
        self.attention_gate = GRUGatingUnit(dim, cfg.gtrxl_identity_bias)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(cfg.transformer_ff_dim)),
            nn.GELU(),
            nn.Dropout(float(cfg.transformer_dropout)),
            nn.Linear(int(cfg.transformer_ff_dim), dim),
        )
        self.ffn_gate = GRUGatingUnit(dim, cfg.gtrxl_identity_bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = self.attention_gate(x, attended)
        x = self.ffn_gate(x, self.ffn(self.ffn_norm(x)))
        # A causal mask leaves early left-padding query rows with no valid key.
        # MultiheadAttention returns NaNs for those unused rows, so clear them
        # before another block can consume them. Valid query rows are unchanged.
        return x.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class GTrXLMemoryCore(TransformerMemoryCore):
    """Gated Transformer-XL-style finite-memory core for online RL."""

    def __init__(self, cfg, input_size: int):
        # Initialize ModelCore directly because the vanilla parent constructs a
        # different stack. State layout intentionally remains identical.
        ModelCore.__init__(self, cfg)
        self.model_dim = int(cfg.transformer_dim)
        self.context_len = int(cfg.transformer_context)
        expected_state_size = self.context_len * self.model_dim + 1
        if int(cfg.rnn_size) != expected_state_size:
            raise ValueError(
                f"GTrXL state requires rnn_size={expected_state_size}, got {cfg.rnn_size}"
            )
        if self.model_dim % int(cfg.transformer_heads) != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads")

        self.input_projection = nn.Linear(input_size, self.model_dim)
        # Slots are right-aligned, so these embeddings represent relative age:
        # the final slot is current and preceding slots are progressively older.
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.context_len, self.model_dim)
        )
        self.blocks = nn.ModuleList(
            GTrXLBlock(cfg) for _ in range(int(cfg.transformer_layers))
        )
        self.final_norm = nn.LayerNorm(self.model_dim)
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
        causal_mask = torch.triu(
            torch.ones(
                self.context_len,
                self.context_len,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        encoded = memory + self.position_embedding
        for block in self.blocks:
            encoded = block(encoded, padding_mask, causal_mask)
        output = self.output_projection(self.final_norm(encoded[:, -1]))
        new_state = torch.cat(
            (memory.reshape(batch, -1), lengths.to(memory.dtype).unsqueeze(1)), dim=1
        )
        return output, new_state


class GRUAttentionMemoryCore(TransformerMemoryCore):
    """GRU recurrence augmented by attention over recent GRU states.

    The attention output projection is zero-initialized, so the core starts as
    a conventional 512-unit GRU and learns an optional history-dependent
    correction without disrupting the proven recurrent pathway.
    """

    def __init__(self, cfg, input_size: int):
        ModelCore.__init__(self, cfg)
        self.hidden_size = int(cfg.gru_attention_hidden_size)
        self.attention_dim = int(cfg.gru_attention_dim)
        self.context_len = int(cfg.transformer_context)
        expected_state_size = (
            self.hidden_size + self.context_len * self.attention_dim + 1
        )
        if int(cfg.rnn_size) != expected_state_size:
            raise ValueError(
                "GRU-attention state requires "
                f"rnn_size={expected_state_size}, got {cfg.rnn_size}"
            )
        if self.attention_dim % int(cfg.transformer_heads) != 0:
            raise ValueError("gru_attention_dim must be divisible by transformer_heads")
        if self.hidden_size != input_size:
            raise ValueError(
                "GRU-attention currently preserves the matched 512-dimensional "
                f"core interface, got hidden_size={self.hidden_size}, input_size={input_size}"
            )

        self.gru = nn.GRUCell(input_size, self.hidden_size)
        self.memory_projection = nn.Linear(self.hidden_size, self.attention_dim)
        self.query_projection = nn.Linear(self.hidden_size, self.attention_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.context_len, self.attention_dim)
        )
        self.attention_norm = nn.LayerNorm(self.attention_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=int(cfg.transformer_heads),
            dropout=float(cfg.transformer_dropout),
            batch_first=True,
        )
        self.output_projection = nn.Linear(self.attention_dim, self.hidden_size)
        self.attention_gate = nn.Parameter(
            torch.tensor(float(cfg.gru_attention_gate_init))
        )
        self.core_output_size = self.hidden_size

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        # Preserve the GRU policy at initialization. Gradients immediately reach
        # this projection; once it moves away from zero they reach attention too.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def _step(self, x: torch.Tensor, state: torch.Tensor):
        batch = x.shape[0]
        hidden = state[:, : self.hidden_size]
        memory_end = self.hidden_size + self.context_len * self.attention_dim
        memory = state[:, self.hidden_size : memory_end].reshape(
            batch, self.context_len, self.attention_dim
        )
        lengths = state[:, -1].round().long().clamp(0, self.context_len)

        hidden = self.gru(x, hidden)
        token = self.memory_projection(hidden).unsqueeze(1)
        memory = torch.cat((memory[:, 1:], token), dim=1)
        lengths = torch.clamp(lengths + 1, max=self.context_len)

        positions = torch.arange(self.context_len, device=x.device).unsqueeze(0)
        padding_mask = positions < (self.context_len - lengths).unsqueeze(1)
        keys = self.attention_norm(memory + self.position_embedding)
        query = self.attention_norm(self.query_projection(hidden)).unsqueeze(1)
        attended, _ = self.attention(
            query,
            keys,
            keys,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        correction = self.output_projection(attended[:, 0])
        output = hidden + torch.sigmoid(self.attention_gate) * correction
        new_state = torch.cat(
            (
                hidden,
                memory.reshape(batch, -1),
                lengths.to(memory.dtype).unsqueeze(1),
            ),
            dim=1,
        )
        return output, new_state


class ResidualLayerNormGRUCore(ModelCore):
    """GRU with a normalized residual path from visual features.

    The recurrent state remains an ordinary 512-value GRU hidden state, so the
    only controlled change from GRU-64 is the residual normalization applied
    to the output consumed by the policy/value heads.
    """

    def __init__(self, cfg, input_size: int):
        super().__init__(cfg)
        hidden_size = int(cfg.rnn_size)
        if int(cfg.rnn_num_layers) != 1:
            raise ValueError("Residual LayerNorm GRU currently requires one recurrent layer")
        if hidden_size != input_size:
            raise ValueError(
                "Residual LayerNorm GRU requires matched input and hidden sizes, "
                f"got input_size={input_size}, hidden_size={hidden_size}"
            )
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.core_output_size = hidden_size

    def forward(self, head_output, rnn_states):
        is_sequence = not torch.is_tensor(head_output)
        gru_input = head_output if is_sequence else head_output.unsqueeze(0)
        output, new_rnn_states = self.gru(
            gru_input, rnn_states.unsqueeze(0).contiguous()
        )
        if isinstance(output, PackedSequence):
            residual_data = output.data + head_output.data
            output = PackedSequence(
                self.output_norm(residual_data),
                output.batch_sizes,
                output.sorted_indices,
                output.unsorted_indices,
            )
        else:
            output = self.output_norm(output + gru_input)
            if not is_sequence:
                output = output.squeeze(0)
        return output, new_rnn_states.squeeze(0)


def make_temporal_core(cfg, input_size: int):
    if getattr(cfg, "memory", None) == "transformer":
        return TransformerMemoryCore(cfg, input_size)
    if getattr(cfg, "memory", None) == "gtrxl":
        return GTrXLMemoryCore(cfg, input_size)
    if getattr(cfg, "memory", None) == "gru_attention":
        return GRUAttentionMemoryCore(cfg, input_size)
    if getattr(cfg, "memory", None) == "gru_residual_ln":
        return ResidualLayerNormGRUCore(cfg, input_size)
    return default_make_core_func(cfg, input_size)


def register_temporal_core():
    global_model_factory().register_model_core_factory(make_temporal_core)


# Backward-compatible name used by older scripts and checkpoints/configs.
register_transformer_core = register_temporal_core
