"""Correctness tests for the GTrXL temporal core."""
import sys
from types import ModuleType
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# The core itself only needs this small interface. Lightweight stubs keep the
# numerical unit tests runnable on development machines without Sample Factory;
# integration tests on the Linux training host exercise the real package.
try:
    import sample_factory  # noqa: F401
except ModuleNotFoundError:
    sample_factory = ModuleType("sample_factory")
    model = ModuleType("sample_factory.model")
    core_module = ModuleType("sample_factory.model.core")
    algo = ModuleType("sample_factory.algo")
    utils = ModuleType("sample_factory.algo.utils")
    context = ModuleType("sample_factory.algo.utils.context")

    class ModelCore(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg

    core_module.ModelCore = ModelCore
    core_module.default_make_core_func = lambda cfg, input_size: None
    context.global_model_factory = lambda: None
    sys.modules.update(
        {
            "sample_factory": sample_factory,
            "sample_factory.model": model,
            "sample_factory.model.core": core_module,
            "sample_factory.algo": algo,
            "sample_factory.algo.utils": utils,
            "sample_factory.algo.utils.context": context,
        }
    )

from sf_transformer_core import GTrXLMemoryCore


def make_cfg():
    return SimpleNamespace(
        rnn_size=4 * 8 + 1,
        transformer_dim=8,
        transformer_context=4,
        transformer_heads=2,
        transformer_ff_dim=16,
        transformer_dropout=0.0,
        transformer_layers=2,
        gtrxl_identity_bias=2.0,
    )


def test_packed_matches_stepwise_execution():
    torch.manual_seed(7)
    core = GTrXLMemoryCore(make_cfg(), input_size=12).eval()
    lengths = torch.tensor([5, 3, 2])
    inputs = torch.randn(3, 5, 12)
    initial_state = torch.zeros(3, 33)

    expected_output = torch.zeros_like(inputs)
    expected_state = initial_state.clone()
    with torch.no_grad():
        for timestep in range(inputs.shape[1]):
            active = lengths > timestep
            output, next_state = core(inputs[active, timestep], expected_state[active])
            expected_output[active, timestep] = output
            expected_state[active] = next_state

        packed = pack_padded_sequence(
            inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, final_state = core(packed, initial_state)
        actual_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=inputs.shape[1]
        )

    torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-6)


def test_gradients_reach_attention_and_gates():
    torch.manual_seed(11)
    core = GTrXLMemoryCore(make_cfg(), input_size=12).train()
    inputs = torch.randn(2, 4, 12)
    lengths = torch.tensor([4, 3])
    packed = pack_padded_sequence(
        inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    output, _ = core(packed, torch.zeros(2, 33))
    output.data.square().mean().backward()

    assert core.blocks[0].attention.in_proj_weight.grad is not None
    assert core.blocks[0].attention.in_proj_weight.grad.abs().sum() > 0
    assert core.blocks[0].attention_gate.reset_update.weight.grad is not None
    assert core.blocks[0].attention_gate.reset_update.weight.grad.abs().sum() > 0
