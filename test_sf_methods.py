"""Pure-Python tests for the canonical Sample Factory method registry."""
import pytest

from sf_methods import METHODS, method_for_training, recurrent_options


def test_catalog_separates_trainable_and_upstream_reference():
    assert method_for_training("gru_long").recurrence == 64
    assert METHODS["upstream_gru"].trainable is False
    with pytest.raises(ValueError, match="evaluation-only"):
        method_for_training("upstream_gru")


def test_standard_and_long_gru_share_architecture_settings():
    standard = recurrent_options("gru", 32, 256, 512, 128)
    long_context = recurrent_options("gru", 64, 256, 512, 128)
    assert standard["rnn_size"] == long_context["rnn_size"] == "512"
    assert standard["rnn_type"] == long_context["rnn_type"] == "gru"
    assert standard["recurrence"] == "32"
    assert long_context["recurrence"] == "64"


def test_residual_layernorm_gru_uses_matched_hidden_size():
    residual = recurrent_options("gru_residual_ln", 64, 256, 512, 128)
    assert residual == {
        "use_rnn": "True",
        "recurrence": "64",
        "rnn_size": "512",
        "rnn_num_layers": "1",
        "rnn_type": "gru",
    }


def test_completion_bonus_method_preserves_gru64_architecture():
    method = method_for_training("gru_completion_bonus")
    assert method.memory == "gru"
    assert method.recurrence == 64


def test_orthogonal_gru_preserves_matched_gru64_shape():
    method = method_for_training("gru_orthogonal")
    assert method.recurrence == 64
    assert recurrent_options("gru_orthogonal", 64, 256, 512, 128) == {
        "use_rnn": "True",
        "recurrence": "64",
        "rnn_size": "512",
        "rnn_num_layers": "1",
        "rnn_type": "gru",
    }


def test_lag_control_method_preserves_standard_gru64_architecture():
    method = method_for_training("gru_lag1")
    assert method.memory == "gru"
    assert method.recurrence == 64


def test_synchronous_method_preserves_standard_gru64_architecture():
    method = method_for_training("gru_sync")
    assert method.memory == "gru"
    assert method.recurrence == 64


def test_custom_temporal_state_sizes_are_explicit():
    transformer = recurrent_options("transformer", 32, 256, 512, 128)
    attention = recurrent_options("gru_attention", 32, 256, 512, 128)
    assert transformer["rnn_size"] == str(32 * 256 + 1)
    assert attention["rnn_size"] == str(512 + 32 * 128 + 1)
