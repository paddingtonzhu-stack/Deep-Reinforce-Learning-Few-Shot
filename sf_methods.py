"""Canonical names and recurrent settings for Sample Factory corridor methods.

Sample Factory calls its optimizer APPO (asynchronous PPO).  The short method
names below describe only the policy architecture; all trainable methods use
the same matched APPO configuration from :mod:`sf_train_corridor`.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Method:
    name: str
    display_name: str
    memory: str
    trainable: bool
    description: str
    recurrence: int | None = None


METHODS = {
    "cnn": Method(
        "cnn",
        "CNN + APPO",
        "cnn",
        True,
        "Visual encoder with no temporal memory (the no-memory control).",
        1,
    ),
    "gru": Method(
        "gru",
        "CNN + GRU + APPO",
        "gru",
        True,
        "Standard 512-unit GRU with rollout/recurrence 32.",
        32,
    ),
    "gru_long": Method(
        "gru_long",
        "CNN + long-context GRU + APPO",
        "gru",
        True,
        "The same 512-unit GRU trained with rollout/recurrence 64.",
        64,
    ),
    "gru_residual_ln": Method(
        "gru_residual_ln",
        "CNN + residual LayerNorm GRU + APPO",
        "gru_residual_ln",
        True,
        "A 512-unit GRU with an input residual and LayerNorm, trained at recurrence 64.",
        64,
    ),
    "gru_completion_bonus": Method(
        "gru_completion_bonus",
        "CNN + GRU-64 + completion-bonus APPO",
        "gru",
        True,
        "Standard GRU-64 trained with a +10 terminal completion bonus.",
        64,
    ),
    "gru_skill_curriculum": Method(
        "gru_skill_curriculum",
        "CNN + GRU-64 + skill curriculum APPO",
        "gru",
        True,
        "GRU-64 pretraining at Doom skill 1 before standard skill-5 training.",
        64,
    ),
    "gru_orthogonal": Method(
        "gru_orthogonal",
        "CNN + orthogonally initialized GRU-64 + APPO",
        "gru_orthogonal",
        True,
        "GRU-64 with gate-wise orthogonal recurrent weights, Xavier input weights, and zero biases.",
        64,
    ),
    "gru_attention": Method(
        "gru_attention",
        "CNN + GRU + optional attention + APPO",
        "gru_attention",
        True,
        "Our gated attention residual on top of a 512-unit GRU.",
        32,
    ),
    "transformer": Method(
        "transformer",
        "CNN + Transformer + APPO",
        "transformer",
        True,
        "Finite-window causal Transformer temporal core.",
        32,
    ),
    "gtrxl": Method(
        "gtrxl",
        "CNN + GTrXL + APPO",
        "gtrxl",
        True,
        "Gated Transformer-XL-style experimental temporal core.",
        32,
    ),
    "upstream_gru": Method(
        "upstream_gru",
        "Upstream Hugging Face CNN + GRU + APPO",
        "gru",
        False,
        "Published MattStammers checkpoint; evaluation-only reference.",
        32,
    ),
}


TRAINABLE_METHODS = tuple(name for name, method in METHODS.items() if method.trainable)


def method_for_training(name: str) -> Method:
    method = METHODS[name]
    if not method.trainable:
        raise ValueError(f"{name} is an evaluation-only reference, not a training method")
    return method


def recurrent_options(memory, context, transformer_dim, gru_hidden, attention_dim):
    """Return Sample Factory recurrent-state settings for one memory core."""
    return {
        "cnn": {"use_rnn": "False", "recurrence": "1"},
        "gru": {
            "use_rnn": "True",
            "recurrence": str(context),
            "rnn_size": "512",
            "rnn_type": "gru",
        },
        "gru_residual_ln": {
            "use_rnn": "True",
            "recurrence": str(context),
            "rnn_size": "512",
            "rnn_num_layers": "1",
            # Sample Factory still uses this flag for recurrent trajectory handling.
            "rnn_type": "gru",
        },
        "gru_orthogonal": {
            "use_rnn": "True",
            "recurrence": str(context),
            "rnn_size": "512",
            "rnn_num_layers": "1",
            "rnn_type": "gru",
        },
        "transformer": {
            "use_rnn": "True",
            "recurrence": str(context),
            "rnn_size": str(context * transformer_dim + 1),
            "rnn_num_layers": "1",
            "rnn_type": "gru",
        },
        "gtrxl": {
            "use_rnn": "True",
            "recurrence": str(context),
            "rnn_size": str(context * transformer_dim + 1),
            "rnn_num_layers": "1",
            # This flag enables Sample Factory recurrent trajectory handling;
            # our registered core factory supplies the actual GTrXL module.
            "rnn_type": "gru",
        },
        "gru_attention": {
            "use_rnn": "True",
            "recurrence": str(context),
            "rnn_size": str(gru_hidden + context * attention_dim + 1),
            "rnn_num_layers": "1",
            "rnn_type": "gru",
        },
    }[memory]


def format_method_catalog():
    rows = ["Available methods (all trainable methods use matched Sample Factory APPO):"]
    for method in METHODS.values():
        role = "train/evaluate" if method.trainable else "evaluate only"
        rows.append(f"  {method.name:16} {method.display_name} [{role}]\n{'':19}{method.description}")
    return "\n".join(rows)
