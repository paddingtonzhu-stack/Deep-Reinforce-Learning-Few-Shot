"""Compatibility imports for the former temporal-core module name.

New code should import :mod:`sf_temporal_cores`. Keeping this shim preserves
older scripts and external imports without duplicating implementations.
"""
from sf_temporal_cores import (  # noqa: F401
    GTrXLMemoryCore,
    GRUAttentionMemoryCore,
    GRUGatingUnit,
    TransformerMemoryCore,
    make_temporal_core,
    register_temporal_core,
    register_transformer_core,
)
