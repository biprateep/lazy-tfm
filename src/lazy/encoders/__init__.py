# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Encoders: from items that are not table cells to table columns.

An encoder turns each item of a column -- an image, a text, a time series
-- into a vector of fixed length, so that a tabular foundation model can
take the column as features. :class:`BaseEncoder` is the contract every
encoder meets, a scikit-learn transformer; :func:`register`,
:func:`list_encoders` and :func:`get_encoder` file encoders by name;
:func:`clear_embedding_cache` empties the cache of embeddings on disk;
:data:`CHECKPOINTS` pins the weights of the pretrained encoders' models;
and :class:`Tabularizer` encodes the columns of items of a table.
"""

# This file's imports re-export the public API by name; that is its purpose.
# ruff: noqa: GS001

from lazy.encoders._base import BaseEncoder
from lazy.encoders._base import MODALITIES
from lazy.encoders._cache import clear_embedding_cache
from lazy.encoders.registry import DEFAULT_ENCODERS
from lazy.encoders.registry import ENCODERS
from lazy.encoders.registry import get_encoder
from lazy.encoders.registry import list_encoders
from lazy.encoders.registry import register
from lazy.encoders.tabularizer import Tabularizer
from lazy.models._hub import DEFAULT_ENCODER_MODELS as DEFAULT_MODELS
from lazy.models._hub import ENCODER_CHECKPOINTS as CHECKPOINTS

__all__ = [
    "CHECKPOINTS",
    "DEFAULT_ENCODERS",
    "DEFAULT_MODELS",
    "ENCODERS",
    "MODALITIES",
    "BaseEncoder",
    "Tabularizer",
    "clear_embedding_cache",
    "get_encoder",
    "list_encoders",
    "register",
]
