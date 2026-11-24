# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Encoders of text.

A text is one ``str``.

:class:`Pretrained`
    A frozen sentence encoder, chosen by ``model``: the default of the
    modality ``"text"``. Its weights download on first use; it needs the
    ``encoders`` extra.
:class:`Statistics`
    Counts and shares of each text's characters and words.
:class:`Hashing`
    Each text's words or character n-grams, hashed into a fixed number of
    columns.

Typical usage example:

  columns = lazy.text.Hashing(n_features=64).fit_transform(descriptions)
"""

from __future__ import annotations

import dataclasses
import os
import string
from typing import Any

import numpy as np
from sklearn.feature_extraction import text as sklearn_text

from lazy import _typing
from lazy.encoders import _base
from lazy.encoders import _pretrained

__all__ = ["STATISTICS", "Hashing", "Pretrained", "Statistics"]

STATISTICS: tuple[str, ...] = (
    "characters",
    "words",
    "word_length",
    "distinct",
    "digits",
    "uppercase",
    "punctuation",
    "whitespace",
)
"""The columns of :class:`Statistics`, in order."""

_PUNCTUATION = frozenset(string.punctuation)


def _texts(items: list[Any]) -> list[str]:
    """The items, checked to be text.

    Raises:
        TypeError: If an item is not a ``str``.
    """
    for item in items:
        if not isinstance(item, str):
            raise TypeError(
                f"a text must be a str; got an item of type"
                f" {type(item).__name__}"
            )
    return items


class Statistics(_base.BaseEncoder):
    """Counts and shares of each text's characters and words.

    The columns are those of :data:`STATISTICS`. A word is a run of
    characters between whitespace.

    - ``characters`` and ``words``: how many the text has;
    - ``word_length``: the mean number of characters per word;
    - ``distinct``: the share of the words that differ, whatever their
      case;
    - ``digits``, ``uppercase``, ``punctuation`` and ``whitespace``: the
      share of the characters of each kind. Punctuation is ASCII's
      (:data:`string.punctuation`).

    A share or a mean of nothing (an empty text, a text of no word) is 0.

    Attributes:
        batch_size: The most texts encoded in one batch.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    modality = "text"

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        batch_size: int = 64,
        cache: bool | str | os.PathLike[str] = False,
    ) -> None:
        self.batch_size = batch_size
        self.cache = cache

    def _width(self) -> int:
        """The number of statistics."""
        return len(STATISTICS)

    def _names(self) -> tuple[str, ...]:
        """The names of the statistics."""
        return STATISTICS

    def _encode(self, items: list[Any]) -> _typing.FloatArray:
        """The statistics of each text, shape (len(items), 8)."""
        return np.array([_statistics(text) for text in _texts(items)])


def _statistics(text: str) -> list[float]:
    """The statistics of one text, as ``STATISTICS``."""
    words = text.split()
    n_characters = max(len(text), 1)
    n_words = max(len(words), 1)
    return [
        len(text),
        len(words),
        sum(map(len, words)) / n_words,
        len({word.casefold() for word in words}) / n_words,
        sum(map(str.isdigit, text)) / n_characters,
        sum(map(str.isupper, text)) / n_characters,
        sum(character in _PUNCTUATION for character in text) / n_characters,
        sum(map(str.isspace, text)) / n_characters,
    ]


class Hashing(_base.BaseEncoder):
    """Each text's words or character n-grams, hashed into columns.

    This is scikit-learn's
    :class:`~sklearn.feature_extraction.text.HashingVectorizer`, as a dense
    array: each n-gram of a lower-cased text adds or subtracts 1 in the
    column that its hash names, and each row is scaled to unit Euclidean
    length. It keeps no vocabulary, so it learns nothing from the texts.
    Two n-grams can share a column, more often with fewer columns.

    Attributes:
        n_features: The number of columns.
        analyzer: What is hashed: ``"char_wb"`` for n-grams of the
            characters inside each word, ``"char"`` for those of the whole
            text, or ``"word"`` for n-grams of words.
        ngram_range: The smallest and the largest n of the n-grams.
        batch_size: The most texts encoded in one batch.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    modality = "text"

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        n_features: int = 128,
        analyzer: str = "char_wb",
        ngram_range: tuple[int, int] = (2, 4),
        batch_size: int = 64,
        cache: bool | str | os.PathLike[str] = False,
    ) -> None:
        self.n_features = n_features
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.batch_size = batch_size
        self.cache = cache

    def _vectorizer(self) -> sklearn_text.HashingVectorizer:
        """scikit-learn's vectorizer with this encoder's parameters."""
        # Every setting that changes an embedding is pinned here, never
        # taken from the installed scikit-learn's defaults.
        return sklearn_text.HashingVectorizer(
            n_features=self.n_features,
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            lowercase=True,
            strip_accents=None,
            token_pattern=r"(?u)\b\w\w+\b",
            alternate_sign=True,
            norm="l2",
            binary=False,
            dtype=np.float64,
        )

    def _width(self) -> int:
        """The number of columns; scikit-learn checks the parameters."""
        n_features = _base.positive_integer("n_features", self.n_features)
        self._vectorizer().fit([])
        return n_features

    def _encode(self, items: list[Any]) -> _typing.FloatArray:
        """The hashed n-grams, shape (len(items), n_features)."""
        hashed = self._vectorizer().transform(_texts(items))
        return np.asarray(hashed.toarray())


@dataclasses.dataclass(frozen=True)
class _Recipe:
    """How one model turns a text into an embedding.

    Attributes:
        width: The width of the embedding.
        pooling: ``"mean"`` of the token states, or the state of the first
            token, ``"cls"``.
        prefix: Text put before each item, as the model's card asks.
        max_tokens: The most tokens of a text that the model reads.
    """

    width: int
    pooling: str
    prefix: str = ""
    max_tokens: int = 512


class Pretrained(_pretrained.PretrainedEncoder):
    """The sentence embedding of each text, from a frozen network.

    ``model`` picks one of the networks of ``recipes``, each pinned to a
    commit on the Hugging Face Hub: ``"e5-small-v2"`` (the default),
    ``"bge-small-en-v1.5"``, ``"all-MiniLM-L6-v2"`` and
    ``"gte-modernbert-base"``. The embedding is the network's pooled token
    states (their mean, or the first token's, as its card says), scaled to
    unit length, as sentence-transformers computes it. E5 reads each text
    after the prefix ``"query: "``. :doc:`/guide/encoders` lists the
    networks and gives the evidence behind the choice.

    A text longer than its model reads is cut to its first tokens.

    Attributes:
        model: The network; ``"e5-small-v2"`` by default.
        device: ``"auto"``, or a torch device such as ``"cpu"``.
        batch_size: The most texts in one forward pass.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    modality = "text"
    package = "transformers"
    recipes = {
        "e5-small-v2": _Recipe(384, "mean", prefix="query: "),
        "bge-small-en-v1.5": _Recipe(384, "cls"),
        "all-MiniLM-L6-v2": _Recipe(384, "mean", max_tokens=256),
        "gte-modernbert-base": _Recipe(768, "cls", max_tokens=8192),
    }

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        model: str = "e5-small-v2",
        device: str = "auto",
        batch_size: int = 64,
        cache: bool | str | os.PathLike[str] = True,
    ) -> None:
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.cache = cache

    def _length(self, item: Any) -> int:
        """The number of characters, which the tokens grow with."""
        return len(item) if isinstance(item, str) else 0

    def _load(self, path: os.PathLike[str], device: str) -> Any:
        """The tokenizer and the network, on ``device``."""
        import transformers  # noqa: PLC0415 - an optional, heavy extra.

        tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        model = transformers.AutoModel.from_pretrained(
            path, **_pretrained.PINNED
        )
        return tokenizer, model.to(device).eval()

    def _embed(self, network: Any, items: list[Any]) -> _typing.FloatArray:
        """The unit-length pooled token states of each text."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        tokenizer, model = network
        recipe = self._recipe()
        tokens = tokenizer(
            [recipe.prefix + text for text in _texts(items)],
            padding=True,
            truncation=True,
            max_length=recipe.max_tokens,
            return_tensors="pt",
        ).to(model.device)
        states = model(**tokens).last_hidden_state
        if recipe.pooling == "cls":
            pooled = states[:, 0]
        else:
            mask = tokens["attention_mask"].unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        unit = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return unit.float().cpu().numpy()
