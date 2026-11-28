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
import re
import string
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction import text as sklearn_text

from lazy import _steps
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

# Why no gradient crosses an encoder of text.
_DISCRETE = "a text is a sequence of characters, which have no gradient"


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
    barrier_reason = _DISCRETE

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
    barrier_reason = _DISCRETE

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
        pooling: ``"mean"`` of the token states, the state of the first
            token, ``"cls"``, or of the last, ``"last"``.
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
    ``"bge-small-en-v1.5"``, ``"all-MiniLM-L6-v2"``,
    ``"gte-modernbert-base"`` and ``"Qwen3-Embedding-0.6B"``. The embedding
    is the network's pooled token states (their mean, the first token's or
    the last token's, as its card says), scaled to unit length, as
    sentence-transformers computes it. E5 reads each text after the prefix
    ``"query: "``; Qwen3-Embedding reads it with no instruction, as its card
    reads a document. :doc:`/guide/encoders` lists the networks and gives
    the evidence behind the choice.

    A text longer than its model reads is cut to its first tokens.

    A forward pass holds every text of its batch padded to the longest, so
    its memory grows with the texts' count times that length. A batch
    whose padded tokens would exceed ``max_batch_tokens`` is split into
    passes that do not, so that a few long
    texts do not exhaust a GPU. A text's embedding is the same either way,
    up to float rounding.

    A token has no gradient, but its input embedding, the vector the
    network looks it up as, does. The encoder's ``forward`` takes the
    input embeddings of the tokens that the network reads of each text,
    shape (n, longest, hidden), padded on the right with ``NaN``, and
    gives the embeddings on torch's autograd graph, without the cache. A
    gradient method of :mod:`lazy.interp` then attributes to each token,
    summed over its vector. The baseline of a text is the same tokens with
    each of the text's own replaced by the padding token, so that the
    special tokens and the prefix, which the baseline keeps, get no
    attribution. :meth:`token_spans` locates each token in the text, and
    :meth:`word_values` sums the attributions to a text's tokens by word.

    Attributes:
        model: The network; ``"e5-small-v2"`` by default.
        max_batch_tokens: The most tokens, padding included, in one forward
            pass; a single text longer than that is a pass of its own. The
            default, 32,768, is 64 texts of 512 tokens, so a larger
            ``batch_size`` needs a larger budget to take effect on long
            texts. The cache does not key it: a split moves an embedding
            by float rounding at most.
        device: ``"auto"``, or a torch device such as ``"cpu"``.
        batch_size: The most texts in one forward pass, which
            ``max_batch_tokens`` may lower.
        cache: Whether, or where, to file the embeddings on disk; see
            :class:`~lazy.encoders.BaseEncoder`.
    """

    modality = "text"
    package = "transformers"
    gradient_kind = "differentiable"
    pads_items = "right"
    element_ndim = 1
    # The split of a batch moves an embedding by float rounding at most.
    _UNKEYED = (*_base.BASE_PARAMS, "device", "max_batch_tokens")
    recipes = {
        "e5-small-v2": _Recipe(384, "mean", prefix="query: "),
        "bge-small-en-v1.5": _Recipe(384, "cls"),
        "all-MiniLM-L6-v2": _Recipe(384, "mean", max_tokens=256),
        "gte-modernbert-base": _Recipe(768, "cls", max_tokens=8192),
        # The card's own maximum; its tokenizer ends each text with
        # <|endoftext|>, whose state is the embedding.
        "Qwen3-Embedding-0.6B": _Recipe(1024, "last", max_tokens=8192),
    }

    def __init__(  # noqa: D107 - arguments documented on the class.
        self,
        *,
        model: str = "e5-small-v2",
        max_batch_tokens: int = 32_768,
        device: str = "auto",
        batch_size: int = 64,
        cache: bool | str | os.PathLike[str] = True,
    ) -> None:
        self.model = model
        self.max_batch_tokens = max_batch_tokens
        self.device = device
        self.batch_size = batch_size
        self.cache = cache

    def _width(self) -> int:
        """The width of the model's embedding; checks the parameters."""
        _base.positive_integer("max_batch_tokens", self.max_batch_tokens)
        return super()._width()

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
        tokenizer, model = network
        recipe = self._recipe()
        tokens = tokenizer(
            [recipe.prefix + text for text in _texts(items)],
            padding=True,
            truncation=True,
            max_length=recipe.max_tokens,
            return_tensors="pt",
        )
        embeddings = self._embedded(
            model, tokens["input_ids"], tokens["attention_mask"]
        )
        return embeddings.numpy()

    def forward(self, items: Any) -> Any:
        """The embeddings of texts' input embeddings, on the autograd graph.

        It runs the network on its ``device``, without the cache, in
        passes of at most ``batch_size`` texts, which ``max_batch_tokens``
        may split. The texts are taken in order of their number of tokens,
        as ``transform`` takes them in order of length, so that a pass pads
        little.

        Args:
            items: The input embeddings of each text's tokens, as
                :meth:`_array` gives them, padded on the right with
                ``NaN``: a float torch tensor of shape (n, length, hidden).

        Returns:
            What ``transform`` gives for the texts, shape (n, width), of
            the type and on the device of ``items``.

        Raises:
            lazy._steps.BarrierError: If ``items`` has another shape, a
                text has no token, or a token's vector is partly ``NaN`` or
                follows padding.
        """
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        _, model = self._network()
        present = self._stack(items, model)
        # Selected, so that no NaN of the padding enters the graph.
        vectors = torch.where(present[..., None], items, 0.0)
        mask = present.long()
        order = torch.argsort(mask.sum(dim=1), stable=True)
        embeddings = self._in_passes(
            order,
            lambda rows: self._embedded(model, vectors[rows], mask[rows]),
        )
        # Back in the order of the items.
        return embeddings[torch.argsort(order)].to(items)

    def _stack(self, items: Any, model: Any) -> Any:
        """Which positions of stacked input embeddings hold a token.

        Args:
            items: The input embeddings, a float torch tensor that should be
                of shape (n, length, hidden), each text's tokens first and
                then its padding, ``NaN``.
            model: The network, whose input embeddings are ``hidden`` long.

        Returns:
            A boolean tensor of shape (n, length).

        Raises:
            lazy._steps.BarrierError: If ``items`` is not of that shape, a
                text has no token, or a token's vector is partly ``NaN`` or
                follows padding.
        """
        hidden = model.get_input_embeddings().embedding_dim
        if items.ndim == 3 and items.shape[2] == hidden:
            finite = items.isfinite()
            present = finite.all(dim=2)
            if (
                bool(present[:, 0].all())
                and bool((finite.any(dim=2) == present).all())
                and not bool((present[:, 1:] > present[:, :-1]).any())
            ):
                return present
        raise _steps.BarrierError(
            f"{type(self).__name__} has a forward only for the input"
            f" embeddings of texts' tokens, shape (n, length, {hidden}),"
            " each text's tokens first and then its padding, NaN"
        )

    def _embedded(self, model: Any, inputs: Any, mask: Any) -> Any:
        """The unit-length pooled token states of texts, on the graph.

        Args:
            model: The network.
            inputs: The token ids, an integer torch tensor of shape (n,
                length), which it looks up pass by pass, or their input
                embeddings, a float one of shape (n, length, hidden).
            mask: Which positions hold a token, an integer tensor of shape
                (n, length); the padding may be on either side.

        Returns:
            The embeddings, a float32 tensor on the CPU, shape (n, width).
        """
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        pooled = []
        for rows in _token_batches(
            mask.sum(dim=1).tolist(), self.max_batch_tokens
        ):
            # The columns that hold a token of these texts: padding beyond
            # their longest is cut, on either side.
            columns = mask[rows].any(dim=0)
            kept = mask[rows][:, columns].to(model.device)
            vectors = inputs[rows][:, columns]
            if vectors.is_floating_point():
                # In float32 first: Apple's MPS has no float64.
                vectors = vectors.float().to(model.device)
            else:
                vectors = model.get_input_embeddings()(vectors.to(model.device))
            states = model(
                inputs_embeds=vectors, attention_mask=kept
            ).last_hidden_state
            # Pooled at once, so that no pass's states outlive it.
            pooled.append(_pooled(states, kept, self._recipe().pooling))
        return torch.cat(pooled)

    def _array(self, item: Any) -> _typing.FloatArray:
        """The input embeddings of the tokens the network reads of a text.

        Returns:
            The vectors, float32 as the network reads them, shape
            (n_tokens, hidden): its special tokens, the prefix and its own
            tokens, in the order the network reads them.

        Raises:
            TypeError: If ``item`` is not a ``str``.
        """
        ids, _ = self._tokens(item)
        return self._looked_up(ids)

    def _baseline(  # noqa: GS030 - scikit-learn's X.
        self, X: Any, stacked: _typing.FloatArray
    ) -> _typing.FloatArray:
        """Texts of padding tokens, with their special and prefix tokens.

        Args:
            X: The texts, a sequence or a pandas Series.
            stacked: Their tokens' input embeddings, from :meth:`_stacked`.

        Returns:
            ``stacked``, with the padding token's input embedding in place
            of each of a text's own tokens.
        """
        tokenizer, _ = self._network()
        own = np.zeros(stacked.shape[:2], dtype=bool)
        for row, text in enumerate(X):
            spans = self._tokens(text)[1]
            own[row, : len(spans)] = spans[:, 0] >= 0
        baselines = stacked.copy()
        baselines[own] = self._looked_up(np.array([tokenizer.pad_token_id]))
        return baselines

    def _tokens(self, text: Any) -> tuple[_typing.IntArray, _typing.IntArray]:
        """The tokens the network reads of one text, and where they are.

        Args:
            text: One text.

        Returns:
            A tuple (ids, spans): the token ids, shape (n_tokens,), and as
            :meth:`token_spans` gives them.

        Raises:
            TypeError: If ``text`` is not a ``str``.
        """
        tokenizer, _ = self._network()
        recipe = self._recipe()
        tokens = tokenizer(
            recipe.prefix + _texts([text])[0],
            truncation=True,
            max_length=recipe.max_tokens,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        spans = np.array(tokens["offset_mapping"]).reshape(-1, 2)
        spans -= len(recipe.prefix)
        # A special token spans no character, and the prefix's end before
        # the text.
        own = (np.array(tokens["special_tokens_mask"]) == 0) & (spans[:, 1] > 0)
        spans = np.where(own[:, None], np.maximum(spans, 0), -1)
        return np.array(tokens["input_ids"]), spans

    def _looked_up(self, ids: _typing.IntArray) -> Any:
        """The input embeddings of token ids, float32, shape (n, hidden)."""
        import torch  # noqa: PLC0415 - an optional, heavy extra.

        _, model = self._network()
        with torch.no_grad():
            vectors = model.get_input_embeddings()(
                torch.as_tensor(ids, device=model.device)
            )
        return vectors.float().cpu().numpy()

    def token_spans(self, text: str) -> _typing.IntArray:
        """Where each token that the network reads of a text is in the text.

        The tokens are in the order of ``forward``'s input and of the
        attributions to a text in :attr:`lazy.interp.Explanation.item_values`.
        It loads the network with its tokenizer, once per process and
        device, and downloads them on first use.

        Args:
            text: One text.

        Returns:
            Each token's first character and the one after its last, as
            positions in ``text``, shape (n_tokens, 2); -1 for a token
            that is not the text's own, a special token or one of the
            prefix.

        Raises:
            TypeError: If ``text`` is not a ``str``.
            UnknownNameError: If ``model`` is not one of ``recipes``.
            ImportError: If the extra of ``model`` is not installed.
            RuntimeError: If ``device`` names an accelerator that is not
                there.
        """
        _texts([text])
        self._import()
        return self._tokens(text)[1]

    def word_values(
        self, text: str, values: npt.ArrayLike
    ) -> tuple[list[str], _typing.FloatArray]:
        """The attributions to a text's tokens, summed over each word.

        A word is a run of characters between whitespace. A token belongs
        to the word of its first character that is not whitespace, and a
        token of whitespace alone to the word before it (the first word if
        none is). Special and prefix tokens belong to none: their
        attributions are 0.

        Args:
            text: One text.
            values: The attributions to its tokens, in the order of
                :meth:`token_spans`, shape (n,) with ``n`` at least its
                number of tokens: one statistic of one row of
                :attr:`lazy.interp.Explanation.item_values`, whose
                positions beyond the text's tokens are padding.

        Returns:
            The words, and the sum of the values of each word's tokens,
            shape (n_words,), as a tuple (words, sums).

        Raises:
            TypeError: If ``text`` is not a ``str``.
            UnknownNameError: If ``model`` is not one of ``recipes``.
            ImportError: If the extra of ``model`` is not installed.
            RuntimeError: If ``device`` names an accelerator that is not
                there.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(
                f"values must be 1-D, one per token: got shape {values.shape}"
            )
        spans = self.token_spans(text)
        if len(values) < len(spans):
            raise ValueError(
                f"values must hold one per token of the text, at least"
                f" {len(spans)}: got {len(values)}"
            )
        words = np.array(
            [match.span() for match in re.finditer(r"\S+", text)], dtype=int
        ).reshape(-1, 2)
        kept = spans[:, 0] >= 0
        if not len(words) or not kept.any():
            return [], np.zeros(len(words))
        start, end = spans[kept].T
        # A token's first character that is not whitespace, if it has
        # one, else its last; then the word that starts at or before it.
        filled = np.array([not character.isspace() for character in text])
        visible = np.flatnonzero(filled)
        after = visible[
            np.minimum(np.searchsorted(visible, start), len(visible) - 1)
        ]
        anchor = np.where((after >= start) & (after < end), after, end - 1)
        word = np.searchsorted(words[:, 0], anchor, side="right") - 1
        sums = np.bincount(
            np.maximum(word, 0),
            weights=values[: len(spans)][kept],
            minlength=len(words),
        )
        return [text[a:b] for a, b in words], sums


def _pooled(states: Any, mask: Any, pooling: str) -> Any:
    """The unit-length pooled states of a batch.

    Args:
        states: The token states, a torch tensor of shape (n, length,
            width).
        mask: Which positions hold a token, shape (n, length).
        pooling: ``"cls"``, ``"last"`` or ``"mean"``, as the recipe says.

    Returns:
        The embeddings, float32 on the CPU, shape (n, width).
    """
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    if pooling == "cls":
        pooled = states[:, 0]
    elif pooling == "last":
        # The last position the mask keeps, on either side of padding.
        positions = torch.arange(mask.shape[1], device=mask.device)
        last = (mask * positions).argmax(dim=1)
        rows = torch.arange(len(states), device=mask.device)
        pooled = states[rows, last]
    else:
        weights = mask.unsqueeze(-1).to(states.dtype)
        pooled = (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
    unit = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return unit.float().cpu()


def _token_batches(lengths: list[int], budget: int) -> list[slice]:
    """The texts of each forward pass, under a budget of padded tokens.

    The texts are taken in their order, which the callers sort by length,
    and a pass ends before its texts, padded to its longest, would exceed
    the budget; a text longer than the budget is a pass of its own. Texts
    in order of length that fit together are therefore one pass.

    Args:
        lengths: Each text's number of tokens.
        budget: The most padded tokens in one pass.

    Returns:
        The positions of the texts of each pass; every text is in one.
    """
    batches = []
    start = longest = 0
    for position, length in enumerate(lengths):
        if (
            position > start
            and (position - start + 1) * max(longest, length) > budget
        ):
            batches.append(slice(start, position))
            start, longest = position, 0
        longest = max(longest, length)
    return [*batches, slice(start, len(lengths))]
