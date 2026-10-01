# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Memory-bounded in-context inference for a fitted TabFM estimator.

TabFM's public ``TabFMClassifier.predict_proba`` builds every ensemble member's
view of every query row up front. On a survey-sized query set that is tens of
gigabytes of host memory, so this module drives the underlying model
directly::

    for each batch of ensemble members:
        prefill the context once     -> per-layer K/V cache
        decode all query rows in chunks of chunk_size against that cache
        free the cache before the next member batch

Peak memory is then ``member_batch x block_rows x n_features`` rather than
``n_members x n_query_rows x n_features``. The upstream package is used
unmodified; only the order of operations differs. The model computes in the
dtype its weights were loaded in. In float32 (always on the CPU) the results
agree with calling ``predict_proba`` on the whole query set at once up to
float rounding; in bfloat16 on CUDA, whose kernels round differently for
different batch shapes, densities differ by up to a few per cent of their
peak.

Raw model outputs (classification logits over ``model.max_classes``) are
handed to a callback per member batch and query block, leaving the caller to
decide what to accumulate.

This is a private module: nothing here is part of the public API.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import gc
import time
from typing import Any, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

from lazy import _typing

Consumer: TypeAlias = Callable[
    [str, int, int, int, int, npt.NDArray[np.float32]], None
]
"""``consumer(target, member_start, member_stop, row_start, row_stop, outputs)``
with ``outputs`` of shape ``(members_in_batch, rows_in_block, output_dim)``."""


class TargetLogits(TypedDict):
    """Logits for one target, as :func:`classification_logits` returns them.

    Attributes:
        mean_logits: Member-averaged logits, shape ``(rows, n_classes)``.
        members: Per-member logits, shape ``(rows, members, n_classes)``, or
            ``None`` unless ``keep_members`` was set.
    """

    mean_logits: _typing.FloatArray
    members: npt.NDArray[np.float16] | None


def _flat_configs(generator: Any) -> list[tuple[str, tuple[Any, ...]]]:
    """Members in the exact order ``prepare_ensemble_tensors`` uses."""
    flat: list[tuple[str, tuple[Any, ...]]] = []
    for norm_method, configs in generator.ensemble_configs_.items():
        flat.extend((norm_method, config) for config in configs)
    return flat


def context_tensors(
    generator: Any,
) -> tuple[
    npt.NDArray[Any],
    npt.NDArray[Any],
    npt.NDArray[Any] | None,
    npt.NDArray[Any] | None,
]:
    """Per-member context tensors (features, targets, categorical mask, depth).

    Args:
        generator: The fitted estimator's ``ensemble_generator_``.

    Returns:
        A tuple ``(features, targets, cat_masks, depths)``, each with the
        ensemble member as its leading axis.
    """
    data = generator.transform_context_only()
    Xs, ys, cat_masks, ds, _ = generator.prepare_ensemble_tensors(data)
    return Xs, ys, cat_masks, ds


class QueryViews:
    """Per-member feature views of one block of encoded query rows.

    Attributes:
        generator: The fitted estimator's ``ensemble_generator_``.
        X: The query rows after TabFM's unique filter and any cross or SVD
            features, shape ``(rows, features)``.
        flat: ``(norm_method, config)`` for every member, in ensemble order.
        max_features: The widest member view; narrower ones are zero-padded
            to it.
    """

    def __init__(self, generator: Any, X_encoded: npt.NDArray[Any]):
        """Applies the context-independent query transforms once.

        Args:
            generator: The fitted estimator's ``ensemble_generator_``.
            X_encoded: Query rows after the estimator's ``X_encoder_``, shape
                ``(rows, n_features)``.
        """
        # TabFM's own private helpers, so that the views match what
        # `predict_proba` builds exactly.
        from tabfm.src import (  # noqa: PLC0415 - optional backend, imported at use.
            classifier_and_regressor as upstream,
        )

        X = generator.unique_filter_.transform(X_encoded)
        if getattr(generator, "cross_pairs_", None):
            X = upstream._append_cross_features(  # noqa: SLF001 - see above.
                X, generator.cross_pairs_
            )
        if getattr(generator, "svd_pipeline_", None):
            X = upstream._append_svd_features(  # noqa: SLF001 - see above.
                X,
                generator.n_original_features_,
                generator.svd_pipeline_,
                is_train=False,
            )
        self.generator = generator
        self.X = X
        self.flat = _flat_configs(generator)
        self.max_features = max(len(config[0]) for _, config in self.flat)
        self._preprocessed: dict[str, npt.NDArray[Any]] = {}

    def _base(self, norm_method: str, cat_perm: Any) -> npt.NDArray[Any]:
        from tabfm.src import (  # noqa: PLC0415 - optional backend, imported at use.
            classifier_and_regressor as upstream,
        )

        preprocessor = self.generator.preprocessors_[norm_method]
        if cat_perm:
            X = self.X.copy()
            upstream._apply_categorical_permutation(X, cat_perm)  # noqa: SLF001 - TabFM's own helper, to match it exactly.
            return preprocessor.transform(X)
        if norm_method not in self._preprocessed:
            self._preprocessed[norm_method] = preprocessor.transform(self.X)
        return self._preprocessed[norm_method]

    def members(self, start: int, stop: int) -> npt.NDArray[np.float32]:
        """Stacked views for members ``[start, stop)``.

        Args:
            start: First member, inclusive.
            stop: Last member, exclusive.

        Returns:
            The views, shape ``(stop - start, rows, max_features)``.
        """
        views = []
        for norm_method, (shuffle_pattern, _, cat_perm, _) in self.flat[
            start:stop
        ]:
            cols = self._base(norm_method, cat_perm)[:, shuffle_pattern]
            if cols.shape[1] < self.max_features:
                cols = np.pad(
                    cols, ((0, 0), (0, self.max_features - cols.shape[1]))
                )
            views.append(cols)
        return np.stack(views, axis=0).astype(np.float32, copy=False)


def stream_icl(
    estimator: Any,
    model: Any,
    targets: Mapping[str, pd.DataFrame],
    consumer: Consumer,
    *,
    member_batch_size: int = 4,
    query_block_rows: int = 262_144,
    chunk_size: int = 8_192,
    keep_cache_on_device: bool = True,
    log: Callable[[str], None] | None = None,
) -> None:
    """Run every ensemble member of ``estimator`` over every target.

    Args:
        estimator: A fitted ``TabFMClassifier``.
        model: The loaded TabFM backbone, with ``prefill`` and ``decode``.
        targets: Maps a name to a DataFrame of raw query features (the
            columns the estimator was fitted on).
        consumer: Receives the outputs as they are produced; see
            :data:`Consumer`.
        member_batch_size: Ensemble members prefilled together.
        query_block_rows: Query rows whose member views are built at once,
            rounded down to a whole number of chunks (at least one).
        chunk_size: Query rows decoded per forward pass; ``0`` decodes
            every row of a target in one pass.
        keep_cache_on_device: If false, the K/V cache is round-tripped
            through host memory so the device can release the prefill's
            scratch space.
        log: Called with a progress line after each member batch, if given.
    """
    from tabfm.src.pytorch import (  # noqa: PLC0415 - optional backend, imported at use.
        model as tabfm_model,
    )
    import torch  # noqa: PLC0415 - optional backend, imported at use.

    generator = estimator.ensemble_generator_
    Xs_ctx, ys_ctx, cat_ctx, d_ctx = context_tensors(generator)
    n_members = int(Xs_ctx.shape[0])
    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"

    encoded = {
        name: estimator.X_encoder_.transform(frame.reset_index(drop=True))
        for name, frame in targets.items()
    }
    started = time.time()

    def to_dev(array: npt.NDArray[Any] | None, dtype: Any = None) -> Any:
        if array is None:
            return None
        tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device)
        return tensor if dtype is None else tensor.to(dtype)

    for m0 in range(0, n_members, member_batch_size):
        m1 = min(m0 + member_batch_size, n_members)
        with torch.no_grad():
            _, cache = model.prefill(
                to_dev(Xs_ctx[m0:m1], torch.float32),
                to_dev(ys_ctx[m0:m1], torch.float32),
                cat_mask=to_dev(None if cat_ctx is None else cat_ctx[m0:m1]),
                d=to_dev(None if d_ctx is None else d_ctx[m0:m1]),
            )
        if not keep_cache_on_device:
            cache = tabfm_model.move_cache_to_device(cache, "cpu")
            if is_cuda:
                torch.cuda.empty_cache()
            cache = tabfm_model.move_cache_to_device(cache, device)
        cat_batch = to_dev(None if cat_ctx is None else cat_ctx[m0:m1])
        d_batch = to_dev(None if d_ctx is None else d_ctx[m0:m1])

        for name, X_enc in encoded.items():
            n_rows = len(X_enc)
            chunk, block = _pass_sizes(n_rows, chunk_size, query_block_rows)
            for b0 in range(0, n_rows, block):
                b1 = min(b0 + block, n_rows)
                views = QueryViews(generator, X_enc[b0:b1]).members(m0, m1)
                for r0 in range(0, b1 - b0, chunk):
                    r1 = min(r0 + chunk, b1 - b0)
                    with torch.no_grad():
                        out = model.decode(
                            to_dev(views[:, r0:r1, :], torch.float32),
                            cache,
                            cat_mask=cat_batch,
                            d=d_batch,
                        )
                    consumer(
                        name,
                        m0,
                        m1,
                        b0 + r0,
                        b0 + r1,
                        out.float().cpu().numpy(),
                    )
                del views
        del cache, cat_batch, d_batch
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()
        if log is not None:
            elapsed = time.time() - started
            eta = elapsed / m1 * (n_members - m1)
            log(
                f"members {m1}/{n_members} | {elapsed / 60:.1f} min"
                f" | eta {eta / 60:.1f} min"
            )


def _pass_sizes(
    n_rows: int, chunk_size: int, query_block_rows: int
) -> tuple[int, int]:
    """Rows per decode pass and per block of views, as a tuple.

    Blocks hold a whole number of passes, so every pass but a target's last
    has exactly ``chunk_size`` rows, whatever ``query_block_rows`` is; with
    ``chunk_size=0`` the one pass and the one block are every row.

    Examples:
        >>> _pass_sizes(100, 8, 20)
        (8, 16)
        >>> _pass_sizes(100, 8, 5)
        (8, 8)
        >>> _pass_sizes(100, 0, 20)
        (100, 100)
    """
    chunk = chunk_size if chunk_size > 0 else max(n_rows, 1)
    return chunk, max(chunk, query_block_rows // chunk * chunk)


def class_shift_offsets(estimator: Any) -> _typing.IntArray:
    """The per-member cyclic class shift TabFM applies for ensembling.

    Args:
        estimator: A fitted ``TabFMClassifier``.

    Returns:
        One offset per ensemble member, in ensemble order.
    """
    offsets = []
    for values in estimator.ensemble_generator_.class_shift_offsets_.values():
        offsets.extend(values)
    return np.asarray(offsets, dtype=int)


def classification_logits(
    estimator: Any,
    model: Any,
    targets: Mapping[str, pd.DataFrame],
    *,
    keep_members: bool = False,
    **kwargs: Any,
) -> dict[str, TargetLogits]:
    """Class-shift-corrected logits, averaged over members (upstream default).

    Args:
        estimator: A fitted ``TabFMClassifier``.
        model: The loaded TabFM backbone.
        targets: Maps a name to a DataFrame of raw query features.
        keep_members: Also return every member's logits, as float16.
        **kwargs: Passed to :func:`stream_icl`.

    Returns:
        ``{target: {"mean_logits": (rows, n_classes), "members": (rows,
        members, n_classes) or None}}``.

    Raises:
        RuntimeError: If the classifier returns fewer logits than the
            estimator has classes.
    """
    n_classes = int(estimator.n_classes_)
    offsets = class_shift_offsets(estimator)
    sizes = {name: len(frame) for name, frame in targets.items()}
    mean_logits = {
        name: np.zeros((n, n_classes), dtype=np.float64)
        for name, n in sizes.items()
    }
    members: dict[str, npt.NDArray[np.float16] | None] = {
        name: np.empty((n, estimator.n_estimators, n_classes), dtype=np.float16)
        if keep_members
        else None
        for name, n in sizes.items()
    }

    def consume(
        name: str,
        m0: int,
        m1: int,
        r0: int,
        r1: int,
        values: npt.NDArray[np.float32],
    ) -> None:
        if values.shape[-1] < n_classes:
            raise RuntimeError(
                f"classifier returned {values.shape[-1]} logits for"
                f" {n_classes} classes"
            )
        values = values[..., :n_classes]
        kept = members[name]
        for local in range(m1 - m0):
            offset = offsets[m0 + local]
            logits = np.concatenate(
                [values[local, :, offset:], values[local, :, :offset]], axis=-1
            )
            mean_logits[name][r0:r1] += logits / estimator.n_estimators
            if kept is not None:
                kept[r0:r1, m0 + local] = logits

    stream_icl(estimator, model, targets, consume, **kwargs)
    return {
        name: {"mean_logits": mean_logits[name], "members": members[name]}
        for name in sizes
    }


def softmax(
    logits: _typing.FloatArray, temperature: float = 0.9
) -> _typing.FloatArray:
    """Temperature-scaled softmax over the last axis.

    Args:
        logits: Logits, any shape; classes on the last axis.
        temperature: Divides the logits before the softmax.

    Returns:
        Probabilities of the same shape, summing to one over the last axis.

    Examples:
        >>> softmax(np.zeros((1, 4))).round(3).tolist()
        [[0.25, 0.25, 0.25, 0.25]]
    """
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    weights = np.exp(scaled)
    return weights / weights.sum(axis=-1, keepdims=True)


def streaming_available() -> bool:
    """Whether the installed ``tabfm`` exposes the KV-cache API needed here.

    The PyPI release of ``tabfm`` 1.0.0/1.0.1 has no ``prefill`` / ``decode``
    and no cache helpers; they arrived later, in the repository. Everything
    here therefore has to be optional, with the upstream ``predict_proba`` as
    the fallback (see :class:`lazy.models.tabfm.TabFMHistogram`).
    """
    try:
        from tabfm.src.pytorch import (  # noqa: PLC0415 - optional backend, imported at use.
            model as tabfm_model,
        )
    except Exception:  # noqa: BLE001 - any failure to import means no streaming.
        return False
    return (
        hasattr(tabfm_model, "move_cache_to_device")
        and hasattr(tabfm_model, "TabFM")
        and hasattr(tabfm_model.TabFM, "prefill")
        and hasattr(tabfm_model.TabFM, "decode")
    )
