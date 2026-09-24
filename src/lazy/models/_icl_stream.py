# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Memory-bounded in-context inference for a fitted TabFM estimator.

TabFM's public ``TabFMClassifier.predict_proba`` builds every ensemble member's
view of every query row up front. On a survey-sized query set that is tens of
gigabytes of host memory, so this module drives the underlying model directly::

    for each batch of ensemble members:
        prefill the context once     -> per-layer K/V cache
        decode all query rows in fixed-size chunks against that cache
        free the cache before the next member batch

Peak memory is then ``member_batch x block_rows x n_features`` rather than
``n_members x n_query_rows x n_features``. The upstream package is used
unmodified; only the order of operations differs, so results are identical to
calling ``predict_proba`` on the whole query set at once.

Raw model outputs (classification logits over ``model.max_classes``) are handed
to a callback per member batch and query block, leaving the caller to decide
what to accumulate.

This is a private module: nothing here is part of the public API.
"""

from __future__ import annotations

from collections.abc import Callable
import gc
import time
from typing import Any

import numpy as np

Consumer = Callable[[str, int, int, int, int, np.ndarray], None]
"""``consumer(target, member_start, member_stop, row_start, row_stop, outputs)``
with ``outputs`` of shape ``(members_in_batch, rows_in_block, output_dim)``."""


def _flat_configs(generator) -> list[tuple[str, tuple]]:
    """Members in the exact order ``prepare_ensemble_tensors`` uses."""

    flat = []
    for norm_method, configs in generator.ensemble_configs_.items():
        flat.extend((norm_method, config) for config in configs)
    return flat


def context_tensors(generator):
    """Per-member context tensors (features, targets, categorical mask, depth)."""

    data = generator.transform_context_only()
    Xs, ys, cat_masks, ds, _ = generator.prepare_ensemble_tensors(data)
    return Xs, ys, cat_masks, ds


class QueryViews:
    """Per-member feature views of one block of encoded query rows."""

    def __init__(self, generator, X_encoded: np.ndarray):
        from tabfm.src.classifier_and_regressor import _append_cross_features
        from tabfm.src.classifier_and_regressor import _append_svd_features

        X = generator.unique_filter_.transform(X_encoded)
        if getattr(generator, "cross_pairs_", None):
            X = _append_cross_features(X, generator.cross_pairs_)
        if getattr(generator, "svd_pipeline_", None):
            X = _append_svd_features(
                X,
                generator.n_original_features_,
                generator.svd_pipeline_,
                is_train=False,
            )
        self.generator = generator
        self.X = X
        self.flat = _flat_configs(generator)
        self.max_features = max(len(config[0]) for _, config in self.flat)
        self._preprocessed: dict[str, np.ndarray] = {}

    def _base(self, norm_method: str, cat_perm) -> np.ndarray:
        from tabfm.src.classifier_and_regressor import (
            _apply_categorical_permutation,
        )

        preprocessor = self.generator.preprocessors_[norm_method]
        if cat_perm:
            X = self.X.copy()
            _apply_categorical_permutation(X, cat_perm)
            return preprocessor.transform(X)
        if norm_method not in self._preprocessed:
            self._preprocessed[norm_method] = preprocessor.transform(self.X)
        return self._preprocessed[norm_method]

    def members(self, start: int, stop: int) -> np.ndarray:
        """Stacked ``(members, rows, features)`` views for members ``[start, stop)``."""
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
    estimator,
    model,
    targets: dict[str, Any],
    consumer: Consumer,
    *,
    member_batch_size: int = 4,
    query_block_rows: int = 262_144,
    decode_chunk_rows: int = 16_384,
    keep_cache_on_device: bool = True,
    log: Callable[[str], None] | None = None,
) -> None:
    """Run every ensemble member of ``estimator`` over every target.

    ``targets`` maps a name to a DataFrame of raw query features (the columns
    the estimator was fitted on). Outputs go to ``consumer`` as produced.
    """

    from tabfm.src.pytorch.model import move_cache_to_device
    import torch

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

    def to_dev(array, dtype=None):
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
            cache = move_cache_to_device(cache, "cpu")
            if is_cuda:
                torch.cuda.empty_cache()
            cache = move_cache_to_device(cache, device)
        cat_batch = to_dev(None if cat_ctx is None else cat_ctx[m0:m1])
        d_batch = to_dev(None if d_ctx is None else d_ctx[m0:m1])

        for name, X_enc in encoded.items():
            n_rows = len(X_enc)
            for b0 in range(0, n_rows, query_block_rows):
                b1 = min(b0 + query_block_rows, n_rows)
                views = QueryViews(generator, X_enc[b0:b1]).members(m0, m1)
                for r0 in range(0, b1 - b0, decode_chunk_rows):
                    r1 = min(r0 + decode_chunk_rows, b1 - b0)
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
                f"members {m1}/{n_members} | {elapsed / 60:.1f} min | eta {eta / 60:.1f} min"
            )


def class_shift_offsets(estimator) -> np.ndarray:
    """The per-member cyclic class shift TabFM applies for ensembling."""

    offsets = []
    for values in estimator.ensemble_generator_.class_shift_offsets_.values():
        offsets.extend(values)
    return np.asarray(offsets, dtype=int)


def classification_logits(
    estimator, model, targets, *, keep_members: bool = False, **kwargs
) -> dict[str, dict[str, np.ndarray]]:
    """Class-shift-corrected logits, averaged over members (upstream default).

    Returns ``{target: {"mean_logits": (rows, n_classes), "members": (rows, members, n_classes)
    or None}}``.
    """

    n_classes = int(estimator.n_classes_)
    offsets = class_shift_offsets(estimator)
    sizes = {name: len(frame) for name, frame in targets.items()}
    out = {
        name: {
            "mean_logits": np.zeros((n, n_classes), dtype=np.float64),
            "members": np.empty(
                (n, estimator.n_estimators, n_classes), dtype=np.float16
            )
            if keep_members
            else None,
        }
        for name, n in sizes.items()
    }

    def consume(name, m0, m1, r0, r1, values):
        if values.shape[-1] < n_classes:
            raise RuntimeError(
                f"classifier returned {values.shape[-1]} logits for {n_classes} classes"
            )
        values = values[..., :n_classes]
        for local in range(m1 - m0):
            offset = offsets[m0 + local]
            logits = np.concatenate(
                [values[local, :, offset:], values[local, :, :offset]], axis=-1
            )
            out[name]["mean_logits"][r0:r1] += logits / estimator.n_estimators
            if keep_members:
                out[name]["members"][r0:r1, m0 + local] = logits

    stream_icl(estimator, model, targets, consume, **kwargs)
    return out


def softmax(logits: np.ndarray, temperature: float = 0.9) -> np.ndarray:
    """Temperature-scaled softmax over the last axis.

    >>> softmax(np.zeros((1, 4))).round(3).tolist()
    [[0.25, 0.25, 0.25, 0.25]]
    """

    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    weights = np.exp(scaled)
    return weights / weights.sum(axis=-1, keepdims=True)


def streaming_available() -> bool:
    """Whether the installed ``tabfm`` exposes the KV-cache API this module needs.

    The PyPI release of ``tabfm`` 1.0.0/1.0.1 has no ``prefill`` / ``decode``
    and no cache helpers; they arrived later, in the repository. Everything
    here therefore has to be optional, with the upstream ``predict_proba`` as
    the fallback (see :class:`lazy.models.tabfm.TabFMHistogram`).
    """

    try:
        from tabfm.src.pytorch.model import move_cache_to_device
        from tabfm.src.pytorch.model import TabFM
    except Exception:
        return False
    return hasattr(TabFM, "prefill") and hasattr(TabFM, "decode")
