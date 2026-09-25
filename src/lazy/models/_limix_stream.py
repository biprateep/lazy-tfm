# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Running LimiX-2's network: once over the context, then query by query.

LimiX-2's transformer alternates, in every layer, attention across a row's
feature tokens, an MLP, and attention across rows ("sample attention"). Only
the last couples rows, and in it every row -- context or query -- attends to
the *context* rows alone (upstream: ``x_kv = x[:, :eval_pos]``). So the
context's input to each layer's sample attention is all a query needs from
the context, and it does not depend on the queries. :func:`prefill` runs the
context through the network once and keeps those inputs; :func:`decode` runs
only the query rows, attending to the cache. The answers equal upstream's
forward pass on the stacked rows up to float rounding: on CPU the logits
agree to about 2e-5, as closely as two chunkings of the same queries do.

Two further differences from upstream's ``LimiXPredictor``, both removing a
dependence on how the queries are chunked:

* the feature positional embedding -- a random draw -- comes from a dedicated
  ``torch.Generator`` seeded per member, not from the global generator, which
  upstream's encoder advances by an amount set by the chunk's size before
  drawing it (a draw of noise for masked cells, discarded at inference);
* the features arrive already preprocessed on the context alone
  (:mod:`lazy.models._limix_preprocess`).

:func:`forward` is the uncached path (``kv_cache=False``): upstream's own
forward pass on context and queries stacked. This is a private module.

Typical usage example:

  network = load_network(checkpoint, device="cuda")
  member = Member(x_context, y_standardised, seed=0)
  cache = prefill(network, member)
  logits = decode(network, member, cache, x_query)
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import io
import math
import pathlib
from typing import Any

import numpy as np

from lazy import _typing
from lazy.models import _limix_source

__all__ = [
    "Member",
    "cache_bytes",
    "cache_supported",
    "decode",
    "forward",
    "load_network",
    "prefill",
]


@dataclasses.dataclass
class Member:
    """One ensemble member's context, as the network sees it.

    Attributes:
        x: Preprocessed context features, shape (n_rows, n_features).
        y: Standardised context redshifts, shape (n_rows,).
        seed: Seed of the feature positional embedding's generator.
    """

    x: _typing.FloatArray
    y: _typing.FloatArray
    seed: int


def load_network(checkpoint: pathlib.Path, device: str) -> Any:
    """Builds LimiX-2's network from its checkpoint, ready for inference.

    The checkpoint is read with ``weights_only=True``: it holds tensors and a
    config dict, so nothing in it is executed.

    Args:
        checkpoint: The ``LimiX-2.ckpt`` file.
        device: The torch device to put it on.

    Returns:
        The upstream ``FeaturesTransformer``, in eval mode.
    """
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    loading = _limix_source.load().loading
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    with contextlib.redirect_stdout(io.StringIO()):  # upstream prints
        network, _ = loading.load_from_checkpoint(state)
    return network.to(device).eval()


def cache_supported(network: Any) -> bool:
    """Whether :func:`decode` implements this network's layer layout.

    It mirrors upstream's ``EncoderBaseLayer.forward`` for pre-norm layers
    with output norms and one, non-isolated sample attention -- LimiX-2's
    layout. Another checkpoint may differ; then the uncached path serves.
    """
    stack = network.transformer_encoder
    layers = list(stack.layers)
    return all(
        layer.pre_norm
        and layer.all_norm
        and not layer.seq_attn_isolated
        and sum(_is_sample_attention(step) for step in layer.layer_steps) == 1
        for layer in layers
    ) and not 0 <= stack.cls_only_start_layer < len(layers)


def cache_bytes(network: Any, n_rows: int, n_features: int, dtype: Any) -> int:
    """The memory :func:`prefill`'s cache takes for one member.

    Args:
        network: The loaded network.
        n_rows: Context rows.
        n_features: Preprocessed feature columns.
        dtype: The torch dtype the cache is held in.

    Returns:
        The size in bytes.
    """
    groups = math.ceil(n_features / network.features_per_group)
    tokens = groups + network.y_token_k
    layers = len(network.transformer_encoder.layers)
    itemsize = dtype.itemsize
    return int(n_rows * tokens * network.embed_dim * layers * itemsize)


def prefill(
    network: Any, member: Member, *, mixed_precision: bool
) -> list[Any]:
    """Runs the context through the network and keeps what queries need.

    Args:
        network: The loaded network.
        member: The member's context.
        mixed_precision: Run under autocast (CUDA only).

    Returns:
        Each layer's context input to its sample attention, shape
        (1, n_rows, n_tokens, embed_dim).
    """
    device = _device_of(network)
    with _inference(device, mixed_precision=mixed_precision):
        tokens, feature_mask, y_type = _encode(network, member, None)
        cache = []
        hidden = tokens
        for layer in network.transformer_encoder.layers:
            captured: list[Any] = []
            hidden = _run_layer(
                layer, hidden, feature_mask, y_type, capture=captured
            )
            cache.append(captured[0].to(tokens.dtype))
        return cache


def decode(
    network: Any,
    member: Member,
    cache: list[Any],
    x_query: _typing.FloatArray,
    *,
    mixed_precision: bool,
) -> _typing.FloatArray:
    """The query rows' bucket logits, attending to a prefilled context.

    Args:
        network: The loaded network.
        member: The member's context, as prefilled.
        cache: What :func:`prefill` returned for it.
        x_query: Preprocessed query features, shape (n_queries, n_features).
        mixed_precision: Run under autocast (CUDA only).

    Returns:
        Logits over the network's buckets, shape (n_queries, n_buckets).
    """
    device = _device_of(network)
    n_context = len(member.y)
    stack = network.transformer_encoder
    with _inference(device, mixed_precision=mixed_precision):
        tokens, feature_mask, y_type = _encode(network, member, x_query)
        hidden = tokens[:, n_context:]
        feature_mask = (
            None if feature_mask is None else feature_mask[:, n_context:]
        )
        y_type = y_type[:, n_context:]
        target_tokens = None
        for index, (layer, context) in enumerate(
            zip(stack.layers, cache, strict=True)
        ):
            hidden = _run_layer(
                layer, hidden, feature_mask, y_type, context=context
            )
            if index == stack.reg_y_emb_layer:
                target_tokens = hidden[:, :, -network.y_token_k :].clone()
        if target_tokens is None:  # the embedding is the last layer's
            target_tokens = hidden[:, :, -network.y_token_k :]
        target_tokens = network.reg_y_encoder_out_norm(target_tokens)
        target_tokens = network.flatten_y_tokens(target_tokens)
        logits = network.decoder_for_regression_task(
            target_tokens, 0, y_type, None
        )["reg_output"][0]
        return _numpy(logits)


def forward(
    network: Any,
    member: Member,
    x_query: _typing.FloatArray,
    *,
    mixed_precision: bool,
) -> _typing.FloatArray:
    """The query rows' bucket logits from upstream's own forward pass.

    Args:
        network: The loaded network.
        member: The member's context.
        x_query: Preprocessed query features, shape (n_queries, n_features).
        mixed_precision: Run under autocast (CUDA only).

    Returns:
        Logits over the network's buckets, shape (n_queries, n_buckets).
    """
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    device = _device_of(network)
    x = np.concatenate([member.x, x_query]).astype(np.float32)
    with _inference(device, mixed_precision=mixed_precision):
        output = network(
            x=torch.from_numpy(x).to(device)[None],
            y=_tensor(member.y, device)[None],
            eval_pos=len(member.y),
            task_type="Regression",
            feature_positional_embedding_generator=_generator(
                member.seed, device
            ),
        )
        return _numpy(output["reg_output"][0])


def _encode(
    network: Any, member: Member, x_query: _typing.FloatArray | None
) -> tuple[Any, Any, Any]:
    """Upstream's encoder over the context, and the queries if any.

    Returns:
        The tokens (1, n_rows, n_tokens, embed_dim), the feature mask (or
        None) and the task types (1, n_rows), as a tuple.
    """
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    device = _device_of(network)
    x = member.x if x_query is None else np.concatenate([member.x, x_query])
    x_tensor = torch.from_numpy(x.astype(np.float32)).to(device)[None]
    features = {
        "data": x_tensor,
        "mask": torch.isnan(x_tensor).to(torch.int32),
    }
    targets = {"data": _tensor(member.y, device)[None]}
    # With no queries upstream prints a warning that this is self-attention;
    # it is, by design.
    with contextlib.redirect_stdout(io.StringIO()):
        tokens, feature_mask, y_type, *_ = network._encoder(  # noqa: SLF001 - upstream exposes no encoder API.
            features,
            targets,
            len(member.y),
            None,
            None,
            None,
            "Regression",
            _generator(member.seed, device),
        )
    return tokens, feature_mask, y_type


def _run_layer(
    layer: Any,
    hidden: Any,
    feature_mask: Any,
    y_type: Any,
    *,
    context: Any = None,
    capture: list[Any] | None = None,
) -> Any:
    """Upstream's ``EncoderBaseLayer.forward``, attending to ``context``.

    The layer is pre-norm with output norms, and its sample attention takes
    its keys and values from ``context``. With ``context`` None the rows
    attend to themselves -- the prefill, where they are the context -- and
    ``capture`` collects the attention's input.
    """
    for index, (step, norm) in enumerate(
        zip(layer.layer_steps, layer.layer_norms, strict=True)
    ):
        residual = hidden
        hidden = norm(hidden)
        if _is_sample_attention(step):
            if capture is not None:
                capture.append(hidden)
            keys = hidden if context is None else context
            attention = layer.sequence_attentions[step.keywords.get("index", 0)]
            hidden = attention(
                x=hidden.transpose(1, 2),
                x_kv=keys.transpose(1, 2),
                y_type=y_type,
            )[0].transpose(1, 2)
        elif isinstance(step, functools.partial):
            hidden = step(hidden, feature_mask, 0, y_type=y_type)
        else:
            hidden = step(hidden, y_type=y_type)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        hidden = layer.output_layer_norms[index](hidden)
        hidden = hidden.add_(residual.to(dtype=hidden.dtype))
    return hidden


def _is_sample_attention(step: Any) -> bool:
    """Whether a layer step is its attention across rows."""
    return (
        isinstance(step, functools.partial)
        and getattr(step.func, "__name__", "") == "call_sequence_attention"
    )


def _inference(device: Any, *, mixed_precision: bool) -> Any:
    """Upstream's inference context: no autograd, autocast on CUDA."""
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    stack = contextlib.ExitStack()
    stack.enter_context(torch.inference_mode())
    stack.enter_context(
        torch.autocast(
            device_type=device.type,
            enabled=mixed_precision and device.type == "cuda",
        )
    )
    return stack


def _generator(seed: int, device: Any) -> Any:
    """A fresh generator for the feature positional embedding."""
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    return torch.Generator(device=device).manual_seed(int(seed))


def _tensor(values: _typing.FloatArray, device: Any) -> Any:
    """A float32 tensor on ``device``."""
    import torch  # noqa: PLC0415 - an optional, heavy extra.

    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)


def _device_of(network: Any) -> Any:
    """The device the network's weights live on."""
    return next(network.parameters()).device


def _numpy(logits: Any) -> _typing.FloatArray:
    """Logits (1, n, n_buckets) as a float32 array (n, n_buckets)."""
    return logits.float().squeeze(0).cpu().numpy()
