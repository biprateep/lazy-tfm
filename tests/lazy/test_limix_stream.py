# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""LimiX-2's network as lazy loads it, on a stand-in for upstream's."""

import gc
import types
import weakref

import numpy as np
import pytest

from lazy.models import _limix_source
from lazy.models import _limix_stream

torch = pytest.importorskip("torch")


class _Network(torch.nn.Module):
    """The part of upstream's ``FeaturesTransformer`` that draws the embedding.

    ``add_embeddings`` is upstream's (``model/v2_0/transformer.py``), and
    ``_encoder`` calls it as upstream's does, recording the generator.
    """

    def __init__(self):
        super().__init__()
        self.feature_positional_embedding = torch.nn.Linear(2, 8)
        self.generators = []

    def add_embeddings(self, x, *, generator=None):
        if generator is not None and torch.device(generator.device) != x.device:
            raise ValueError(
                "feature positional embedding generator must use the input "
                f"device: generator={generator.device}, input={x.device}"
            )
        embs = torch.randn(
            (x.shape[2], x.shape[3] // 4),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        embs = self.feature_positional_embedding(embs)
        x += embs[None, None]
        return x

    def _encoder(self, features, targets, eval_pos, *args):
        del targets, eval_pos  # Unused.
        generator = args[-1]
        self.generators.append(generator)
        data = features["data"]
        tokens = torch.zeros((1, data.shape[1], 3, 8), device=data.device)
        y_type = torch.zeros((1, data.shape[1]), device=data.device)
        return self.add_embeddings(tokens, generator=generator), None, y_type


@pytest.fixture
def load(monkeypatch, tmp_path):
    """Loads the stand-in through :func:`_limix_stream.load_network`."""
    loading = types.SimpleNamespace(
        load_from_checkpoint=lambda state: (_Network(), None)
    )
    monkeypatch.setattr(
        _limix_source, "load", lambda: types.SimpleNamespace(loading=loading)
    )
    checkpoint = tmp_path / "LimiX-2.ckpt"
    torch.save({}, checkpoint)
    return lambda device: _limix_stream.load_network(checkpoint, device)


def test_the_positional_embedding_is_drawn_on_the_cpu_on_any_device(load):
    network = load("meta")  # neither a CPU nor able to give a generator
    member = _limix_stream.Member(np.zeros((4, 2)), np.zeros(4), seed=7)
    tokens, _, _ = _limix_stream._encode(network, member, None)
    (generator,) = network.generators
    assert generator.device == torch.device("cpu")
    assert generator.initial_seed() == 7
    assert tokens.device == torch.device("meta")


def test_on_a_cpu_the_positional_embedding_is_upstreams(load):
    network = load("cpu")
    x = torch.randn((1, 4, 3, 8), generator=torch.Generator().manual_seed(0))
    expected = _Network.add_embeddings(
        network, x.clone(), generator=torch.Generator().manual_seed(5)
    )
    drawn = network.add_embeddings(
        x.clone(), generator=_limix_stream._generator(5)
    )
    assert torch.equal(drawn, expected)


def test_a_dropped_network_is_freed_without_the_cycle_collector(load):
    network = load("cpu")
    alive = weakref.ref(network)
    gc.disable()
    try:
        del network
        assert alive() is None
    finally:
        gc.enable()
