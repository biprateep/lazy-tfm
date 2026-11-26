# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Fake upstream objects for the pretrained encoders.

The fakes replace what each encoder loads from ``transformers`` and
``chronos`` (a tokenizer and a network, an image processor and a network,
a Chronos pipeline, TimesFM's network) and the checkpoint download, so
that an encoder's own code (its ``_load``, its pooling, its padding, its
reading of items) runs on a CPU in milliseconds with nothing downloaded.
Each fake pads a batch as upstream does, and gives a padded position a
state of 1e6, so that pooling that reads padding shows. They need torch,
transformers and chronos.
"""

import pathlib
import types

import numpy as np
import pytest

import fakes
import lazy
from lazy.encoders import _pretrained

LAZY_SEAMS = ("lazy.models._hub.Checkpoint.download",)
"""The private lazy names that :func:`install` patches."""

PAD = 1e6
"""The state of a padded or missing position, which pooling must skip."""


class Tokens(dict):
    """A batch of token ids and their mask, as a tokenizer returns it."""

    def to(self, device):
        """The batch itself; it lives on the CPU."""
        return self


class Tokenizer:
    """One token per word between a first and a last, padded on the right."""

    def __call__(self, text, padding, truncation, max_length, return_tensors):
        import torch  # noqa: PLC0415 - collected without torch installed.

        assert (padding, truncation, return_tensors) == (True, True, "pt")
        codes = [[101, *map(len, t.split()), 102][:max_length] for t in text]
        width = max(map(len, codes))
        ids = torch.zeros((len(text), width), dtype=torch.long)
        mask = torch.zeros((len(text), width), dtype=torch.long)
        for row, code in enumerate(codes):
            ids[row, : len(code)] = torch.tensor(code)
            mask[row, : len(code)] = 1
        return Tokens(input_ids=ids, attention_mask=mask)


class Network:
    """What a network shares: its device, ``to`` and ``eval``."""

    def __init__(self, width):
        import torch  # noqa: PLC0415

        self.width = width
        self.device = torch.device("cpu")

    def to(self, device):
        """The network itself, on the CPU."""
        return self

    def eval(self):
        """The network itself."""
        return self


class TextModel(Network):
    """Token states that depend on each token alone; padding is 1e6."""

    def __call__(self, input_ids, attention_mask):
        import torch  # noqa: PLC0415

        waves = torch.arange(1, self.width + 1, dtype=torch.float32)
        states = torch.cos(input_ids[..., None] * waves)
        states[attention_mask == 0] = PAD
        return types.SimpleNamespace(last_hidden_state=states)


class ImageProcessor:
    """The mean of each RGB image's channels, as its 'pixels'."""

    def __call__(self, images, return_tensors):
        import torch  # noqa: PLC0415

        assert return_tensors == "pt"
        assert all(image.mode == "RGB" for image in images)
        means = [np.asarray(image, float).mean(axis=(0, 1)) for image in images]
        return {"pixel_values": torch.tensor(np.array(means))}


class ImageModel(Network):
    """An embedding that is a fixed function of the channel means.

    It reads the means of :class:`ImageProcessor`, or the means of the
    pixels that lazy itself prepares, in [0, 1], for Zoobot.
    """

    def __call__(self, pixel_values):
        import torch  # noqa: PLC0415

        if pixel_values.ndim == 4:
            pixel_values = pixel_values.double().mean(dim=(2, 3)) * 255
        waves = torch.arange(1, self.width // 3 + 2, dtype=torch.float64)
        token = torch.sin(pixel_values[:, :, None] * waves / 255)
        token = token.reshape(len(pixel_values), -1)[:, : self.width]
        return types.SimpleNamespace(pooler_output=token, image_embeds=token)


class Chronos:
    """Chronos's ``embed``: left padding, tokens of ``patch`` values.

    A token's state is the mean of its observed values times a ramp, and
    the summary token's is the mean of the series; a padded or wholly
    missing token's is 1e6.
    """

    def __init__(self, width, patch):
        self.width = width
        self.patch = patch

    def embed(self, context):
        import torch  # noqa: PLC0415

        tokens = [-(-len(values) // self.patch) for values in context]
        states = torch.full((len(context), max(tokens) + 1, self.width), PAD)
        ramp = torch.linspace(0, 1, self.width)
        for row, values in enumerate(context):
            pad = torch.full((tokens[row] * self.patch - len(values),), np.nan)
            grid = torch.cat([pad, values]).reshape(-1, self.patch)
            means = grid.nanmean(dim=1)
            own = torch.cat([means, values.nanmean()[None]])
            own = torch.where(own.isnan(), PAD, own)
            states[row, -len(own) :] = own[:, None] * ramp
        return states, None


class Chronos2(Chronos):
    """Chronos-2's ``embed``: a masked output token after the summary.

    The output token's state is 1e6, and each series' states come as a
    tensor of one variate.
    """

    def embed(self, inputs):
        import torch  # noqa: PLC0415

        states, _ = super().embed(inputs)
        output = torch.full((len(inputs), 1, self.width), PAD)
        return list(torch.cat([states, output], dim=1)[:, None]), None


class TimesFM(Network):
    """TimesFM's decoder: tokens of ``patch`` values, its padding masked.

    A token's state is the mean of its values that are not padding times a
    ramp to 1; a token of padding alone gets 1e6.
    """

    def __init__(self, width, patch):
        super().__init__(width)
        self.patch = patch

    def __call__(self, past_values, past_values_padding):
        import torch  # noqa: PLC0415

        tokens = (len(past_values), -1, self.patch)
        values = past_values.double().reshape(tokens)
        real = (past_values_padding == 0).reshape(tokens)
        means = (values * real).sum(dim=2) / real.sum(dim=2).clamp(min=1)
        ramp = torch.linspace(0, 1, self.width, dtype=torch.float64)
        states = means[..., None] * ramp
        states[~real.any(dim=2)] = PAD
        return types.SimpleNamespace(last_hidden_state=states)


def install(monkeypatch, tmp_path):
    """Installs the fakes; returns how often each model's network loads.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture.
        tmp_path: A directory to stand in for the downloaded weights.

    Returns:
        A dict that counts loads by ``"<modality>:<model>"``.
    """
    transformers = pytest.importorskip("transformers")
    chronos = pytest.importorskip("chronos")
    loads = {}

    def download(checkpoint, local_files_only=False):
        return tmp_path / checkpoint.backend / checkpoint.version

    def recipe(path):
        """The recipe of the network whose weights are at ``path``."""
        path = pathlib.Path(path)
        key = f"{path.parent.name}:{path.name}"
        loads[key] = loads.get(key, 0) + 1
        return lazy.encoders.get_encoder(path.parent.name).recipes[path.name]

    def network(path, **settings):
        # Every setting that changes an embedding is pinned, never left to
        # the installed transformers' defaults.
        assert settings == _pretrained.PINNED
        found = recipe(path)
        modality = pathlib.Path(path).parent.name
        if modality == "time":
            return TimesFM(found.states, found.patch)
        cls = {"text": TextModel, "image": ImageModel}
        return cls[modality](found.width)

    def pipeline(path, device_map, **settings):
        import torch  # noqa: PLC0415

        found = recipe(path)
        float32 = {"torch_dtype": torch.float32}
        assert settings == (_pretrained.PINNED if found.pinned else float32)
        cls = Chronos2 if found.outputs else Chronos
        return cls(found.states, found.patch)

    fakes.patch(
        monkeypatch,
        LAZY_SEAMS,
        {"lazy.models._hub.Checkpoint.download": download},
    )
    for owner in (
        transformers.AutoModel,
        transformers.SiglipVisionModel,
        transformers.CLIPVisionModelWithProjection,
    ):
        monkeypatch.setattr(owner, "from_pretrained", network)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda path: Tokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        lambda path, backend: ImageProcessor(),
    )
    monkeypatch.setattr(
        chronos.BaseChronosPipeline, "from_pretrained", pipeline
    )
    return loads
