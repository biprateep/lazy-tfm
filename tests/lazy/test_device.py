# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The ``device`` parameter, resolved without touching a GPU."""

import pytest

from lazy.models import _device

torch = pytest.importorskip("torch")


@pytest.fixture
def no_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.fixture
def two_gpus(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)


def test_auto_falls_back_to_the_cpu(no_accelerator):
    assert _device.resolve_device("auto") == "cpu"


def test_auto_prefers_cuda(two_gpus):
    assert _device.resolve_device("auto") == "cuda"


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", "cpu"),
        ("CPU", "cpu"),
        ("CUDA", "cuda"),
        ("cuda:1", "cuda:1"),
        (torch.device("cuda", 1), "cuda:1"),
        (torch.device("cpu"), "cpu"),
    ],
)
def test_strings_and_torch_devices_are_accepted(two_gpus, device, expected):
    assert _device.resolve_device(device) == expected


@pytest.mark.parametrize("device", [None, True, "gpu", "cuda1", "tpu", 0])
def test_an_unknown_device_is_named_in_a_value_error(device):
    with pytest.raises(ValueError, match="'auto', 'cpu', 'cuda'"):
        _device.resolve_device(device)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "mps"])
def test_a_missing_accelerator_is_an_error(no_accelerator, device):
    with pytest.raises(RuntimeError, match="unavailable"):
        _device.resolve_device(device)


def test_a_cuda_index_beyond_the_visible_devices_is_an_error(two_gpus):
    with pytest.raises(RuntimeError, match="only 2 CUDA"):
        _device.resolve_device("cuda:2")
