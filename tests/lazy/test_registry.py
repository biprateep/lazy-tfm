# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey

import json

import pytest

import lazy


def _explode(*args, **kwargs):
    raise AssertionError("constructing an estimator must not touch the hub")


def test_registered_names():
    assert lazy.list_estimators() == ["limix", "tabfm", "tabicl", "tabpfn"]


@pytest.mark.parametrize("name", ["limix", "tabfm", "tabicl", "tabpfn"])
def test_get_estimator_builds_a_photoz_estimator(name):
    est = lazy.get_estimator(name)
    assert isinstance(est, lazy.BasePhotoZEstimator)
    assert isinstance(est, lazy.ESTIMATORS[name])


def test_get_estimator_forwards_parameters():
    assert lazy.get_estimator("tabfm", n_dither=3).n_dither == 3


def test_unknown_estimator_names_the_alternatives():
    with pytest.raises(KeyError, match="tabicl"):
        lazy.get_estimator("tabdpt")


def test_unknown_parameter_is_rejected_by_the_constructor():
    with pytest.raises(TypeError):
        lazy.get_estimator("tabicl", not_a_parameter=1)


def test_every_estimator_has_a_default_version():
    """`LazyModel(name)` has to resolve to weights with no version given."""
    for name in lazy.list_estimators():
        assert name in lazy.DEFAULT_VERSIONS
        assert lazy.get_checkpoint(name).version == lazy.DEFAULT_VERSIONS[name]


def test_every_checkpoint_is_pinned():
    """An unpinned checkpoint would silently change published numbers."""
    for key, spec in lazy.CHECKPOINTS.items():
        assert key == spec.key, (
            "a key that disagrees with its record points at the wrong weights"
        )
        assert spec.backend in lazy.ESTIMATORS
        assert spec.repo_id and spec.package and spec.version
        assert spec.revision is not None or spec.filename is not None
        assert spec.license_note


def test_a_version_is_reachable_by_name_or_key():
    assert (
        lazy.get_checkpoint("tabpfn", "v2.5") is lazy.CHECKPOINTS["tabpfn:v2.5"]
    )
    assert lazy.get_checkpoint("tabpfn:v2.5") is lazy.CHECKPOINTS["tabpfn:v2.5"]
    assert (
        lazy.get_checkpoint("tabpfn").version == lazy.DEFAULT_VERSIONS["tabpfn"]
    )


def test_an_unknown_version_names_the_ones_that_exist():
    with pytest.raises(KeyError, match=r"v3\.5-fast"):
        lazy.get_checkpoint("tabpfn", "v4")


def test_an_unknown_backend_is_reported_as_such():
    with pytest.raises(KeyError, match="unknown backend"):
        lazy.get_checkpoint("tabdpt", "v1")


def test_list_versions_is_sorted_and_scoped_to_one_backend():
    assert lazy.list_versions("tabpfn") == [
        "v2",
        "v2.5",
        "v2.6",
        "v3",
        "v3.5",
        "v3.5-fast",
    ]
    assert lazy.list_versions("tabicl") == ["v2"]
    assert not lazy.list_versions("tabdpt")


def test_provenance_identifies_the_weights_and_the_code():
    """What a results table needs pasted beside it, nothing machine-specific."""
    record = lazy.get_checkpoint("tabpfn", "v3").provenance(device="cuda")
    assert record["backend"] == "tabpfn"
    assert record["version"] == "v3"
    assert record["repo_id"] == "Prior-Labs/tabpfn_3"
    assert record["revision"] == "24a16a89d245878b846555110985634aa2e656d7"
    assert record["package"].startswith("tabpfn ")
    assert record["lazy"].startswith("lazy-tfm ")
    assert record["device"] == "cuda"
    assert not any(
        isinstance(v, str) and "/home" in v for v in record.values()
    ), (
        "a provenance record travels between machines, so it carries no local"
        " paths"
    )


def test_provenance_is_json_serialisable():
    """It is meant to be written out beside a results table."""
    for spec in lazy.CHECKPOINTS.values():
        assert (
            json.loads(json.dumps(spec.provenance(device="cpu")))["version"]
            == spec.version
        )


def test_constructing_an_estimator_downloads_nothing(monkeypatch):
    monkeypatch.setattr("lazy.models._hub.Checkpoint.download", _explode)
    for name in lazy.list_estimators():
        lazy.get_estimator(name)
