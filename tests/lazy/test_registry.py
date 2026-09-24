import pytest

from lazy import (
    CHECKPOINTS,
    DEFAULT_VERSIONS,
    ESTIMATORS,
    get_checkpoint,
    get_estimator,
    list_estimators,
    list_versions,
)
from lazy.base import BasePhotoZEstimator


def test_registered_names():
    assert list_estimators() == ["tabfm", "tabicl", "tabpfn"]


@pytest.mark.parametrize("name", ["tabfm", "tabicl", "tabpfn"])
def test_get_estimator_builds_a_photoz_estimator(name):
    est = get_estimator(name)
    assert isinstance(est, BasePhotoZEstimator)
    assert isinstance(est, ESTIMATORS[name])


def test_get_estimator_forwards_parameters():
    assert get_estimator("tabfm", n_dither=3).n_dither == 3


def test_unknown_estimator_names_the_alternatives():
    with pytest.raises(KeyError, match="tabicl"):
        get_estimator("tabdpt")


def test_unknown_parameter_is_rejected_by_the_constructor():
    with pytest.raises(TypeError):
        get_estimator("tabicl", not_a_parameter=1)


def test_every_estimator_has_a_default_version():
    """`LazyModel(name)` has to resolve to weights without being told a version."""
    for name in list_estimators():
        assert name in DEFAULT_VERSIONS
        assert get_checkpoint(name).version == DEFAULT_VERSIONS[name]


def test_every_checkpoint_is_pinned():
    """An unpinned checkpoint would silently change published numbers."""
    for key, spec in CHECKPOINTS.items():
        assert key == spec.key, "a key that disagrees with its record points at the wrong weights"
        assert spec.backend in ESTIMATORS
        assert spec.repo_id and spec.package and spec.version
        assert spec.revision is not None or spec.filename is not None
        assert spec.license_note


def test_a_version_is_reachable_by_name_or_key():
    assert get_checkpoint("tabpfn", "v2.5") is CHECKPOINTS["tabpfn:v2.5"]
    assert get_checkpoint("tabpfn:v2.5") is CHECKPOINTS["tabpfn:v2.5"]
    assert get_checkpoint("tabpfn").version == DEFAULT_VERSIONS["tabpfn"]


def test_an_unknown_version_names_the_ones_that_exist():
    with pytest.raises(KeyError, match=r"v3\.5-fast"):
        get_checkpoint("tabpfn", "v4")


def test_an_unknown_backend_is_reported_as_such():
    with pytest.raises(KeyError, match="unknown backend"):
        get_checkpoint("tabdpt", "v1")


def test_list_versions_is_sorted_and_scoped_to_one_backend():
    assert list_versions("tabpfn") == ["v2", "v2.5", "v2.6", "v3", "v3.5", "v3.5-fast"]
    assert list_versions("tabicl") == ["v2"]
    assert list_versions("tabdpt") == []


def test_provenance_identifies_the_weights_and_the_code():
    """What a results table needs pasted next to it, and nothing machine-specific."""
    record = get_checkpoint("tabpfn", "v3").provenance(device="cuda")
    assert record["backend"] == "tabpfn"
    assert record["version"] == "v3"
    assert record["repo_id"] == "Prior-Labs/tabpfn_3"
    assert record["revision"] == "24a16a89d245878b846555110985634aa2e656d7"
    assert record["package"].startswith("tabpfn ")
    assert record["lazy"].startswith("lazy-photoz ")
    assert record["device"] == "cuda"
    assert not any(isinstance(v, str) and "/home" in v for v in record.values()), (
        "a provenance record travels between machines, so it carries no local paths"
    )


def test_provenance_is_json_serialisable():
    """It is meant to be written out beside a results table."""
    import json

    for spec in CHECKPOINTS.values():
        assert json.loads(json.dumps(spec.provenance(device="cpu")))["version"] == spec.version


def test_constructing_an_estimator_downloads_nothing(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("constructing an estimator must not touch the hub")

    monkeypatch.setattr("lazy.models._hub.Checkpoint.download", explode)
    for name in list_estimators():
        get_estimator(name)
