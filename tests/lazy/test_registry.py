import pytest

from lazy import ESTIMATORS, get_estimator, list_estimators
from lazy.base import BasePhotoZEstimator
from lazy.models import CHECKPOINTS


def test_registered_names():
    assert list_estimators() == ["tabfm", "tabicl"]


@pytest.mark.parametrize("name", ["tabfm", "tabicl"])
def test_get_estimator_builds_a_photoz_estimator(name):
    est = get_estimator(name)
    assert isinstance(est, BasePhotoZEstimator)
    assert isinstance(est, ESTIMATORS[name])


def test_get_estimator_forwards_parameters():
    assert get_estimator("tabfm", n_dither=3).n_dither == 3


def test_unknown_estimator_names_the_alternatives():
    with pytest.raises(KeyError, match="tabicl"):
        get_estimator("tabpfn")


def test_unknown_parameter_is_rejected_by_the_constructor():
    with pytest.raises(TypeError):
        get_estimator("tabicl", not_a_parameter=1)


def test_every_estimator_has_a_pinned_checkpoint():
    """An unpinned checkpoint would silently change published numbers."""
    for name in list_estimators():
        spec = CHECKPOINTS[name]
        assert spec.repo_id
        assert spec.revision is not None or spec.filename is not None
        assert spec.license_note


def test_constructing_an_estimator_downloads_nothing(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("constructing an estimator must not touch the hub")

    monkeypatch.setattr("lazy.models._hub.Checkpoint.download", explode)
    for name in list_estimators():
        get_estimator(name)
