import numpy as np
import pytest

from lazy.features import FEATURE_RECIPES, build_features

EXPECTED_COLUMNS = {
    "raw12": 12,
    "adjcolors": 12,
    "adjcolors_cerr": 12,
    "allcolors": 27,
    "colors5": 5,
}


@pytest.mark.parametrize("recipe", sorted(FEATURE_RECIPES))
def test_every_recipe_builds_the_documented_width(photometry, recipe):
    X = build_features(photometry, recipe)
    assert len(X) == len(photometry)
    assert X.shape[1] == EXPECTED_COLUMNS[recipe]
    assert np.isfinite(X.to_numpy()).all()


def test_adjcolors_are_magnitude_differences(photometry):
    X = build_features(photometry, "adjcolors")
    assert np.allclose(X["G-R"], photometry["G"] - photometry["R"])
    assert np.allclose(X["I"], photometry["I"])


def test_colour_errors_are_the_quadrature_sum_of_the_two_magnitude_errors(photometry):
    """This is the whole point of adjcolors_cerr over adjcolors."""
    X = build_features(photometry, "adjcolors_cerr")
    assert np.allclose(X["G-RERR"], np.hypot(photometry["GERR"], photometry["RERR"]))


def test_adjcolors_variants_have_the_same_shape_so_they_isolate_the_errors(photometry):
    plain = build_features(photometry, "adjcolors")
    propagated = build_features(photometry, "adjcolors_cerr")
    assert plain.shape == propagated.shape


def test_colors5_carries_no_magnitude_or_error(photometry):
    X = build_features(photometry, "colors5")
    assert not any(c.endswith("ERR") for c in X.columns)
    assert "I" not in X.columns


def test_unknown_recipe_names_the_alternatives(photometry):
    with pytest.raises(ValueError, match="adjcolors_cerr"):
        build_features(photometry, "magic")


def test_missing_columns_are_reported(photometry):
    with pytest.raises(KeyError, match="UERR"):
        build_features(photometry.drop(columns=["UERR"]), "raw12")


def test_alternative_band_sets_are_supported(photometry):
    X = build_features(photometry, "colors5", bands=("G", "R", "I"))
    assert list(X.columns) == ["G-R", "R-I"]
