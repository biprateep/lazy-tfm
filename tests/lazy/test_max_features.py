# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Each member on a random subset of the columns (``max_features``).

The plan is tested directly, and the views through a stand-in that records
the features each member's group is fitted on.
"""

import numpy as np
import pandas as pd
import pytest
import standins

from lazy.models import _members
from lazy.models import _transforms

N_ROWS, N_FEATURES = 40, 6


def _plan(n_kept, *, feature_shuffle=True, bag_rows=N_ROWS):
    """The plan of an ensemble of four members, with the given subsets."""
    return _members.plan(
        n_estimators=4,
        transforms=None,
        native_transforms={"none": "none"},
        feature_shuffle=feature_shuffle,
        bag_rows=bag_rows,
        n_rows=N_ROWS,
        n_features=N_FEATURES,
        random_state=0,
        supports_native_bagging=True,
        n_kept=n_kept,
    )


class _Recording(standins.ScaffoldedHistogramStandIn):
    """Records the features of each group's view and its categorical mask."""

    def _fit_group(self, X, y, group):
        views = getattr(self, "views_", [])
        self.views_ = [*views, (X.copy(), self._categorical_columns(group))]
        return super()._fit_group(X, y, group)


def _features():
    """Columns whose values name them: column j holds 100 * j + row."""
    rows = np.arange(N_ROWS)[:, None]
    return (100.0 * np.arange(N_FEATURES) + rows).astype(float)


def test_every_column_kept_plans_as_before():
    assert _plan(N_FEATURES) == _plan(None)
    assert all(group.columns is None for group in _plan(None))


def test_a_subset_makes_each_member_a_group_with_its_own_sorted_columns():
    groups = _plan(3)
    assert [group.members[0].index for group in groups] == [0, 1, 2, 3]
    subsets = {tuple(group.columns) for group in groups}
    assert len(subsets) > 1
    for group in groups:
        assert group.columns.size == 3
        assert np.all(np.diff(group.columns) > 0)
        assert sorted(group.permutation) == [0, 1, 2]
    unshuffled = _plan(3, feature_shuffle=False)
    assert all(group.permutation is None for group in unshuffled)
    np.testing.assert_array_equal(
        [g.columns for g in unshuffled], [g.columns for g in groups]
    )


@pytest.mark.parametrize("transforms", ["auto", ("quantile+original",)])
def test_each_view_holds_its_members_columns_in_its_order(transforms):
    model = _Recording(
        n_estimators=3,
        max_features=0.5,
        transforms=transforms,
        progress=False,
    ).fit(_features(), np.arange(N_ROWS, dtype=float))
    assert model.provenance_["feature_columns"] == 3
    for (view, _), group in zip(
        model.views_, model.member_groups_, strict=True
    ):
        order = group.columns[group.permutation]
        # The last half is the untransformed copy under "+original".
        names = np.round(view[0, -order.size :] / 100).astype(int)
        np.testing.assert_array_equal(names, order)
        halves = 2 if transforms != "auto" else 1
        assert view.shape[1] == halves * order.size


def test_the_categorical_mask_follows_the_subset():
    frame = pd.DataFrame(_features()).assign(
        kind=pd.Categorical(np.arange(N_ROWS) % 3)
    )
    frame.columns = [str(c) for c in frame.columns]
    model = _Recording(n_estimators=4, max_features=4, progress=False)
    model.fit(frame, np.arange(N_ROWS, dtype=float))
    for (_, mask), group in zip(
        model.views_, model.member_groups_, strict=True
    ):
        order = group.columns[group.permutation]
        np.testing.assert_array_equal(mask, order == N_FEATURES)


@pytest.mark.parametrize(
    ("max_features", "message"),
    [
        (0, "an int max_features must be positive"),
        (1.5, r"a float max_features must lie in \(0, 1\]"),
        ("half", "max_features must be an int, a float or None"),
    ],
)
def test_a_bad_max_features_is_refused_by_name(max_features, message):
    model = standins.HistogramStandIn(max_features=max_features)
    with pytest.raises(ValueError, match=message):
        model.fit(_features(), np.arange(N_ROWS, dtype=float))


def test_a_transform_is_fitted_on_the_members_own_columns():
    spec = _transforms.parse(("quantile",))[0]
    features = _features()
    model = _Recording(
        n_estimators=2,
        max_features=2,
        transforms=("quantile",),
        progress=False,
    ).fit(features, np.arange(N_ROWS, dtype=float))
    for (view, _), group in zip(
        model.views_, model.member_groups_, strict=True
    ):
        kept = features[:, group.columns]
        expected = _transforms.ScaffoldTransform(spec, group.seed)
        expected = expected.fit(kept).transform(kept)[:, group.permutation]
        np.testing.assert_allclose(view, expected)


def test_a_negative_seed_draws_subsets_as_its_wrapped_value_does():
    """Numpy's generators refuse a negative seed; the plan wraps it."""
    groups = _members.plan(
        n_estimators=3,
        transforms=None,
        native_transforms={"none": "none"},
        feature_shuffle=True,
        bag_rows=20,
        n_rows=N_ROWS,
        n_features=N_FEATURES,
        random_state=-1,
        supports_native_bagging=False,
        n_kept=3,
    )
    wrapped = _members.plan(
        n_estimators=3,
        transforms=None,
        native_transforms={"none": "none"},
        feature_shuffle=True,
        bag_rows=20,
        n_rows=N_ROWS,
        n_features=N_FEATURES,
        random_state=2**64 - 1,
        supports_native_bagging=False,
        n_kept=3,
    )
    for group, other in zip(groups, wrapped, strict=True):
        np.testing.assert_array_equal(group.columns, other.columns)
        np.testing.assert_array_equal(group.rows, other.rows)


def test_a_model_pickled_before_max_features_keeps_its_parameters():
    model = standins.HistogramStandIn(n_estimators=2, progress=False)
    model.fit(_features(), np.arange(N_ROWS, dtype=float))
    state = model.__getstate__()
    del state["max_features"]  # As in a pickle from before the parameter.
    old = standins.HistogramStandIn.__new__(standins.HistogramStandIn)
    old.__setstate__(state)
    assert old.get_params()["max_features"] is None
    np.testing.assert_array_equal(
        old.predict(_features(), method="mode"),
        model.predict(_features(), method="mode"),
    )
