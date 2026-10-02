# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Planning an ensemble: which member sees which rows, columns and transform.

An ensemble of ``n_estimators`` members is described once, the same way for
every model, and then split into *groups*: runs of members one call of the
model can serve. A model that implements a feature natively (its own column
shuffles, its own transforms, its own row subsampling) takes a whole group
at once, so its own ensembling -- and its tuned default recipe -- is used
unchanged; whatever it lacks is scaffolded by giving that member a group of
its own, with its rows, transform and column permutation applied here.

The rules, in order:

* member ``i`` gets transform ``transforms[i % k]`` (round robin, as every
  upstream model assigns its own), or the model's own recipe for ``"auto"``;
* the bags are drawn here, for every model, exactly as the paper's bagged
  LimiX-2 drew them (:func:`draw_bags`), so member ``i`` sees the same rows
  whichever model runs it;
* without bagging, members whose transform the model has natively form one
  group, and each other transform forms a group of its own, in the order
  transforms first appear;
* with bagging, a model with native row subsampling keeps its native
  members in one group and is handed each member's bag; every scaffolded
  member, and every member of a model without row subsampling, is a group
  of its own, restricted to its bag and given an explicit column
  permutation, so that its transform is fitted on its own rows;
* a group is seeded by its first member, and member ``i``'s seed is drawn
  from ``SeedSequence([random_state, i])`` (:func:`member_seed`), so that no
  member of one ``random_state`` shares a seed with any member of another.

Typical usage example:

  groups = _members.plan(
      n_estimators=8, transforms=None, native_transforms={"none": "none"},
      feature_shuffle=True, bag_rows=20_000, n_rows=35_000, n_features=11,
      random_state=0, supports_native_bagging=False,
  )
"""

from collections.abc import Mapping, Sequence
import dataclasses
import math
import numbers
from typing import Any

import numpy as np

from lazy import _typing
from lazy.models import _transforms

__all__ = [
    "MemberGroup",
    "MemberSpec",
    "draw_bags",
    "feature_permutations",
    "member_seed",
    "plan",
    "resolve_bag_size",
]

#: Fewer context rows per bag than this draws a warning at fit.
MIN_BAG_ROWS = 10


@dataclasses.dataclass(frozen=True)
class MemberSpec:
    """One ensemble member.

    Attributes:
        index: The member's position in the ensemble.
        transform: Its transform, or None for the model's own recipe.
        rows: Its sorted context rows, or None for all of them.
    """

    index: int
    transform: _transforms.TransformSpec | None
    rows: _typing.IntArray | None = None


@dataclasses.dataclass(frozen=True)
class MemberGroup:
    """Members one call of the model serves together.

    Attributes:
        index: The group's position.
        seed: The seed the model is run with.
        members: The members, in ensemble order.
        native_transforms: The model's own tokens for the members'
            transforms, one per member, or None for its own recipe.
        scaffold: A transform the package applies before the model sees the
            features, or None.
        permutation: A column permutation the package applies, or None to
            leave column shuffling to the model.
        feature_shuffle: Whether the model shuffles columns itself.
        rows: The group's context rows (scaffolded bagging), or None.
        member_rows: Each member's rows, for a model that subsamples rows
            natively, or None.
    """

    index: int
    seed: int
    members: tuple[MemberSpec, ...]
    native_transforms: tuple[Any, ...] | None
    scaffold: _transforms.TransformSpec | None
    permutation: _typing.IntArray | None
    feature_shuffle: bool
    rows: _typing.IntArray | None = None
    member_rows: tuple[_typing.IntArray, ...] | None = None

    @property
    def n_members(self) -> int:
        """How many ensemble members the group serves."""
        return len(self.members)


def resolve_bag_size(bag_size: float | None, n_rows: int) -> int:
    """The number of context rows each member sees.

    Args:
        bag_size: An integer count (Python or NumPy), a float fraction in
            (0, 1], or None for all. The type decides: ``1`` is one row,
            ``1.0`` the whole context.
        n_rows: The context size.

    Returns:
        The rows per member, between 1 and ``n_rows``.

    Raises:
        ValueError: If ``bag_size`` is none of those.
    """
    if bag_size is None:
        return n_rows
    if isinstance(bag_size, bool) or not isinstance(bag_size, numbers.Real):
        raise ValueError(
            f"bag_size must be an int, a float or None: {bag_size=}"
        )
    if not isinstance(bag_size, numbers.Integral):
        fraction = float(bag_size)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"a float bag_size must lie in (0, 1]: {bag_size=}"
            )
        return max(1, min(n_rows, round(fraction * n_rows)))
    if bag_size < 1:
        raise ValueError(f"an int bag_size must be positive: {bag_size=}")
    return min(int(bag_size), n_rows)


def member_seed(random_state: int, index: int) -> int:
    """The seed of member ``index`` of an ensemble seeded ``random_state``.

    Drawn from ``SeedSequence([random_state, index])``, which hashes the pair,
    so seeds of neighbouring ensembles do not overlap: an offset scheme such
    as ``random_state + index`` would give member ``i`` at ``random_state=r``
    the seed of member ``i - 1`` at ``r + 1``. Kept below 2**31, so that it is
    a valid seed for every upstream model.

    Args:
        random_state: The ensemble's seed.
        index: The member's position in the ensemble.

    Returns:
        The seed, in ``[0, 2**31)``.

    Examples:
        >>> member_seed(0, 1) == member_seed(1, 0)
        False
        >>> 0 <= member_seed(0, 0) < 2**31
        True
    """
    sequence = np.random.SeedSequence([int(random_state) % 2**64, int(index)])
    return int(sequence.generate_state(1)[0] % 2**31)


def draw_bags(
    n_members: int, bag_rows: int, n_rows: int, seed: int
) -> list[_typing.IntArray]:
    """Each member's sorted context rows, drawn as the paper's LimiX did.

    Rows are drawn without replacement within a bag and independently across
    bags, from one generator seeded ``seed``.

    Args:
        n_members: How many bags.
        bag_rows: Rows per bag.
        n_rows: The context size.
        seed: The generator's seed.

    Returns:
        One sorted index array per member.
    """
    rng = np.random.default_rng(seed)
    bags = []
    for _ in range(n_members):
        if bag_rows >= n_rows:
            bags.append(np.arange(n_rows))
        else:
            bags.append(np.sort(rng.choice(n_rows, bag_rows, replace=False)))
    return bags


def feature_permutations(
    n_members: int, n_features: int, seed: int
) -> list[_typing.IntArray]:
    """A distinct column permutation per member, as far as possible.

    Member ``i`` gets a random base permutation rolled by ``i``, which is
    distinct for the first ``n_features`` members; later members get fresh
    random permutations, redrawn until new. Only once every one of the
    ``n_features!`` permutations is taken do they repeat.
    """
    rng = np.random.default_rng(seed)
    base = rng.permutation(n_features)
    available = math.factorial(n_features)
    seen: set[tuple[int, ...]] = set()
    permutations = []
    for i in range(n_members):
        candidate = np.roll(base, i) if i < n_features else None
        while candidate is None or (
            tuple(candidate) in seen and len(seen) < available
        ):
            candidate = rng.permutation(n_features)
        seen.add(tuple(candidate))
        permutations.append(candidate)
    return permutations


def plan(
    *,
    n_estimators: int,
    transforms: Sequence[_transforms.TransformSpec] | None,
    native_transforms: Mapping[str, Any],
    feature_shuffle: bool,
    bag_rows: int,
    n_rows: int,
    n_features: int,
    random_state: int,
    supports_native_bagging: bool,
    auto_tokens: Sequence[Any] | None = None,
) -> tuple[MemberGroup, ...]:
    """Splits an ensemble into the groups the model is called with.

    Args:
        n_estimators: The number of members.
        transforms: The transforms members cycle through, or None for the
            model's own recipe.
        native_transforms: Uniform transform name to the model's own token,
            for the transforms the model implements.
        feature_shuffle: Whether members see permuted columns.
        bag_rows: Rows per member; ``n_rows`` means no bagging.
        n_rows: The context size.
        n_features: The number of feature columns.
        random_state: The ensemble's seed.
        supports_native_bagging: Whether the model subsamples rows itself,
            given each member's bag.
        auto_tokens: The model's own default recipe as its tokens, which
            single-member groups cycle through under ``"auto"`` so that
            scaffolded bagging keeps the model's mix of transforms.

    Returns:
        The groups, in order.
    """
    specs = [
        None if transforms is None else transforms[i % len(transforms)]
        for i in range(n_estimators)
    ]
    bagging = bag_rows < n_rows
    bags = (
        draw_bags(n_estimators, bag_rows, n_rows, random_state)
        if bagging
        else None
    )
    members = [
        MemberSpec(i, spec, None if bags is None else bags[i])
        for i, spec in enumerate(specs)
    ]
    if not bagging:
        return _grouped(
            members, native_transforms, feature_shuffle, random_state
        )
    alone = [
        member
        for member in members
        if not (
            supports_native_bagging
            and _is_native(member.transform, native_transforms)
        )
    ]
    permutations = (
        feature_permutations(n_estimators, n_features, random_state)
        if feature_shuffle and alone
        else None
    )
    groups = list(
        _one_group_per_member(
            alone, native_transforms, permutations, random_state, auto_tokens
        )
    )
    alone_indices = {member.index for member in alone}
    shared = [m for m in members if m.index not in alone_indices]
    if shared:
        groups.extend(
            _grouped(shared, native_transforms, feature_shuffle, random_state)
        )
    groups.sort(key=lambda group: group.members[0].index)
    return tuple(
        dataclasses.replace(group, index=index)
        for index, group in enumerate(groups)
    )


def _is_native(
    spec: _transforms.TransformSpec | None, native: Mapping[str, Any]
) -> bool:
    """Whether the model implements this member's transform itself."""
    return spec is None or spec.name in native


def _grouped(
    members: list[MemberSpec],
    native: Mapping[str, Any],
    feature_shuffle: bool,
    random_state: int,
) -> tuple[MemberGroup, ...]:
    """Members grouped by who implements their transform; see :func:`plan`."""
    keys: list[str] = []
    for member in members:
        key = (
            "native"
            if _is_native(member.transform, native)
            else member.transform.name  # type: ignore[union-attr]  # None is native
        )
        if key not in keys:
            keys.append(key)
    groups = []
    for index, key in enumerate(keys):
        chosen = tuple(
            m
            for m in members
            if (
                "native"
                if _is_native(m.transform, native)
                else m.transform.name  # type: ignore[union-attr]  # None is native
            )
            == key
        )
        if key == "native":
            tokens = (
                None
                if chosen[0].transform is None
                else tuple(native[m.transform.name] for m in chosen)  # type: ignore[union-attr]  # checked native
            )
            scaffold = None
        else:
            tokens = (
                (native["none"],) * len(chosen) if "none" in native else None
            )
            scaffold = chosen[0].transform
        member_rows = (
            None
            if chosen[0].rows is None
            else tuple(m.rows for m in chosen if m.rows is not None)
        )
        groups.append(
            MemberGroup(
                index=index,
                seed=member_seed(random_state, chosen[0].index),
                members=chosen,
                native_transforms=tokens,
                scaffold=scaffold,
                permutation=None,
                feature_shuffle=feature_shuffle,
                member_rows=member_rows,
            )
        )
    return tuple(groups)


def _one_group_per_member(
    members: list[MemberSpec],
    native: Mapping[str, Any],
    permutations: list[_typing.IntArray] | None,
    random_state: int,
    auto_tokens: Sequence[Any] | None,
) -> tuple[MemberGroup, ...]:
    """One group per bagged member, restricted to its rows.

    Args:
        members: The members, each with its bag.
        native: Uniform transform name to the model's own token.
        permutations: Every member's column permutation, indexed by member,
            or None to leave the columns in order.
        random_state: The ensemble's seed.
        auto_tokens: The model's own recipe as tokens, cycled under
            ``"auto"``.

    Returns:
        The groups, in member order.
    """
    groups = []
    for member in members:
        if member.transform is None:
            tokens = (
                None
                if not auto_tokens
                else (auto_tokens[member.index % len(auto_tokens)],)
            )
            scaffold = None
        elif _is_native(member.transform, native):
            tokens = (native[member.transform.name],)
            scaffold = None
        else:
            tokens = (native["none"],) if "none" in native else None
            scaffold = member.transform
        groups.append(
            MemberGroup(
                index=member.index,
                seed=member_seed(random_state, member.index),
                members=(member,),
                native_transforms=tokens,
                scaffold=scaffold,
                permutation=(
                    None if permutations is None else permutations[member.index]
                ),
                feature_shuffle=False,
                rows=member.rows,
            )
        )
    return tuple(groups)
