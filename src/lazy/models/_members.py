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
* with ``max_features`` below every column, every member is a group of its
  own that sees its own random subset of the columns, drawn as the bags
  are (:func:`draw_bags`) from a seed of their own, transformed on its own
  and then permuted when columns are shuffled, as a scaffolded bagged
  member is;
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
import inspect
import math
import numbers
from typing import Any

import numpy as np

from lazy import _typing
from lazy.models import _transforms

__all__ = [
    "MemberGroup",
    "MemberSpec",
    "cycle_tokens",
    "draw_bags",
    "feature_permutations",
    "member_seed",
    "plan",
    "resolve_count",
]

MIN_BAG_ROWS = 10
"""Fewer context rows per bag than this draws a warning at fit."""


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
        columns: The feature columns the member sees, in increasing order,
            or None for all of them (``max_features``).
        permutation: A permutation of those columns that the package
            applies, or None to leave column shuffling to the model.
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
    columns: _typing.IntArray | None = None

    @property
    def n_members(self) -> int:
        """How many ensemble members the group serves."""
        return len(self.members)


def resolve_count(value: float | None, total: int, name: str) -> int:
    """How many of ``total`` rows or columns each member sees.

    Args:
        value: An integer count (Python or NumPy), a float fraction in
            (0, 1], or None for all. The type decides: ``1`` is one,
            ``1.0`` all of them.
        total: How many there are.
        name: The parameter's name, for the error.

    Returns:
        The count per member, between 1 and ``total``.

    Raises:
        ValueError: If ``value`` is none of those.
    """
    if value is None:
        return total
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be an int, a float or None: {value=}")
    if not isinstance(value, numbers.Integral):
        fraction = float(value)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"a float {name} must lie in (0, 1]: {value=}")
        return max(1, min(total, round(fraction * total)))
    if value < 1:
        raise ValueError(f"an int {name} must be positive: {value=}")
    return min(int(value), total)


def member_seed(random_state: int, *indices: int) -> int:
    """The seed of member ``index`` of an ensemble seeded ``random_state``.

    Further indices seed a part of a member, such as step ``j`` of its
    pipeline.

    Drawn from ``SeedSequence([random_state, *indices])``, which hashes them,
    so seeds of neighboring ensembles do not overlap: an offset scheme such
    as ``random_state + index`` would give member ``i`` at ``random_state=r``
    the seed of member ``i - 1`` at ``r + 1``. Kept below 2**31, so that it is
    a valid seed for every upstream model.

    Args:
        random_state: The ensemble's seed.
        *indices: The member's position in the ensemble, then the
            position of a part within it, and so on.

    Returns:
        The seed, in ``[0, 2**31)``.

    Examples:
        >>> member_seed(0, 1) == member_seed(1, 0)
        False
        >>> 0 <= member_seed(0, 0) < 2**31
        True
    """
    sequence = np.random.SeedSequence(
        [int(random_state) % 2**64, *(int(index) for index in indices)]
    )
    return int(sequence.generate_state(1)[0] % 2**31)


def same_value(first: object, second: object) -> bool:
    """Whether two parameter values are equal, False if undecidable."""
    if type(first) is not type(second):
        return False
    try:
        return bool(first == second)
    except (TypeError, ValueError):  # An array, say, with no single truth.
        return False


def seed_if_default(estimator: Any, seed: int) -> None:
    """Seeds an estimator whose ``random_state`` is still its default.

    A value equal to the constructor's default is how an estimator that
    was given no seed is told apart from one that was: ``clone`` writes
    every default out.

    Args:
        estimator: An estimator, modified in place; one without a
            ``random_state`` with a default is left as it is.
        seed: The seed to give it.
    """
    if not hasattr(estimator, "get_params"):
        return
    current = estimator.get_params(deep=False)
    default = inspect.signature(type(estimator).__init__).parameters.get(
        "random_state"
    )
    if (
        "random_state" in current
        and default is not None
        and same_value(current["random_state"], default.default)
    ):
        estimator.set_params(random_state=seed)


def draw_bags(
    n_members: int,
    bag_rows: int,
    n_rows: int,
    seed: int | np.random.SeedSequence,
) -> list[_typing.IntArray]:
    """Each member's sorted context rows, drawn as the paper's LimiX did.

    Rows are drawn without replacement within a bag and independently across
    bags, from one generator seeded ``seed``. Each member's columns under
    ``max_features`` are drawn the same way, from a seed of their own.

    Args:
        n_members: How many bags.
        bag_rows: Rows per bag.
        n_rows: The context size.
        seed: The generator's seed, non-negative.

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
    n_kept: int | None = None,
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
        n_kept: The feature columns each member sees (``max_features``);
            None, or ``n_features``, for all. Fewer makes every member a
            group of its own, with its own subset.

    Returns:
        The groups, in order.
    """
    specs = [
        None if transforms is None else transforms[i % len(transforms)]
        for i in range(n_estimators)
    ]
    # numpy's generators take a non-negative seed; a negative random_state
    # wraps, which leaves every non-negative one as it was.
    entropy = random_state % 2**64
    bagging = bag_rows < n_rows
    bags = (
        draw_bags(n_estimators, bag_rows, n_rows, entropy) if bagging else None
    )
    members = [
        MemberSpec(i, spec, None if bags is None else bags[i])
        for i, spec in enumerate(specs)
    ]
    subsets = (
        # A stream of their own, so the bags and permutations stay as they
        # were.
        draw_bags(
            n_estimators,
            n_kept,
            n_features,
            np.random.SeedSequence([entropy, n_features, n_kept]),
        )
        if n_kept is not None and n_kept < n_features
        else None
    )
    if not bagging and subsets is None:
        return _grouped(
            members, native_transforms, feature_shuffle, random_state
        )
    alone = (
        members
        if subsets is not None
        else [
            member
            for member in members
            if not (
                supports_native_bagging
                and _is_native(member.transform, native_transforms)
            )
        ]
    )
    n_view = n_features if subsets is None else subsets[0].size
    permutations = (
        feature_permutations(n_estimators, n_view, entropy)
        if feature_shuffle and alone
        else None
    )
    groups = list(
        _one_group_per_member(
            alone,
            native_transforms,
            permutations,
            subsets,
            random_state,
            auto_tokens,
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


def cycle_tokens(tokens: Sequence[Any], message: str) -> tuple[Any, ...]:
    """The distinct tokens of a plan, in order, if the plan cycles through them.

    Upstream assigns member ``k`` the option ``k % len(options)``, so the
    distinct tokens in order run the plan only when it cycles through them.

    Args:
        tokens: The token of each member's transform, in member order.
        message: The error text, with ``{tokens}`` standing for ``tokens``
            as a tuple.

    Returns:
        The distinct tokens, in order.

    Raises:
        ValueError: With ``message`` if the plan does not cycle, as
            ``("power", "power", "none")`` would.
    """
    distinct = tuple(dict.fromkeys(tokens))
    cycled = tuple(distinct[i % len(distinct)] for i in range(len(tokens)))
    if cycled != tuple(tokens):
        raise ValueError(message.format(tokens=tuple(tokens)))
    return distinct


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
    keys = [
        "native" if _is_native(m.transform, native) else m.transform.name  # type: ignore[union-attr]  # None is native
        for m in members
    ]
    groups = []
    for index, key in enumerate(dict.fromkeys(keys)):
        chosen = tuple(
            m for m, k in zip(members, keys, strict=True) if k == key
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
    subsets: list[_typing.IntArray] | None,
    random_state: int,
    auto_tokens: Sequence[Any] | None,
) -> tuple[MemberGroup, ...]:
    """One group per member, restricted to its rows and its columns.

    Args:
        members: The members, each with its bag, if any.
        native: Uniform transform name to the model's own token.
        permutations: Every member's permutation of the columns it sees,
            indexed by member, or None to leave them in order.
        subsets: Every member's columns, indexed by member, or None for
            all.
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
                columns=None if subsets is None else subsets[member.index],
            )
        )
    return tuple(groups)
