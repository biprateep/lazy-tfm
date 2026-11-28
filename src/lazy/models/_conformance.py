# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The static contract a backend must meet before it can be registered.

:func:`lazy.models.registry.register` runs :func:`check_static` on every
backend, so none can join the registry without the uniform features: the
parameters (keyword-only, with the shared defaults), a coherent declaration of
what its model does natively, and a pinned checkpoint. The behavioral half of
the contract -- that the features also *work* -- is the conformance test
suite, which runs over everything registered.

Typical usage example:

  problems = _conformance.problems(MyBackend, "mine")
  _conformance.check_static(MyBackend, "mine")  # raises TypeError
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
from typing import Any

from lazy.models import _ensemble
from lazy.models import _hub
from lazy.models import _transforms

__all__ = ["check_static", "parameter_problems", "problems", "refuse"]


def problems(cls: type, name: str) -> list[str]:
    """Lists every way ``cls`` falls short of the backend contract.

    Args:
        cls: The candidate backend class.
        name: The name it would be registered under.

    Returns:
        One message per problem; empty when the class conforms.
    """
    if not (
        isinstance(cls, type)
        and issubclass(cls, _ensemble.ContextEnsembleEstimator)
    ):
        return [f"{cls!r} is not a ContextEnsembleEstimator subclass"]
    found = []
    if cls.backend != name:
        found.append(f"backend = {cls.backend!r}, registered as {name!r}")
    if name not in _hub.DEFAULT_VERSIONS:
        found.append(f"no pinned checkpoint: add {name!r} to DEFAULT_VERSIONS")
    found.extend(
        parameter_problems(
            cls, _ensemble.UNIFORM_PARAMS, _ensemble.UNIFORM_DEFAULTS
        )
    )
    found.extend(_declaration_problems(cls))
    return found


def check_static(cls: type, name: str) -> None:
    """Raises unless ``cls`` meets the backend contract.

    Args:
        cls: The candidate backend class.
        name: The name it would be registered under.

    Raises:
        TypeError: If it falls short, listing every problem.
    """
    refuse(cls, problems(cls, name))


def refuse(cls: type, found: Sequence[str]) -> None:
    """Raises if a class has problems that keep it out of a registry.

    Args:
        cls: The candidate class.
        found: Its problems, one message each.

    Raises:
        TypeError: If there are any, listing every one.
    """
    if found:
        details = "\n  - ".join(found)
        name = getattr(cls, "__name__", repr(cls))
        raise TypeError(f"{name} cannot be registered:\n  - {details}")


def parameter_problems(
    cls: type, names: Sequence[str], defaults: Mapping[str, Any]
) -> list[str]:
    """The ways a constructor falls short of a set of shared parameters.

    Args:
        cls: The candidate class.
        names: The parameters its constructor must take, keyword-only.
        defaults: The default each of some of them must have.

    Returns:
        One message per problem: ``**kwargs``, a parameter that is missing,
        positional or without a default (so that ``cls()`` cannot build
        it), or another default than the shared one.
    """
    parameters = inspect.signature(cls.__init__).parameters  # type: ignore[misc]  # a class's own __init__
    found = []
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        found.append("__init__ takes **kwargs, which hides parameters")
    for name in names:
        parameter = parameters.get(name)
        if parameter is None:
            found.append(f"__init__ lacks the uniform parameter {name!r}")
        elif parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            found.append(f"{name!r} must be keyword-only")
    for parameter in list(parameters.values())[1:]:
        if parameter.default is inspect.Parameter.empty and (
            parameter.kind
            not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
        ):
            found.append(f"{parameter.name!r} has no default")
    for name, default in defaults.items():
        parameter = parameters.get(name)
        if parameter is not None and parameter.default != default:
            found.append(
                f"{name!r} defaults to {parameter.default!r}, not {default!r}"
            )
    return found


def _declaration_problems(
    cls: type[_ensemble.ContextEnsembleEstimator],
) -> list[str]:
    """A coherent declaration of the model's native capabilities."""
    found = []
    if not cls.display_name or not cls.extra:
        found.append("display_name and extra must be set")
    if cls.native_output not in ("histogram", "quantiles"):
        found.append(f"native_output {cls.native_output!r} is not known")
    found.extend(_combination_problems(cls))
    # Removed from the contract: a backend that still sets one would be
    # ignored without a word.
    for removed, instead in (
        ("exact_chunking", "every backend's chunking is checked to rounding"),
        ("member_trim_fraction", "override _member_trim_fraction instead"),
    ):
        if hasattr(cls, removed):
            found.append(f"{removed} was removed; {instead}")
    for flag in (
        "supports_native_bagging",
        "chunks_queries",
        "has_softmax",
        "native_outlier_clipping",
        "native_feature_shuffle",
        "native_categorical",
        "cpu_friendly",
    ):
        if not isinstance(getattr(cls, flag), bool):
            found.append(f"{flag} must be a bool")
    for listed in ("cpu_friendly_versions", "ranking_versions"):
        unknown_versions = sorted(
            set(getattr(cls, listed))
            - set(_hub.list_versions(cls.backend or ""))
        )
        if unknown_versions:
            found.append(f"{listed} names unknown {unknown_versions}")
    if True not in cls.kv_cache_modes or False not in cls.kv_cache_modes:
        found.append("kv_cache_modes must include True and False")
    vocabulary = set(_transforms.BASE_TRANSFORMS)
    vocabulary |= {name + "+original" for name in _transforms.BASE_TRANSFORMS}
    unknown = sorted(set(cls.native_transforms) - vocabulary)
    if unknown:
        found.append(f"native_transforms has unknown names {unknown}")
    if cls.native_transforms and "none" not in cls.native_transforms:
        found.append("native_transforms must include 'none' (for scaffolds)")
    return found


def _combination_problems(
    cls: type[_ensemble.ContextEnsembleEstimator],
) -> list[str]:
    """A member combination the model's outputs and hooks support."""
    found = []
    if cls.member_combination not in (
        "mixture",
        "quantile_average",
        "quantile_trimmed_average",
    ):
        found.append(
            f"member_combination {cls.member_combination!r} is not known"
        )
    if cls.member_combination.startswith("quantile") and (
        cls.native_output != "quantiles"
    ):
        found.append(
            f"{cls.member_combination} needs native_output='quantiles'"
        )
    # The default _predict_members only raises.
    inherited = vars(_ensemble.ContextEnsembleEstimator)["_predict_members"]
    if (
        cls.member_combination == "quantile_trimmed_average"
        and cls._predict_members is inherited
    ):
        found.append("quantile_trimmed_average needs _predict_members")
    trim = vars(_ensemble.ContextEnsembleEstimator)["_member_trim_fraction"]
    if (
        cls.member_combination != "quantile_trimmed_average"
        and cls._member_trim_fraction is not trim
    ):
        found.append("_member_trim_fraction needs quantile_trimmed_average")
    return found
