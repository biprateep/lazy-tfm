# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The static contract a backend must meet before it can be registered.

:func:`lazy.models.registry.register` runs :func:`check_static` on every
backend, so none can join the registry without the uniform features: the
parameters (keyword-only, with the shared defaults), a coherent declaration of
what its model does natively, and a pinned checkpoint. The behavioural half of
the contract -- that the features also *work* -- is the conformance test
suite, which runs over everything registered.

Typical usage example:

  problems = _conformance.problems(MyBackend, "mine")
  _conformance.check_static(MyBackend, "mine")  # raises TypeError
"""

from __future__ import annotations

import inspect

from lazy.models import _ensemble
from lazy.models import _hub
from lazy.models import _transforms

__all__ = ["check_static", "problems"]


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
    found.extend(_parameter_problems(cls))
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
    found = problems(cls, name)
    if found:
        details = "\n  - ".join(found)
        raise TypeError(f"{cls.__name__} cannot be registered:\n  - {details}")


def _parameter_problems(cls: type) -> list[str]:
    """The uniform parameters, keyword-only, with the shared defaults."""
    parameters = inspect.signature(cls.__init__).parameters  # type: ignore[misc]  # a class's own __init__
    found = []
    if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        found.append("__init__ takes **kwargs, which hides parameters")
    for name in _ensemble.UNIFORM_PARAMS:
        parameter = parameters.get(name)
        if parameter is None:
            found.append(f"__init__ lacks the uniform parameter {name!r}")
        elif parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            found.append(f"{name!r} must be keyword-only")
    for name, default in _ensemble.UNIFORM_DEFAULTS.items():
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
    if cls.member_combination not in ("mixture", "quantile_average"):
        found.append(
            f"member_combination {cls.member_combination!r} is not known"
        )
    if cls.member_combination == "quantile_average" and cls.native_output != (
        "quantiles"
    ):
        found.append("quantile_average needs native_output='quantiles'")
    for flag in (
        "supports_native_bagging",
        "exact_chunking",
        "chunks_queries",
        "has_softmax",
        "native_outlier_clipping",
        "cpu_friendly",
    ):
        if not isinstance(getattr(cls, flag), bool):
            found.append(f"{flag} must be a bool")
    unknown_versions = sorted(
        set(cls.cpu_friendly_versions)
        - set(_hub.list_versions(cls.backend or ""))
    )
    if unknown_versions:
        found.append(f"cpu_friendly_versions names unknown {unknown_versions}")
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
