# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""The test fakes against the upstream classes they stand in for.

Each check needs the upstream package, so it runs in the backends job. The
fakes of TabFM and Kumo patch or subclass upstream's own classes, which fails
as soon as a name they use goes away, and LimiX's source is not on PyPI; so
the stand-in classes of TabPFN and TabICL are the ones checked here. The
TabPFN fake also binds its constructor's arguments to upstream's signature.
"""

import inspect

from fakes import tabicl as fake_tabicl
from fakes import tabpfn as fake_tabpfn
import pytest

STAND_INS = {
    "tabpfn.TabPFNRegressor": fake_tabpfn.Regressor,
    "tabicl.TabICLRegressor": fake_tabicl.Regressor,
    "tabicl._model.tabicl.TabICL": fake_tabicl.Network,
}
"""Each fake class, by the dotted name of the upstream class it replaces."""

FAKE_ONLY = {"cached_context"}
"""Helpers a fake defines that upstream has no counterpart for."""


def _upstream(dotted):
    """The upstream class named ``dotted``, or a skip without its package."""
    module, _, name = dotted.rpartition(".")
    return getattr(pytest.importorskip(module), name)


def _parameters(function):
    """The named parameters of ``function``, and whether it takes any keyword.

    Returns:
        The names, without ``self``, and True if it has ``**kwargs``.
    """
    parameters = inspect.signature(function).parameters.values()
    names = {
        p.name
        for p in parameters
        if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    return names, any(p.kind == p.VAR_KEYWORD for p in parameters)


@pytest.mark.parametrize("dotted", sorted(STAND_INS))
def test_every_method_of_a_fake_exists_upstream_with_its_parameters(dotted):
    upstream = _upstream(dotted)
    for name, member in vars(STAND_INS[dotted]).items():
        if name in FAKE_ONLY or name == "__init__" or not callable(member):
            continue
        real = getattr(upstream, name, None)
        assert real is not None, f"{dotted} has no {name}"
        accepted, any_keyword = _parameters(real)
        missing = set() if any_keyword else _parameters(member)[0] - accepted
        assert not missing, f"{dotted}.{name} takes no {sorted(missing)}"
