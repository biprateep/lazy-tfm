# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Biprateep Dey
"""Fake upstream models, one module per registered backend.

A fake replaces what a backend needs from outside the package (the upstream
model, its network and the checkpoint download) with stand-ins that run on a
CPU in milliseconds, so that a test exercises the backend's own code with
nothing downloaded. A fake's answers depend on the query and on the member's
context alone, deterministically, so the behavioral conformance suite can
compare them across seeds, chunks, caches and pickles; and it records what
each member is handed, for the bagging tests.

Each module ``fakes/<backend>.py`` defines ``install(monkeypatch, recorder)``,
which installs the fake and returns the backend's class, and ``SETTINGS``,
the parameters that keep the backend small. A fake that needs the upstream
package itself (for its preprocessing, say) skips the test without it.
tests/lazy/test_backend_completeness.py requires a module for every
registered backend.

A fake also replaces private names of lazy itself, its seams. Each module
lists them in ``LAZY_SEAMS`` as dotted names, and installs them through
:func:`patch`, which requires the list and the patches to match;
tests/lazy/test_fakes.py requires every listed name to exist in lazy. The seams
that every fake shares are ``SHARED_SEAMS``.
"""

import collections
import importlib
import pathlib
import types

import numpy as np

FAKES_DIR = pathlib.Path(__file__).parent
"""The directory holding one fake per registered backend."""


class Recorder:
    """What each fake upstream model was handed, keyed by member."""

    def __init__(self):
        self.calls = collections.defaultdict(list)
        self.loads = []

    def add(self, key, **seen):
        self.calls[key].append(types.SimpleNamespace(**seen))


def row_ids(features):
    """The rows some prepared features came from, in their order.

    The bagging tests' features are ``row + 1000 * column``, so a row's
    smallest value names it whatever order its columns arrive in.
    """
    return np.asarray(features, dtype=float).min(axis=1).round().astype(int)


def response(queries, context):
    """A fake model's answer for each query, in (-1, 1).

    It depends on the query's own features and on the context, so it moves
    with the seed (a member's bag) and never with the other queries.

    Args:
        queries: Query features, shape ``(n_queries, n_features)``.
        context: The member's context features, shape
            ``(n_rows, n_features)``.

    Returns:
        One value per query, shape ``(n_queries,)``.
    """
    queries = np.asarray(queries, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    if not len(queries):
        return np.zeros(0)
    centre = np.nanmean(context)
    spread = np.nanstd(context) or 1.0
    with np.errstate(all="ignore"):
        signal = np.nanmean(queries, axis=1)
    return np.tanh(np.nan_to_num((signal - centre) / spread))


def bucket_logits(signal, n_buckets):
    """Logits over ``n_buckets`` buckets that peak where ``signal`` says.

    Args:
        signal: One value in (-1, 1) per query.
        n_buckets: The number of buckets.

    Returns:
        The logits, float32, shape ``(n_queries, n_buckets)``.
    """
    centres = np.linspace(-1.0, 1.0, n_buckets)
    return (-4.0 * (centres - np.asarray(signal)[:, None]) ** 2).astype(
        np.float32
    )


SHARED_SEAMS = (
    "lazy.models._weights._NETWORKS",
    "lazy.models._hub.Checkpoint.download",
)
"""The private lazy names that :func:`install` patches for every backend."""


def resolve(dotted):
    """The object named by a dotted path, importing its longest module prefix.

    Args:
        dotted: A path such as ``lazy.models.limix.LimiXBarDistribution.fit``.

    Returns:
        The object.

    Raises:
        AttributeError: If the path does not name an object.
    """
    parts = dotted.split(".")
    for end in range(len(parts), 0, -1):
        prefix = ".".join(parts[:end])
        try:
            obj = importlib.import_module(prefix)
        except ModuleNotFoundError as error:
            # Only the prefix itself being no module means "shorten it"; a
            # module that fails on a missing dependency is a real error.
            if error.name is None or not (
                prefix == error.name or prefix.startswith(error.name + ".")
            ):
                raise
            continue
        for part in parts[end:]:
            obj = getattr(obj, part)
        return obj
    raise AttributeError(f"cannot resolve {dotted}")


def patch(monkeypatch, seams, patches):
    """Replaces the private lazy names in ``patches``, which must be ``seams``.

    Args:
        monkeypatch: The test's ``monkeypatch`` fixture.
        seams: The dotted names the fake lists.
        patches: Maps each dotted name to its replacement.

    Raises:
        AssertionError: If ``patches`` and ``seams`` name different things.
    """
    assert set(patches) == set(seams), "the fake's seams are out of date"
    for dotted, value in patches.items():
        owner, _, name = dotted.rpartition(".")
        monkeypatch.setattr(resolve(owner), name, value)


def no_download(checkpoint, **kwargs):
    """Stands in for ``Checkpoint.download`` where nothing may download."""
    del checkpoint, kwargs  # Unused.
    raise AssertionError("the checkpoint was fetched")


def _download(checkpoint, *, local_files_only=False):
    """Where a checkpoint would be, without fetching it."""
    del local_files_only  # Unused: nothing is fetched.
    name = checkpoint.filename or checkpoint.repo_id.replace("/", "--")
    return pathlib.Path("/nonexistent/lazy-fakes") / name


def install(name, monkeypatch, recorder=None, **options):
    """Installs the fake for the backend registered as ``name``.

    The checkpoint download is faked for every backend, and the networks the
    fakes load go to an empty process-wide cache that the test then drops.

    Args:
        name: The backend's registered name.
        monkeypatch: The test's ``monkeypatch`` fixture.
        recorder: Where the fake records what it is handed; a fresh one if
            None.
        **options: Passed to the fake's ``install``.

    Returns:
        The backend's class.
    """
    module = importlib.import_module(f"fakes.{name}")
    patch(
        monkeypatch,
        SHARED_SEAMS,
        {
            "lazy.models._weights._NETWORKS": {},
            "lazy.models._hub.Checkpoint.download": _download,
        },
    )
    return module.install(
        monkeypatch, Recorder() if recorder is None else recorder, **options
    )


class Backend:
    """A registered backend on its fake upstream, called like its class.

    Calling it builds the backend with its fake's ``SETTINGS`` on a CPU,
    under the parameters given; any other attribute is the class's.
    """

    def __init__(self, name, monkeypatch):
        self.cls = install(name, monkeypatch)
        self.settings = {
            "device": "cpu",
            "progress": False,
            **settings(name),
        }

    def __call__(self, **params):
        return self.cls(**{**self.settings, **params})

    def __getattr__(self, name):
        return getattr(self.cls, name)


def settings(name):
    """The parameters that keep the backend ``name`` small, as a dict."""
    return dict(importlib.import_module(f"fakes.{name}").SETTINGS)
