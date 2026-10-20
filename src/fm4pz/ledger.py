# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""The ledger: what every experiment has produced, read off its files.

An experiment is one method at one training size of one regime: a search
(an Optuna journal and its exported best setting) and five production runs
(one record per seed). The ledger is never written by hand. It is rebuilt
from the files themselves, on this host and on the others whose results
``scripts/ledger.sh`` copies under ``results/remote/<host>/``, so a run that
failed or a search that stalled shows up as such whether or not anyone
noticed. For each experiment it reports the search round in use, the trial
counts, whether the best setting is exported, how many seeds have a record
made from that setting (a record made from another setting is stale), and
one status word.

Typical usage example:

  python -m fm4pz.ledger            # writes results/ledger.{csv,md}
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import dataclasses
import datetime as dt
import json
import pathlib
import re
import time
from typing import Any

import optuna
from optuna.storages import journal
import pandas as pd

from fm4pz import config

__all__ = ["MIN_COMPLETE", "OWNERS", "ROUNDS", "Entry", "build", "main"]

#: The search round each method's final results come from.
ROUNDS: dict[str, int] = {
    "flexzboost": 2,
    "cmnn": 1,
    "gpz": 2,
    "tabpfn": 2,
    "limix": 2,
    "kumo": 2,
}

#: Which session runs each method's experiments: the coordinating session
#: the classical baselines, the foundation-model session the rest.
OWNERS: dict[str, str] = {
    "flexzboost": "coordinator",
    "cmnn": "coordinator",
    "gpz": "coordinator",
    "tabpfn": "tfm-scaling",
    "limix": "tfm-scaling",
    "kumo": "tfm-scaling",
}

#: The completed trials a search should reach before it is called done.
MIN_COMPLETE: dict[str, int] = {
    "flexzboost": 40,
    "cmnn": 200,
    "gpz": 40,
    "tabpfn": 40,
    "limix": 40,
    "kumo": 40,
}

# A search whose journal has been silent this long is called stalled.
_STALL_SECONDS = 3 * 3600


@dataclasses.dataclass
class Entry:
    """One experiment's state.

    Attributes:
        method: The method.
        owner: The session that runs it (:data:`OWNERS`).
        regime: ``"representative"`` or ``"biased"``.
        n_train: The training size.
        round: The search round its results come from.
        trials: Trials in its journal, of any state.
        complete: Completed trials.
        failed: Failed trials.
        running: Trials still marked running.
        journal_age_h: Hours since the journal was last written.
        best_exported: Whether the round's best setting is exported.
        cv_loss: The best setting's cross-validated CDE loss.
        seeds_done: Seeds whose record was made from the best setting.
        seeds_stale: Seeds whose record was made from another setting.
        test_cde: Mean test CDE loss over the current records.
        host: Where the newest search files are.
        status: One of ``search-todo``, ``searching``, ``search-incomplete``,
            ``search-thin``, ``produce-todo``, ``producing``, ``done``.
    """

    method: str
    owner: str
    regime: str
    n_train: int
    round: int
    trials: int = 0
    complete: int = 0
    failed: int = 0
    running: int = 0
    journal_age_h: float | None = None
    best_exported: bool = False
    cv_loss: float | None = None
    seeds_done: int = 0
    seeds_stale: int = 0
    test_cde: float | None = None
    host: str = ""
    status: str = ""


def _newest(paths: Iterable[pathlib.Path]) -> pathlib.Path | None:
    """The most recently written of some files that exist, if any."""
    existing = [path for path in paths if path.exists()]
    return max(existing, key=lambda path: path.stat().st_mtime, default=None)


def _sources(results: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    """This host's results and every copied host's, as (name, root)."""
    roots = [("local", results)]
    remote = results / "remote"
    if remote.is_dir():
        roots += [(path.name, path) for path in sorted(remote.iterdir())]
    return roots


def _trial_counts(path: pathlib.Path, study_name: str) -> dict[str, int]:
    """Trial counts by state in a search journal."""
    study = optuna.load_study(
        study_name=study_name,
        storage=optuna.storages.JournalStorage(
            journal.JournalFileBackend(str(path))
        ),
    )
    states = [trial.state.name for trial in study.trials]
    return {
        "trials": len(states),
        "complete": states.count("COMPLETE"),
        "failed": states.count("FAIL"),
        "running": states.count("RUNNING"),
    }


def _entry(
    method: str,
    regime: str,
    n_train: int,
    sources: Sequence[tuple[str, pathlib.Path]],
) -> Entry:
    """The state of one experiment, from the newest of its files."""
    round_ = ROUNDS[method]
    suffix = ".journal" if round_ == 1 else f".r{round_}.journal"
    stem = f"{regime}_n{n_train}"
    entry = Entry(method, OWNERS[method], regime, n_train, round_)
    journals = {
        name: root / "tuning" / method / f"{stem}{suffix}"
        for name, root in sources
    }
    newest = _newest(journals.values())
    if newest is not None:
        entry.host = next(n for n, p in journals.items() if p == newest)
        entry.journal_age_h = (time.time() - newest.stat().st_mtime) / 3600
        counts = _trial_counts(newest, f"{method}-{regime}-{n_train}")
        entry.trials, entry.complete = counts["trials"], counts["complete"]
        entry.failed, entry.running = counts["failed"], counts["running"]
    best_path = _newest(
        path.with_suffix(".best.json") for path in journals.values()
    )
    best: dict[str, Any] | None = None
    if best_path is not None:
        best = json.loads(best_path.read_text())
        entry.best_exported = True
        entry.cv_loss = float(best["cv_loss"])
    losses = []
    for seed in config.SEEDS:
        record_path = _newest(
            root / "runs" / method / f"{stem}_s{seed}.json"
            for _, root in sources
        )
        if record_path is None:
            continue
        record = json.loads(record_path.read_text())
        current = best is not None and (
            record.get("params") == best["params"]
            and record.get("post") == best["post"]
        )
        if current:
            entry.seeds_done += 1
            losses.append(float(record["metrics"]["cde_loss"]))
        else:
            entry.seeds_stale += 1
    if losses:
        entry.test_cde = sum(losses) / len(losses)
    entry.status = _status(entry)
    return entry


def _status(entry: Entry) -> str:
    """The one-word status of an experiment."""
    if entry.seeds_done == len(config.SEEDS):
        return "done"
    if entry.best_exported:
        if entry.complete < MIN_COMPLETE[entry.method]:
            return "search-thin"
        return "producing" if entry.seeds_done else "produce-todo"
    if entry.trials == 0:
        return "search-todo"
    stale = (entry.journal_age_h or 0.0) * 3600 > _STALL_SECONDS
    return "search-incomplete" if stale else "searching"


def build(results: pathlib.Path | None = None) -> pd.DataFrame:
    """The ledger of every experiment, one row each.

    Args:
        results: The results directory; ``$FM4PZ_RESULTS`` by default.

    Returns:
        One row per method, regime and training size, the columns of
        :class:`Entry`.
    """
    results = config.results_dir() if results is None else results
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    sources = _sources(results)
    sizes = {
        "representative": config.REPRESENTATIVE_SIZES,
        "biased": config.BIASED_SIZES,
    }
    rows = []
    for method in ROUNDS:
        for regime, regime_sizes in sizes.items():
            for n_train in regime_sizes:
                entry = _entry(method, regime, n_train, sources)
                rows.append(dataclasses.asdict(entry))
    return pd.DataFrame(rows)


# A launcher's line for a finished run or search, and the watchdog's stop.
_DONE = re.compile(
    r"^(?P<time>\S+) done\s+(?P<method>\w+) (?P<regime>\w+) (?P<size>\d+)"
    r"(?: s(?P<seed>\d+))? exit=(?P<status>\d+)"
)
_STOP = re.compile(
    r"^(?P<time>\S+) stopping (?P<method>\w+) (?P<regime>\w+) (?P<size>\d+)"
)


def _failures(results: pathlib.Path) -> list[tuple[tuple[str, ...], str]]:
    """The failures in every log, on every host, that nothing has redone.

    A run or search is identified by its method, regime, size and seed (none
    for a search), wherever it ran. Its latest outcome across all the logs
    decides: a failure followed by a success of the same run, on any host,
    is resolved and not listed.

    Args:
        results: The results directory.

    Returns:
        One tuple (run, line) per failure no later success redid: the run's
        key (method, regime, size, seed or ``"search"``) and where and what
        was logged.
    """
    latest: dict[tuple[str, ...], tuple[dt.datetime, bool, str]] = {}
    for name, root in _sources(results):
        for log in sorted(root.glob("logs*/*.log")):
            for line in log.read_text(errors="replace").splitlines():
                event = _event(line)
                if event is None:
                    continue
                key, when, ok = event
                if key not in latest or when >= latest[key][0]:
                    latest[key] = (when, ok, f"{name}:{log.name}: {line}")
    return [
        (key, text)
        for key, (_, ok, text) in sorted(latest.items(), key=lambda kv: kv[1])
        if not ok
    ]


def _redone(key: tuple[str, ...], table: pd.DataFrame) -> bool:
    """Whether a failed run's product exists all the same.

    A search is redone when its round's best setting is exported, and a
    production run when its seed has a record made from that setting
    (a search finished or a run made by hand leaves no line in the logs).
    """
    method, regime, size, seed = key
    if method not in ROUNDS:
        return False
    row = table[
        (table["method"] == method)
        & (table["regime"] == regime)
        & (table["n_train"] == int(size))
    ]
    if row.empty:
        return False
    if seed == "search":
        return bool(row["best_exported"].iloc[0])
    return bool(row["seeds_done"].iloc[0] == len(config.SEEDS))


def _event(
    line: str,
) -> tuple[tuple[str, ...], dt.datetime, bool] | None:
    """A log line's run, time and outcome; None for any other line."""
    done = _DONE.match(line)
    if done is not None:
        key = (
            done["method"],
            done["regime"],
            done["size"],
            done["seed"] or "search",
        )
        ok = done["status"] == "0"
        return key, dt.datetime.fromisoformat(done["time"]), ok
    stop = _STOP.match(line)
    if stop is not None:
        key = (stop["method"], stop["regime"], stop["size"], "search")
        return key, dt.datetime.fromisoformat(stop["time"]), False
    return None


def _markdown(table: pd.DataFrame, failures: Sequence[str]) -> str:
    """The ledger as a Markdown page: a summary, the table, the failures."""
    counts = table.groupby(["method", "status"]).size().unstack(fill_value=0)
    shown = table.copy()
    for column in ("journal_age_h", "cv_loss", "test_cde"):
        shown[column] = shown[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    parts = [
        f"# Ledger ({time.strftime('%Y-%m-%d %H:%M %Z')})",
        "",
        "## Status by method",
        "",
        counts.to_markdown(),
        "",
        "## Experiments",
        "",
        shown.to_markdown(index=False),
        "",
        "## Failures in the logs that nothing has redone",
        "",
        *([f"- `{line}`" for line in failures] or ["None."]),
        "",
    ]
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    """Writes ``ledger.csv`` and ``ledger.md`` under the results directory.

    Args:
        argv: Unused; there are no options.

    Returns:
        The exit status.
    """
    del argv  # Unused: the ledger has no options.
    results = config.results_dir()
    table = build(results)
    table.to_csv(results / "ledger.csv", index=False)
    failures = [
        text for key, text in _failures(results) if not _redone(key, table)
    ]
    (results / "ledger.md").write_text(_markdown(table, failures))
    print(table.groupby(["method", "status"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
