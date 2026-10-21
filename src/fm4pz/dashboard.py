# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Biprateep Dey
"""A live dashboard of the ledger, served over the tailnet.

Serves one page that draws ``results/ledger.json`` and polls it, and rebuilds
the ledger with ``scripts/ledger.sh`` every few minutes and whenever the page's
Refresh button asks. The server listens on one address only, by default this
host's Tailscale address, so only devices on the tailnet can open it.

Routes:

  GET  /              the page
  GET  /ledger.json   the ledger as ``fm4pz.ledger`` last wrote it
  GET  /status        whether a rebuild is running, and how the last one went
  POST /refresh       starts a rebuild unless one is running

Typical usage example:

  python -m fm4pz.dashboard --host 100.73.83.68 --port 8765
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import datetime as dt
import http
from http import server
import json
import pathlib
import subprocess
import threading
from typing import Any

from fm4pz import config

__all__ = ["Rebuilder", "main"]

_PAGE = pathlib.Path(__file__).with_name("dashboard.html")
_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _now() -> str:
    """This moment as ISO 8601 with the local offset."""
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Rebuilder:
    """Runs ``scripts/ledger.sh`` one rebuild at a time.

    Attributes:
        running: Whether a rebuild is running.
        started: When the latest rebuild started, ISO 8601.
        finished: When the latest rebuild finished, ISO 8601.
        error: The tail of the latest rebuild's output if it failed.
    """

    def __init__(self, timeout: float = 600.0) -> None:
        """Initializes a rebuilder.

        Args:
            timeout: Seconds after which a rebuild is stopped.
        """
        self._lock = threading.Lock()
        self._timeout = timeout
        self.running = False
        self.started: str | None = None
        self.finished: str | None = None
        self.error: str | None = None

    def start(self) -> bool:
        """Starts a rebuild in the background.

        Returns:
            True if a rebuild started, False if one was already running.
        """
        with self._lock:
            if self.running:
                return False
            self.running = True
            self.started = _now()
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        """Runs one rebuild and records how it went."""
        error = None
        try:
            done = subprocess.run(
                [str(_ROOT / "scripts" / "ledger.sh")],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
            if done.returncode != 0:
                error = (done.stdout + done.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            error = f"ledger.sh ran past {self._timeout:.0f} s"
        with self._lock:
            self.running = False
            self.finished = _now()
            self.error = error

    def status(self) -> dict[str, Any]:
        """The rebuild state, as a JSON-ready dict."""
        with self._lock:
            return {
                "running": self.running,
                "started": self.started,
                "finished": self.finished,
                "error": self.error,
            }


def _handler(
    rebuilder: Rebuilder,
    ledger: pathlib.Path,
) -> type[server.BaseHTTPRequestHandler]:
    """A request handler class bound to one rebuilder and ledger file."""

    class Handler(server.BaseHTTPRequestHandler):
        """Serves the page, the ledger and the rebuild routes."""

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, value: Any) -> None:
            body = json.dumps(value).encode()
            self._send(status, body, "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                body = _PAGE.read_bytes()
                self._send(http.HTTPStatus.OK, body, "text/html; charset=utf-8")
            elif path == "/ledger.json":
                if not ledger.exists():
                    self._json(
                        http.HTTPStatus.NOT_FOUND, {"error": "no ledger"}
                    )
                    return
                self._send(
                    http.HTTPStatus.OK, ledger.read_bytes(), "application/json"
                )
            elif path == "/status":
                # The page's own version, so an open page reloads after an edit.
                page = int(_PAGE.stat().st_mtime)
                self._json(
                    http.HTTPStatus.OK, rebuilder.status() | {"page": page}
                )
            else:
                self._json(http.HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/refresh":
                self._json(http.HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            started = rebuilder.start()
            status = rebuilder.status() | {"started_now": started}
            self._json(http.HTTPStatus.ACCEPTED, status)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - name fixed by http.server.
            del format, args  # Unused: polling would flood the log.

    return Handler


def _every(rebuilder: Rebuilder, seconds: float) -> None:
    """Starts a rebuild every so many seconds, forever."""
    stop = threading.Event()
    while not stop.wait(seconds):
        rebuilder.start()


def main(argv: Sequence[str] | None = None) -> int:
    """Serves the dashboard until stopped.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` by default.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port.")
    parser.add_argument(
        "--every",
        type=float,
        default=600.0,
        help="Seconds between scheduled rebuilds; 0 for none.",
    )
    args = parser.parse_args(argv)
    rebuilder = Rebuilder()
    ledger = config.results_dir() / "ledger.json"
    if args.every > 0:
        threading.Thread(
            target=_every, args=(rebuilder, args.every), daemon=True
        ).start()
    rebuilder.start()
    httpd = server.ThreadingHTTPServer(
        (args.host, args.port), _handler(rebuilder, ledger)
    )
    print(f"Serving on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
