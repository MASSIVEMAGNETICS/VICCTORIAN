"""Structured JSONL event logger with daily file naming and size rotation.

Every event is a single JSON line with the fields:
  ts        — UTC ISO-8601 timestamp
  run_id    — unique identifier for the current agent run
  tick      — agent tick counter
  component — e.g. "runtime", "memory", "model", "policy"
  type      — e.g. "TICK_START", "ACTION", "MEMORY_STORE"
  payload   — arbitrary dict of additional data
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# EventLogger
# ---------------------------------------------------------------------------

class EventLogger:
    """Thread-safe JSONL event logger with daily file naming and size rotation.

    Parameters
    ----------
    log_dir:
        Directory where ``events-YYYY-MM-DD[.N].jsonl`` files are written.
        Created automatically if it does not exist.
    run_id:
        Unique identifier for the current agent run, embedded in every event.
    max_bytes:
        Maximum size of a single log file before it is rotated.  Defaults to
        10 MB.
    """

    def __init__(
        self,
        log_dir: str | Path,
        run_id: str,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._fh: Any = None          # current open file handle
        self._current_path: Path | None = None
        self._current_day: str = ""   # YYYY-MM-DD of current file
        self._open_file()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _file_path(self, day: str, seq: int = 0) -> Path:
        suffix = f".{seq}" if seq > 0 else ""
        return self._log_dir / f"events-{day}{suffix}.jsonl"

    def _open_file(self) -> None:
        """Open (or rotate to) the appropriate log file."""
        day = self._today()
        seq = 0
        path = self._file_path(day, seq)

        # Find the latest sequence file for today
        while path.exists() and path.stat().st_size >= self._max_bytes:
            seq += 1
            path = self._file_path(day, seq)

        if self._fh is not None:
            self._fh.close()

        self._fh = path.open("a", encoding="utf-8")
        self._current_path = path
        self._current_day = day

    def _rotate_if_needed(self) -> None:
        """Rotate the log file when day changes or size limit is reached."""
        day = self._today()
        size = self._current_path.stat().st_size if self._current_path else 0
        if day != self._current_day or size >= self._max_bytes:
            self._open_file()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        *,
        tick: int,
        component: str,
        type: str,  # noqa: A002  (shadows built-in; intentional field name)
        payload: dict | None = None,
    ) -> None:
        """Write a structured event to the current JSONL file.

        Parameters
        ----------
        tick:
            Current agent tick counter.
        component:
            Subsystem that produced the event (e.g. ``"runtime"``).
        type:
            Event type label (e.g. ``"TICK_START"``).
        payload:
            Arbitrary extra data included verbatim.
        """
        event: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self._run_id,
            "tick": tick,
            "component": component,
            "type": type,
            "payload": payload or {},
        }
        line = json.dumps(event, ensure_ascii=False, default=str)

        with self._lock:
            self._rotate_if_needed()
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        """Flush and close the current log file."""
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None

    def current_path(self) -> Path | None:
        """Return the path of the currently active log file."""
        return self._current_path

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
