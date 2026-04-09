"""CLI entrypoints for the Victor agent runtime.

Usage
-----
::

    python -m victor run [--config config.toml] [--ticks N]
    python -m victor inspect --db path/to/victor_memory.db [--last N]
    python -m victor replay --run-id <id> [--log-dir victor_logs]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from victor.config import RuntimeConfig, build_parser, load_config


def _cmd_run(args: object) -> None:
    """Start the agent runtime loop."""
    from victor.runtime import VictorRuntime  # noqa: PLC0415

    config_path = getattr(args, "config", None)
    cli_overrides = {
        "max_ticks": getattr(args, "max_ticks", None),
        "device": getattr(args, "device", None),
        "db_path": getattr(args, "db_path", None),
        "log_dir": getattr(args, "log_dir", None),
    }

    cfg: RuntimeConfig = load_config(
        config_path=Path(config_path) if config_path else None,
        cli_overrides=cli_overrides,
    )

    runtime = VictorRuntime(cfg)
    runtime.start()


def _cmd_inspect(args: object) -> None:
    """Print the last N memories from the SQLite store."""
    from victor.memory import MemoryStore  # noqa: PLC0415

    db_path: str = args.db_path  # type: ignore[attr-defined]
    last: int = args.last  # type: ignore[attr-defined]

    if not Path(db_path).exists():
        print(f"[inspect] Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    with MemoryStore(db_path=db_path) as store:
        memories = store.retrieve_recent(limit=last)

    if not memories:
        print("[inspect] No memories found.")
        return

    for mem in memories:
        print(
            f"[{mem.id:>6}] {mem.created_at}  kind={mem.kind:<16}  "
            f"weight={mem.emotional_weight:.2f}  tags={mem.tags}  "
            f"run={mem.run_id[:8] if mem.run_id else '–'}\n"
            f"         {mem.text[:120]}"
        )


def _cmd_replay(args: object) -> None:
    """Best-effort replay of events from JSONL logs for a given run ID."""
    run_id: str = args.run_id  # type: ignore[attr-defined]
    log_dir = Path(args.log_dir)  # type: ignore[attr-defined]

    if not log_dir.exists():
        print(f"[replay] Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    jsonl_files = sorted(log_dir.glob("events-*.jsonl"))
    if not jsonl_files:
        print(f"[replay] No JSONL event files found in {log_dir}.", file=sys.stderr)
        sys.exit(1)

    count = 0
    for filepath in jsonl_files:
        with filepath.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[replay] Skipping malformed line {lineno} in {filepath}: {exc}")
                    continue

                if event.get("run_id") != run_id:
                    continue

                ts = event.get("ts", "?")
                tick = event.get("tick", "?")
                component = event.get("component", "?")
                etype = event.get("type", "?")
                payload = event.get("payload", {})
                print(f"[{ts}] tick={tick:<5} {component:>8}.{etype:<20} {payload}")
                count += 1

    if count == 0:
        print(f"[replay] No events found for run_id={run_id!r}.")
    else:
        print(f"\n[replay] Replayed {count} events for run_id={run_id!r}.")


def main() -> None:
    """Entry point dispatching subcommands."""
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "run": _cmd_run,
        "inspect": _cmd_inspect,
        "replay": _cmd_replay,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
