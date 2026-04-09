"""Configuration loading for the Victor agent runtime.

Priority (lowest → highest): defaults → TOML file → environment variables → CLI args.
TOML parsing uses ``tomllib`` (Python ≥ 3.11 stdlib) with a fallback to the
third-party ``tomli`` package when available.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict:
    """Load a TOML file and return the parsed dict.

    Tries the stdlib ``tomllib`` first (Python ≥ 3.11), then ``tomli``.
    Raises ``ImportError`` if neither is available.
    """
    if sys.version_info >= (3, 11):
        import tomllib  # noqa: PLC0415
        with path.open("rb") as fh:
            return tomllib.load(fh)
    try:
        import tomli  # type: ignore[import]  # noqa: PLC0415
        with path.open("rb") as fh:
            return tomli.load(fh)
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "TOML parsing requires Python ≥ 3.11 (stdlib tomllib) or the "
            "'tomli' package (pip install tomli)."
        ) from exc


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    """Top-level runtime configuration."""

    # --- identity ---
    run_id: str = ""  # generated at runtime if empty

    # --- agent loop ---
    tick_interval_s: float = 1.0
    max_ticks: int = 0  # 0 = run forever

    # --- model ---
    embed_dim: int = 256
    fractal_depth: int = 3
    drop_path_prob: float = 0.1
    use_quantization: bool = False
    device: str = "cpu"

    # --- memory ---
    db_path: str = "victor_memory.db"

    # --- logging ---
    log_dir: str = "victor_logs"
    log_max_bytes: int = 10 * 1024 * 1024  # 10 MB
    log_level: str = "INFO"

    # --- policy ---
    max_actions_per_minute: int = 30
    memory_retrieve_limit: int = 5

    # --- search ---
    search_limit: int = 10

    # --- extra ---
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _apply_toml(cfg: RuntimeConfig, path: Path) -> None:
    """Merge TOML values into *cfg* in-place."""
    data = _load_toml(path)
    # Flat keys in [runtime] section
    runtime_section: dict = data.get("runtime", {})
    for key, value in runtime_section.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            cfg.extra[key] = value


def _apply_env(cfg: RuntimeConfig) -> None:
    """Merge VICTOR_* environment variables into *cfg* in-place.

    Env var names follow the pattern ``VICTOR_<UPPER_KEY>``.
    """
    mapping = {
        "VICTOR_RUN_ID": ("run_id", str),
        "VICTOR_TICK_INTERVAL_S": ("tick_interval_s", float),
        "VICTOR_MAX_TICKS": ("max_ticks", int),
        "VICTOR_EMBED_DIM": ("embed_dim", int),
        "VICTOR_FRACTAL_DEPTH": ("fractal_depth", int),
        "VICTOR_DROP_PATH_PROB": ("drop_path_prob", float),
        "VICTOR_USE_QUANTIZATION": ("use_quantization", lambda v: v.lower() in ("1", "true", "yes")),
        "VICTOR_DEVICE": ("device", str),
        "VICTOR_DB_PATH": ("db_path", str),
        "VICTOR_LOG_DIR": ("log_dir", str),
        "VICTOR_LOG_MAX_BYTES": ("log_max_bytes", int),
        "VICTOR_LOG_LEVEL": ("log_level", str),
        "VICTOR_MAX_ACTIONS_PER_MINUTE": ("max_actions_per_minute", int),
        "VICTOR_MEMORY_RETRIEVE_LIMIT": ("memory_retrieve_limit", int),
        "VICTOR_SEARCH_LIMIT": ("search_limit", int),
    }
    for env_key, (attr, cast) in mapping.items():
        raw = os.environ.get(env_key)
        if raw is not None:
            setattr(cfg, attr, cast(raw))


def load_config(
    config_path: Optional[Path] = None,
    cli_overrides: Optional[dict] = None,
) -> RuntimeConfig:
    """Build a :class:`RuntimeConfig` from file + env + CLI overrides.

    Parameters
    ----------
    config_path:
        Path to a TOML config file.  Skipped if *None* or file does not exist.
    cli_overrides:
        Dict of attribute-name → value pairs from parsed CLI args.  ``None``
        values are ignored so that unset flags don't clobber config.

    Returns
    -------
    RuntimeConfig
    """
    cfg = RuntimeConfig()

    if config_path is not None and config_path.exists():
        _apply_toml(cfg, config_path)

    _apply_env(cfg)

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)

    return cfg


# ---------------------------------------------------------------------------
# Argument parser (shared across subcommands)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser for the ``victor`` CLI."""
    parser = argparse.ArgumentParser(
        prog="victor",
        description="Victor agent runtime CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run_p = sub.add_parser("run", help="Start the agent runtime loop.")
    run_p.add_argument("--config", type=Path, default=None, metavar="FILE",
                       help="Path to TOML config file.")
    run_p.add_argument("--ticks", type=int, default=None, dest="max_ticks",
                       help="Stop after N ticks (0 = run forever).")
    run_p.add_argument("--device", type=str, default=None,
                       help="Torch device string, e.g. 'cpu' or 'cuda'.")
    run_p.add_argument("--db", type=str, default=None, dest="db_path",
                       help="SQLite memory database path.")
    run_p.add_argument("--log-dir", type=str, default=None, dest="log_dir",
                       help="Directory for JSONL event logs.")

    # --- inspect ---
    inspect_p = sub.add_parser("inspect", help="Inspect stored memories.")
    inspect_p.add_argument("--db", type=str, required=True, dest="db_path",
                           help="SQLite memory database path.")
    inspect_p.add_argument("--last", type=int, default=10, metavar="N",
                           help="Show last N memories.")

    # --- replay ---
    replay_p = sub.add_parser("replay", help="Replay events from a JSONL log.")
    replay_p.add_argument("--run-id", type=str, required=True, dest="run_id",
                          help="Run ID to replay.")
    replay_p.add_argument("--log-dir", type=str, default="victor_logs", dest="log_dir",
                          help="Directory containing JSONL event logs.")

    return parser
