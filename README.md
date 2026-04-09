# thanos — Victor Agent Runtime

> **Frontier, production-grade, headless AI agent runtime** built on a deterministic tick loop, persistent SQLite memory, structured JSONL telemetry, and an adaptive fractal neural core.

---

## Architecture

```
Sense → Retrieve → Think → Decide → Act → Reflect → Store
```

| Phase | Component | Description |
|-------|-----------|-------------|
| Sense | `runtime.py` | Build observation vector from ego state + noise |
| Retrieve | `memory.py` | Pull recent memories from SQLite |
| Think | `model.py` | Forward pass through `FractalLayerV2` |
| Decide | `policy.py` | Planner + rate limiter → `Action` |
| Act | `runtime.py` | Execute: LOG_NOTE / STORE_MEMORY / ASK_USER / NOOP |
| Reflect | `runtime.py` | EMA ego-state update |
| Store | `memory.py` | Persist tick summary every 20 ticks |

### Package layout

```
victor/
├── __init__.py       # package metadata
├── __main__.py       # CLI entrypoints (run / inspect / replay)
├── config.py         # TOML + env + CLI config loading
├── logger.py         # Structured JSONL event logger with rotation
├── memory.py         # SQLite persistent memory store
├── model.py          # FractalLayerV2 neural core (torch optional)
├── policy.py         # Action space + rate limiter + planner
└── runtime.py        # Headless deterministic tick loop

tests/
├── test_logger.py    # JSONL formatting + rotation + thread-safety
├── test_memory.py    # SQLite CRUD + search + relevance scoring
└── test_model.py     # Model shape / telemetry / drop-path (skipped if no torch)

config.toml           # Sample configuration file
pyproject.toml        # Project metadata
```

---

## Quick Start

### Requirements

- Python ≥ 3.11 (uses stdlib `tomllib`, `sqlite3`, `dataclasses`, `signal`)
- `torch ≥ 2.0` (optional — model runs as a passthrough stub if absent)

### Install

```bash
pip install -e .
# Optional GPU/neural core:
pip install torch
```

### Run the agent

```bash
# Using the default config
python -m victor run

# Using a custom config file
python -m victor run --config config.toml

# Fixed number of ticks
python -m victor run --ticks 50

# Override device and DB path
python -m victor run --device cpu --db /tmp/victor.db
```

### Inspect memories

```bash
python -m victor inspect --db victor_memory.db --last 20
```

### Replay events from a past run

```bash
python -m victor replay --run-id <uuid> --log-dir victor_logs
```

---

## Configuration

All configuration lives in `config.toml` (or overridable via env vars / CLI flags).

| Key | Default | Description |
|-----|---------|-------------|
| `tick_interval_s` | `1.0` | Seconds between ticks |
| `max_ticks` | `0` | 0 = run forever |
| `embed_dim` | `256` | Embedding dimension |
| `fractal_depth` | `3` | Max recursive blocks |
| `drop_path_prob` | `0.1` | Block drop probability |
| `use_quantization` | `false` | CPU-only int8 quantization |
| `device` | `"cpu"` | Torch device |
| `db_path` | `"victor_memory.db"` | SQLite database path |
| `log_dir` | `"victor_logs"` | JSONL event log directory |
| `log_max_bytes` | `10485760` | Log rotation size (10 MB) |
| `max_actions_per_minute` | `30` | Action rate limit |
| `memory_retrieve_limit` | `5` | Memories per tick |

**Priority (lowest → highest):** defaults → `config.toml` → `VICTOR_*` env vars → CLI flags.

---

## Event Log Format

Every event in `victor_logs/events-YYYY-MM-DD.jsonl` is one JSON line:

```json
{
  "ts": "2024-01-15T12:34:56.789012+00:00",
  "run_id": "3f7a1c2d-...",
  "tick": 42,
  "component": "policy",
  "type": "ACTION_STORE_MEMORY",
  "payload": { "memory_id": 7, "text": "Tick 42: gate fired..." }
}
```

Files rotate when they exceed `log_max_bytes` or on day change:
`events-2024-01-15.jsonl`, `events-2024-01-15.1.jsonl`, …

---

## Memory Store

The SQLite schema (`victor_memory.db`) stores:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `kind` | TEXT | Category: `observation`, `reflection`, `directive`, `tick_summary` |
| `text` | TEXT | Free-form content |
| `tags` | TEXT | Comma-separated tags |
| `emotional_weight` | REAL | Salience in [0, 1] |
| `links` | TEXT | Comma-separated related memory IDs |
| `created_at` | TEXT | UTC ISO-8601 timestamp |
| `run_id` | TEXT | Run that created this memory |

### Python API

```python
from victor.memory import MemoryStore

with MemoryStore(db_path="victor_memory.db", run_id="my-run") as store:
    mid = store.store_memory(
        kind="observation",
        text="The sky is clear today.",
        tags=["weather"],
        emotional_weight=0.6,
    )
    recent = store.retrieve_recent(limit=10)
    results = store.search_text("sky clear", limit=5)
```

---

## Neural Core — FractalLayerV2

Production fixes over the prototype:

| Issue | Fix |
|-------|-----|
| Activation checkpointing in inference | Only applied `if self.training and torch.is_grad_enabled()` |
| Python `random` in drop-path (non-reproducible) | Uses `torch.rand(1)` tied to torch RNG seed |
| Quantization on CUDA (unsupported) | `ValueError` raised at build time with clear message |
| Gate telemetry unavailable | `layer.telemetry` exposes trigger stats + gate mask after every forward pass |

```python
from victor.model import build_model
import torch

model = build_model(input_dim=256, depth=3, drop_path_prob=0.1, device="cpu")
model.eval()

x = torch.randn(1, 256)
with torch.no_grad():
    out = model(x)

print(model.telemetry)
# ModelTelemetry(trigger_mean=0.512, gate_fired=True, adaptive_depth_used=2, ...)
```

---

## Tests

```bash
pip install pytest
pytest tests/ -v
```

Model tests are automatically skipped when `torch` is not installed.

---

## Action Space

| Action | Trigger |
|--------|---------|
| `NOOP` | Default / rate-limited fallback |
| `LOG_NOTE` | Every 10 ticks (heartbeat) |
| `STORE_MEMORY` | Gate fired + trigger_mean > 0.75 |
| `ASK_USER` | Every 50 ticks |

All actions pass through a token-bucket rate limiter (`max_actions_per_minute`).