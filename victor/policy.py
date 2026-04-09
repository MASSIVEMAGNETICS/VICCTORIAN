"""Policy / planner for the Victor agent runtime.

Converts model telemetry + retrieved memories into a structured :class:`Action`
with a token-bucket rate limiter to cap actions per minute.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from victor.memory import Memory
    from victor.model import ModelTelemetry


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class ActionKind(str, Enum):
    """Enumeration of all valid agent actions."""

    NOOP = "NOOP"                 # Do nothing
    LOG_NOTE = "LOG_NOTE"         # Emit an event to the logger
    STORE_MEMORY = "STORE_MEMORY" # Persist a new memory record
    ASK_USER = "ASK_USER"        # Print a message / question to stdout


@dataclass
class Action:
    """A single planned agent action.

    Attributes
    ----------
    kind:
        Which action to perform.
    text:
        Human-readable message or note associated with the action.
    tags:
        Optional semantic tags for STORE_MEMORY actions.
    emotional_weight:
        Salience in [0, 1] used when storing a memory.
    meta:
        Additional context passed to the executor.
    """

    kind: ActionKind
    text: str = ""
    tags: list[str] = field(default_factory=list)
    emotional_weight: float = 0.5
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rate limiter (token-bucket)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple token-bucket rate limiter.

    Parameters
    ----------
    max_per_minute:
        Maximum number of tokens (actions) allowed per 60-second window.
    """

    def __init__(self, max_per_minute: int = 30) -> None:
        self._max = max_per_minute
        self._tokens: float = float(max_per_minute)
        self._last_refill: float = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        # Refill rate: max_per_minute tokens per 60 seconds
        self._tokens = min(
            float(self._max),
            self._tokens + elapsed * self._max / 60.0,
        )
        self._last_refill = now

    def acquire(self) -> bool:
        """Attempt to consume one token.

        Returns
        -------
        bool
            ``True`` if a token was available; ``False`` if rate-limited.
        """
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class Planner:
    """Convert model output + retrieved memories into a structured action.

    Parameters
    ----------
    max_actions_per_minute:
        Token-bucket capacity; actions beyond this rate are converted to NOOP.
    """

    def __init__(self, max_actions_per_minute: int = 30) -> None:
        self._rate = RateLimiter(max_actions_per_minute)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        tick: int,
        telemetry: "ModelTelemetry",
        memories: list["Memory"],
    ) -> Action:
        """Decide the next action given model telemetry and recent memories.

        Decision heuristics (in priority order):
        1. Every 10 ticks, emit a ``LOG_NOTE`` with model telemetry summary.
        2. If the gate fired and trigger is high (>0.75), store a reflection
           memory.
        3. Every 50 ticks, ask the user for input.
        4. Otherwise, NOOP.

        The chosen action is rate-limited; if the bucket is exhausted the
        method falls back to NOOP.

        Parameters
        ----------
        tick:
            Current tick counter.
        telemetry:
            Snapshot from the model's most recent forward pass.
        memories:
            Recently retrieved memories used to colour the reflection text.

        Returns
        -------
        Action
        """
        candidate = self._decide(tick, telemetry, memories)
        if candidate.kind is ActionKind.NOOP:
            return candidate
        if not self._rate.acquire():
            return Action(kind=ActionKind.NOOP, meta={"reason": "rate_limited"})
        return candidate

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decide(
        self,
        tick: int,
        telemetry: "ModelTelemetry",
        memories: list["Memory"],
    ) -> Action:
        # --- Periodic ASK_USER (every 50 ticks, starting at tick 50) ---
        if tick > 0 and tick % 50 == 0:
            return Action(
                kind=ActionKind.ASK_USER,
                text=f"[Victor @ tick {tick}] How am I doing? Any directives?",
                meta={"tick": tick},
            )

        # --- High trigger + gate fired → store reflection ---
        if telemetry.gate_fired and telemetry.trigger_mean > 0.75:
            memory_snippet = (
                f" Context: {memories[0].text[:80]!r}" if memories else ""
            )
            return Action(
                kind=ActionKind.STORE_MEMORY,
                text=(
                    f"Tick {tick}: gate fired at trigger_mean="
                    f"{telemetry.trigger_mean:.3f}, depth="
                    f"{telemetry.adaptive_depth_used}.{memory_snippet}"
                ),
                tags=["reflection", "gate_fired"],
                emotional_weight=min(1.0, telemetry.trigger_mean),
                meta={"telemetry": _tel_dict(telemetry)},
            )

        # --- Periodic LOG_NOTE ---
        if tick % 10 == 0:
            return Action(
                kind=ActionKind.LOG_NOTE,
                text=f"Tick {tick} heartbeat.",
                meta={"telemetry": _tel_dict(telemetry), "memory_count": len(memories)},
            )

        return Action(kind=ActionKind.NOOP)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tel_dict(tel: "ModelTelemetry") -> dict:
    """Serialize telemetry to a plain dict (no tensors)."""
    return {
        "trigger_mean": round(tel.trigger_mean, 4),
        "trigger_max": round(tel.trigger_max, 4),
        "gate_fired": tel.gate_fired,
        "adaptive_depth_used": tel.adaptive_depth_used,
        "blocks_skipped": tel.blocks_skipped,
    }
