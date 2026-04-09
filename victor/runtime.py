"""Victor agent runtime — headless deterministic tick loop.

Tick pipeline
-------------
Sense → Retrieve → Think → Decide → Act → Reflect → Store

Each phase is logged via the structured :class:`~victor.logger.EventLogger`.
"""

from __future__ import annotations

import signal
import time
import uuid
from typing import Optional

from victor.config import RuntimeConfig
from victor.logger import EventLogger
from victor.memory import Memory, MemoryStore
from victor.model import ModelTelemetry, build_model
from victor.policy import Action, ActionKind, Planner

# Torch is optional — import guard
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class VictorRuntime:
    """Headless agent runtime.

    Parameters
    ----------
    cfg:
        :class:`~victor.config.RuntimeConfig` instance.
    """

    def __init__(self, cfg: RuntimeConfig) -> None:
        self._cfg = cfg
        self._run_id: str = cfg.run_id or str(uuid.uuid4())
        self._tick: int = 0
        self._running: bool = False

        # Subsystems
        self._logger = EventLogger(
            log_dir=cfg.log_dir,
            run_id=self._run_id,
            max_bytes=cfg.log_max_bytes,
        )
        self._memory = MemoryStore(db_path=cfg.db_path, run_id=self._run_id)
        self._model = build_model(
            input_dim=cfg.embed_dim,
            depth=cfg.fractal_depth,
            drop_path_prob=cfg.drop_path_prob,
            use_quantization=cfg.use_quantization,
            device=cfg.device,
        )
        self._planner = Planner(max_actions_per_minute=cfg.max_actions_per_minute)

        # Ego state vector (CPU float)
        if _TORCH_AVAILABLE:
            self._device = torch.device(cfg.device)
            self._ego: Optional["torch.Tensor"] = torch.zeros(cfg.embed_dim, device=self._device)
        else:
            self._device = None
            self._ego = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the agent loop until stopped or max_ticks is reached."""
        self._running = True

        # Register graceful shutdown on SIGINT / SIGTERM
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self._log("runtime", "BOOT", {
            "run_id": self._run_id,
            "config": {
                "embed_dim": self._cfg.embed_dim,
                "fractal_depth": self._cfg.fractal_depth,
                "device": self._cfg.device,
                "tick_interval_s": self._cfg.tick_interval_s,
                "max_ticks": self._cfg.max_ticks,
            },
        })

        # Seed initial memories
        self._memory.store_memory(
            kind="directive",
            text="Victor runtime started.",
            tags=["system", "boot"],
            emotional_weight=0.9,
        )

        try:
            while self._running:
                self._tick_once()
                if self._cfg.max_ticks > 0 and self._tick >= self._cfg.max_ticks:
                    self._running = False
                    break
                time.sleep(self._cfg.tick_interval_s)
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the runtime to stop after the current tick."""
        self._running = False

    # ------------------------------------------------------------------
    # Tick pipeline
    # ------------------------------------------------------------------

    def _tick_once(self) -> None:
        """Execute one full Sense→Retrieve→Think→Decide→Act→Reflect→Store cycle."""
        self._tick += 1
        tick = self._tick

        self._log("runtime", "TICK_START", {"tick": tick})

        # 1. Sense: build an observation vector
        obs = self._sense(tick)

        # 2. Retrieve: pull recent memories
        memories = self._retrieve()

        # 3. Think: forward pass through the model
        telemetry = self._think(obs)

        # 4. Decide: planner selects action
        action = self._decide(tick, telemetry, memories)

        # 5. Act: execute the action
        self._act(action)

        # 6. Reflect: ego update
        self._reflect(obs, telemetry)

        # 7. Store: persist a tick summary memory
        self._store_tick(tick, telemetry, action)

        self._log("runtime", "TICK_END", {
            "tick": tick,
            "action_kind": action.kind.value,
            "trigger_mean": round(telemetry.trigger_mean, 4),
        })

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _sense(self, tick: int) -> Optional["torch.Tensor"]:
        """Build a synthetic observation tensor for the current tick.

        In production this would read real sensor / environment data.
        """
        if not _TORCH_AVAILABLE or self._ego is None:
            return None

        # Combine ego state with a small tick-derived perturbation
        with torch.no_grad():
            noise = torch.randn(self._cfg.embed_dim, device=self._device) * 0.01
            obs = (self._ego + noise).unsqueeze(0)   # [1, embed_dim]

        self._log("runtime", "SENSE", {"tick": tick, "ego_norm": float(self._ego.norm().item())})
        return obs

    def _retrieve(self) -> list[Memory]:
        """Retrieve the most recent memories."""
        memories = self._memory.retrieve_recent(self._cfg.memory_retrieve_limit)
        self._log("memory", "RETRIEVE", {"count": len(memories)})
        return memories

    def _think(self, obs: Optional["torch.Tensor"]) -> ModelTelemetry:
        """Run the model forward pass and return telemetry."""
        if not _TORCH_AVAILABLE or obs is None:
            tel = ModelTelemetry()
            self._log("model", "THINK_SKIP", {"reason": "torch_unavailable"})
            return tel

        if hasattr(self._model, "eval"):
            self._model.eval()  # ensure inference mode

        with torch.no_grad():
            _ = self._model(obs)

        tel = self._model.telemetry
        self._log("model", "THINK", {
            "trigger_mean": round(tel.trigger_mean, 4),
            "trigger_max": round(tel.trigger_max, 4),
            "gate_fired": tel.gate_fired,
            "adaptive_depth_used": tel.adaptive_depth_used,
            "blocks_skipped": tel.blocks_skipped,
        })
        return tel

    def _decide(
        self,
        tick: int,
        telemetry: ModelTelemetry,
        memories: list[Memory],
    ) -> Action:
        """Run the planner to select the next action."""
        action = self._planner.plan(tick, telemetry, memories)
        self._log("policy", "DECIDE", {
            "action_kind": action.kind.value,
            "action_text": action.text[:100],
        })
        return action

    def _act(self, action: Action) -> None:
        """Execute the chosen action."""
        if action.kind is ActionKind.NOOP:
            return

        if action.kind is ActionKind.LOG_NOTE:
            self._log("policy", "ACTION_LOG_NOTE", {"text": action.text, **action.meta})

        elif action.kind is ActionKind.STORE_MEMORY:
            mem_id = self._memory.store_memory(
                kind="reflection",
                text=action.text,
                tags=action.tags,
                emotional_weight=action.emotional_weight,
                links=[],
            )
            self._log("policy", "ACTION_STORE_MEMORY", {"memory_id": mem_id, "text": action.text[:80]})

        elif action.kind is ActionKind.ASK_USER:
            print(action.text, flush=True)
            self._log("policy", "ACTION_ASK_USER", {"text": action.text})

    def _reflect(
        self,
        obs: Optional["torch.Tensor"],
        telemetry: ModelTelemetry,
    ) -> None:
        """Update the ego state vector using the model output."""
        if not _TORCH_AVAILABLE or obs is None:
            return

        import torch.nn.functional as _F  # noqa: PLC0415
        with torch.no_grad():
            out = self._model(obs)
            vec = out.squeeze(0)
            # EMA update toward latest output
            self._ego = self._ego * 0.9 + vec * 0.1
            self._ego = _F.normalize(self._ego, dim=0)

        self._log("runtime", "REFLECT", {"ego_norm": float(self._ego.norm().item())})

    def _store_tick(self, tick: int, telemetry: ModelTelemetry, action: Action) -> None:
        """Persist a lightweight tick summary to memory every 20 ticks."""
        if tick % 20 == 0:
            self._memory.store_memory(
                kind="tick_summary",
                text=(
                    f"Tick {tick}: action={action.kind.value}, "
                    f"gate={'fired' if telemetry.gate_fired else 'silent'}, "
                    f"trigger_mean={telemetry.trigger_mean:.3f}"
                ),
                tags=["tick_summary"],
                emotional_weight=0.3,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log(self, component: str, type_: str, payload: dict) -> None:
        self._logger.log(tick=self._tick, component=component, type=type_, payload=payload)

    def _handle_signal(self, signum: int, _frame: object) -> None:
        print(f"\n[Victor] Caught signal {signum}, shutting down…", flush=True)
        self._running = False

    def _shutdown(self) -> None:
        self._log("runtime", "SHUTDOWN", {"tick": self._tick})
        self._memory.close()
        self._logger.close()
        print(f"[Victor] Run {self._run_id} complete after {self._tick} ticks.", flush=True)
