"""Production FractalLayerV2 neural model.

Key production fixes over the prototype:
- Activation checkpointing is **only** applied during training when gradients
  are enabled; inference is plain forward pass.
- Drop-path uses ``torch.rand`` tied to the global torch RNG for
  reproducibility; Python ``random`` is not used in the compute path.
- Dynamic quantization is CPU-only; a clear ``ValueError`` is raised if
  ``use_quantization=True`` with a CUDA device.
- ``last_gate_mask`` and trigger stats are exposed for telemetry.
- ``torch`` is an *optional* dependency.  If not installed, a lightweight
  ``NoopModel`` stub is returned from :func:`build_model`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Telemetry container
# ---------------------------------------------------------------------------

@dataclass
class ModelTelemetry:
    """Statistics from the most recent forward pass."""

    trigger_mean: float = 0.0
    trigger_max: float = 0.0
    gate_fired: bool = False          # True if gate_mask > 0.5
    adaptive_depth_used: int = 0
    blocks_skipped: int = 0
    last_gate_mask: Optional[Any] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Attempt PyTorch import
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint as cp
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Quantized linear
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class QuantizedFractalLinear(nn.Module):  # type: ignore[misc]
        """Linear layer with optional CPU dynamic quantization (int8).

        Parameters
        ----------
        input_dim, output_dim:
            Dimensions of the linear transformation.
        use_dynamic_quant:
            When ``True`` the layer is quantized to int8 after instantiation.
            *Only valid on CPU.*  A ``ValueError`` is raised at model build
            time if a CUDA device is requested alongside quantization.
        """

        def __init__(self, input_dim: int, output_dim: int, use_dynamic_quant: bool = False) -> None:
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self._quantized = False
            if use_dynamic_quant:
                self.linear = torch.quantization.quantize_dynamic(
                    self.linear, {nn.Linear}, dtype=torch.qint8
                )
                self._quantized = True

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.linear(x)

    # -----------------------------------------------------------------------
    # Trigger / gate submodules
    # -----------------------------------------------------------------------

    class MetaTrigger(nn.Module):  # type: ignore[misc]
        """Small MLP that outputs a scalar gate probability in [0, 1]."""

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class SparseRecursionGate(nn.Module):  # type: ignore[misc]
        """Hard threshold gate with Straight-Through Estimator for training."""

        def __init__(self, threshold: float = 0.5) -> None:
            super().__init__()
            self.threshold = threshold

        def forward(self, trigger: "torch.Tensor") -> "torch.Tensor":
            gate = (trigger > self.threshold).float()
            if self.training and torch.is_grad_enabled():
                # STE: allows gradient to flow through the hard threshold
                gate = gate + trigger - trigger.detach()
            return gate

    # -----------------------------------------------------------------------
    # FractalLayerV2
    # -----------------------------------------------------------------------

    class FractalLayerV2(nn.Module):  # type: ignore[misc]
        """Adaptive-depth fractal residual layer with telemetry hooks.

        Parameters
        ----------
        input_dim:
            Embedding dimensionality.
        depth:
            Maximum number of recursive blocks.
        drop_path_prob:
            Probability of *dropping* (skipping) a block during the forward
            pass.  Uses ``torch.rand`` for reproducibility with
            ``torch.manual_seed``.
        use_quantization:
            Apply CPU dynamic int8 quantization to all fractal blocks.
        """

        def __init__(
            self,
            input_dim: int,
            depth: int = 3,
            drop_path_prob: float = 0.1,
            use_quantization: bool = False,
        ) -> None:
            super().__init__()
            self.depth = depth
            self.input_dim = input_dim
            self.drop_path_prob = drop_path_prob
            self.trigger = MetaTrigger(input_dim)
            self.gate = SparseRecursionGate()
            self.fractal_blocks = nn.ModuleList([
                QuantizedFractalLinear(input_dim, input_dim, use_dynamic_quant=use_quantization)
                for _ in range(depth)
            ])
            self.norm = nn.LayerNorm(input_dim)

            # Telemetry — updated every forward call
            self.telemetry = ModelTelemetry()

        def _block_forward(self, idx: int, x: "torch.Tensor") -> "torch.Tensor":
            """Run fractal block *idx* on *x*; wrapped for checkpointing."""
            return self.fractal_blocks[idx](x)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: D102
            trigger_output = self.trigger(x)         # [batch, 1]
            gate_mask = self.gate(trigger_output)    # [batch, 1]

            # ---- telemetry snapshot (no grad) ----
            with torch.no_grad():
                t_mean = trigger_output.mean().item()
                t_max = trigger_output.max().item()
            self.telemetry = ModelTelemetry(
                trigger_mean=t_mean,
                trigger_max=t_max,
                gate_fired=bool(gate_mask.mean().item() > 0.5),
                last_gate_mask=gate_mask.detach().cpu(),
            )

            # Adaptive recursion depth: clamp to [1, depth]
            adaptive_depth = int(
                (trigger_output.mean() * self.depth).clamp(1, self.depth).item()
            )
            self.telemetry.adaptive_depth_used = adaptive_depth

            output = x
            skipped = 0
            for i in range(adaptive_depth):
                # Use torch RNG for drop-path reproducibility with manual_seed.
                # torch.rand(1) keeps the value on CPU since we extract it immediately.
                drop = torch.rand(1).item()
                if drop > self.drop_path_prob:
                    if self.training and torch.is_grad_enabled():
                        # Checkpointing only helps during backprop
                        residual = cp.checkpoint(
                            self._block_forward, i, output, use_reentrant=False
                        )
                    else:
                        residual = self._block_forward(i, output)
                    # gate_mask broadcasts: [batch,1] * [batch,dim]
                    output = residual * gate_mask + output
                else:
                    skipped += 1

            self.telemetry.blocks_skipped = skipped
            return self.norm(output)

    # -----------------------------------------------------------------------
    # Build helper
    # -----------------------------------------------------------------------

    def build_model(
        input_dim: int,
        depth: int = 3,
        drop_path_prob: float = 0.1,
        use_quantization: bool = False,
        device: str = "cpu",
    ) -> "FractalLayerV2":
        """Construct and return a :class:`FractalLayerV2`.

        Raises
        ------
        ValueError
            If ``use_quantization=True`` and ``device`` is not ``"cpu"``.
        """
        if use_quantization and device != "cpu":
            raise ValueError(
                "Dynamic quantization is only supported on CPU. "
                f"Got device={device!r}.  Either set device='cpu' or "
                "disable use_quantization."
            )
        model = FractalLayerV2(
            input_dim=input_dim,
            depth=depth,
            drop_path_prob=drop_path_prob,
            use_quantization=use_quantization,
        )
        model.to(torch.device(device))
        return model

else:  # torch not available
    class FractalLayerV2:  # type: ignore[no-redef]
        """Stub when torch is not installed."""

        telemetry = ModelTelemetry()

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def eval(self) -> "FractalLayerV2":
            return self

        def __call__(self, x: Any) -> Any:
            return x

    def build_model(*args: Any, **kwargs: Any) -> "FractalLayerV2":  # type: ignore[misc]
        """Return stub model when torch is unavailable."""
        return FractalLayerV2()
