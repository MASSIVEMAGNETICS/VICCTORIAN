"""Unit tests for victor.model — FractalLayerV2 and telemetry.

These tests are skipped automatically when PyTorch is not installed.
"""

from __future__ import annotations

import unittest

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_SKIP = unittest.skipUnless(_TORCH_AVAILABLE, "torch not installed")


@_SKIP
class TestFractalLayerForwardShape(unittest.TestCase):
    """Output tensor shape and dtype tests."""

    def _make_layer(self, dim: int = 32, depth: int = 2) -> "object":
        from victor.model import FractalLayerV2  # noqa: PLC0415
        return FractalLayerV2(input_dim=dim, depth=depth, drop_path_prob=0.0)

    def test_output_shape_matches_input(self) -> None:
        layer = self._make_layer(dim=64)
        layer.eval()  # type: ignore[union-attr]
        x = torch.randn(4, 64)
        with torch.no_grad():
            out = layer(x)  # type: ignore[operator]
        self.assertEqual(out.shape, x.shape)

    def test_batch_size_one(self) -> None:
        layer = self._make_layer(dim=32)
        layer.eval()  # type: ignore[union-attr]
        x = torch.randn(1, 32)
        with torch.no_grad():
            out = layer(x)  # type: ignore[operator]
        self.assertEqual(out.shape, (1, 32))

    def test_output_dtype_float32(self) -> None:
        layer = self._make_layer(dim=16)
        layer.eval()  # type: ignore[union-attr]
        x = torch.randn(2, 16)
        with torch.no_grad():
            out = layer(x)  # type: ignore[operator]
        self.assertEqual(out.dtype, torch.float32)

    def test_no_nan_in_output(self) -> None:
        layer = self._make_layer(dim=32)
        layer.eval()  # type: ignore[union-attr]
        x = torch.randn(8, 32)
        with torch.no_grad():
            out = layer(x)  # type: ignore[operator]
        self.assertFalse(torch.isnan(out).any().item())


@_SKIP
class TestFractalLayerTelemetry(unittest.TestCase):
    """Telemetry fields are populated after forward pass."""

    def _run_forward(self, dim: int = 32) -> object:
        from victor.model import FractalLayerV2  # noqa: PLC0415
        layer = FractalLayerV2(input_dim=dim, depth=3, drop_path_prob=0.0)
        layer.eval()
        x = torch.randn(2, dim)
        with torch.no_grad():
            layer(x)
        return layer.telemetry

    def test_telemetry_trigger_mean_in_range(self) -> None:
        tel = self._run_forward()
        self.assertGreaterEqual(tel.trigger_mean, 0.0)  # type: ignore[union-attr]
        self.assertLessEqual(tel.trigger_mean, 1.0)  # type: ignore[union-attr]

    def test_telemetry_trigger_max_ge_mean(self) -> None:
        tel = self._run_forward()
        self.assertGreaterEqual(tel.trigger_max, tel.trigger_mean)  # type: ignore[union-attr]

    def test_telemetry_gate_fired_is_bool(self) -> None:
        tel = self._run_forward()
        self.assertIsInstance(tel.gate_fired, bool)  # type: ignore[union-attr]

    def test_telemetry_adaptive_depth_in_range(self) -> None:
        depth = 3
        from victor.model import FractalLayerV2  # noqa: PLC0415
        layer = FractalLayerV2(input_dim=32, depth=depth, drop_path_prob=0.0)
        layer.eval()
        x = torch.randn(1, 32)
        with torch.no_grad():
            layer(x)
        tel = layer.telemetry
        self.assertGreaterEqual(tel.adaptive_depth_used, 1)
        self.assertLessEqual(tel.adaptive_depth_used, depth)

    def test_last_gate_mask_shape(self) -> None:
        from victor.model import FractalLayerV2  # noqa: PLC0415
        layer = FractalLayerV2(input_dim=32, depth=2, drop_path_prob=0.0)
        layer.eval()
        x = torch.randn(4, 32)
        with torch.no_grad():
            layer(x)
        mask = layer.telemetry.last_gate_mask
        self.assertIsNotNone(mask)
        # gate_mask is [batch, 1]
        self.assertEqual(mask.shape, (4, 1))


@_SKIP
class TestDeterministicDropPath(unittest.TestCase):
    """Drop-path is reproducible when the torch RNG seed is fixed."""

    def _run_seeded(self, seed: int, drop_path_prob: float = 0.5) -> list:
        from victor.model import FractalLayerV2  # noqa: PLC0415
        torch.manual_seed(seed)
        layer = FractalLayerV2(input_dim=32, depth=4, drop_path_prob=drop_path_prob)
        layer.eval()
        torch.manual_seed(seed)  # re-seed before forward so rand() is deterministic
        x = torch.randn(1, 32)
        with torch.no_grad():
            out = layer(x)
        return out.tolist()

    def test_same_seed_same_output(self) -> None:
        out1 = self._run_seeded(seed=42)
        out2 = self._run_seeded(seed=42)
        self.assertEqual(out1, out2)

    def test_different_seed_different_output(self) -> None:
        out1 = self._run_seeded(seed=1)
        out2 = self._run_seeded(seed=999)
        # With drop-path prob 0.5 and depth 4 there is a very high chance
        # different seeds produce different outputs.
        self.assertNotEqual(out1, out2)


@_SKIP
class TestQuantizationGuard(unittest.TestCase):
    """CUDA + quantization should raise a clear error at build time."""

    @unittest.skipUnless(
        _TORCH_AVAILABLE and not __import__("torch").cuda.is_available(),
        "Only runs on CPU-only machines with torch installed",
    )
    def test_no_cuda_available_cpu_quantization_ok(self) -> None:
        from victor.model import build_model  # noqa: PLC0415
        # Should not raise on CPU
        model = build_model(input_dim=16, depth=1, use_quantization=True, device="cpu")
        self.assertIsNotNone(model)

    def test_cuda_quantization_raises(self) -> None:
        from victor.model import build_model  # noqa: PLC0415
        with self.assertRaises(ValueError):
            build_model(input_dim=16, depth=1, use_quantization=True, device="cuda")


@_SKIP
class TestNoCheckpointInInference(unittest.TestCase):
    """Verify inference does not call activation checkpointing."""

    def test_inference_runs_without_grad(self) -> None:
        """Should complete without raising even though no gradients exist."""
        from victor.model import FractalLayerV2  # noqa: PLC0415
        layer = FractalLayerV2(input_dim=32, depth=3, drop_path_prob=0.0)
        layer.eval()
        x = torch.randn(1, 32)
        # This must not raise even though no grad context is active
        with torch.no_grad():
            out = layer(x)
        self.assertEqual(out.shape, (1, 32))


if __name__ == "__main__":
    unittest.main()
