"""Unit tests for victor.logger — structured JSONL event logger."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from victor.logger import EventLogger


class TestEventLoggerFormatting(unittest.TestCase):
    """Verify JSONL output structure and field presence."""

    def _make_logger(self, tmp: Path) -> EventLogger:
        return EventLogger(log_dir=tmp, run_id="run-abc123", max_bytes=1024 * 1024)

    def test_event_fields_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._make_logger(Path(tmp))
            logger.log(tick=1, component="test", type="PING", payload={"msg": "hello"})
            logger.close()

            path = logger.current_path()
            self.assertIsNotNone(path)
            assert path is not None
            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)

            event = json.loads(lines[0])
            self.assertIn("ts", event)
            self.assertIn("run_id", event)
            self.assertIn("tick", event)
            self.assertIn("component", event)
            self.assertIn("type", event)
            self.assertIn("payload", event)

    def test_run_id_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = EventLogger(log_dir=tmp, run_id="my-run-42", max_bytes=10 * 1024)
            logger.log(tick=0, component="c", type="T")
            logger.close()

            path = logger.current_path()
            assert path is not None
            event = json.loads(path.read_text().strip())
            self.assertEqual(event["run_id"], "my-run-42")

    def test_tick_and_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._make_logger(Path(tmp))
            logger.log(tick=42, component="memory", type="STORE", payload={"id": 7})
            logger.close()

            path = logger.current_path()
            assert path is not None
            event = json.loads(path.read_text().strip())
            self.assertEqual(event["tick"], 42)
            self.assertEqual(event["component"], "memory")
            self.assertEqual(event["type"], "STORE")
            self.assertEqual(event["payload"]["id"], 7)

    def test_empty_payload_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._make_logger(Path(tmp))
            logger.log(tick=1, component="c", type="T")
            logger.close()

            path = logger.current_path()
            assert path is not None
            event = json.loads(path.read_text().strip())
            self.assertIsInstance(event["payload"], dict)

    def test_multiple_events_multiple_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._make_logger(Path(tmp))
            for i in range(5):
                logger.log(tick=i, component="c", type="T", payload={"i": i})
            logger.close()

            path = logger.current_path()
            assert path is not None
            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 5)
            for i, line in enumerate(lines):
                event = json.loads(line)
                self.assertEqual(event["tick"], i)
                self.assertEqual(event["payload"]["i"], i)

    def test_ts_is_utc_iso(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = self._make_logger(Path(tmp))
            logger.log(tick=1, component="c", type="T")
            logger.close()

            path = logger.current_path()
            assert path is not None
            event = json.loads(path.read_text().strip())
            ts = event["ts"]
            # Must be parseable as ISO datetime with timezone
            from datetime import datetime  # noqa: PLC0415
            dt = datetime.fromisoformat(ts)
            self.assertIsNotNone(dt.tzinfo)


class TestEventLoggerRotation(unittest.TestCase):
    """Verify size-based rotation creates new files."""

    def test_size_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Very small max_bytes so rotation triggers quickly
            logger = EventLogger(log_dir=Path(tmp), run_id="r", max_bytes=200)
            # Write many events to exceed 200 bytes
            for i in range(30):
                logger.log(tick=i, component="c", type="T", payload={"data": "x" * 20})
            logger.close()

            jsonl_files = sorted(Path(tmp).glob("events-*.jsonl"))
            self.assertGreater(len(jsonl_files), 1, "Expected at least one rotation")

    def test_context_manager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with EventLogger(log_dir=tmp, run_id="ctx", max_bytes=1024) as logger:
                logger.log(tick=1, component="c", type="T")
            path = logger.current_path()
            assert path is not None
            self.assertTrue(path.exists())


class TestEventLoggerThreadSafety(unittest.TestCase):
    """Verify concurrent writes produce valid JSONL without interleaving."""

    def test_concurrent_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = EventLogger(log_dir=Path(tmp), run_id="thread-test")
            errors: list[Exception] = []

            def write_events(thread_id: int) -> None:
                try:
                    for i in range(20):
                        logger.log(
                            tick=i,
                            component=f"thread-{thread_id}",
                            type="CONCURRENT",
                            payload={"tid": thread_id, "i": i},
                        )
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=write_events, args=(t,)) for t in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            logger.close()
            self.assertEqual(errors, [], f"Thread errors: {errors}")

            # All lines must be valid JSON
            path = logger.current_path()
            assert path is not None
            for line in path.read_text().splitlines():
                if line.strip():
                    json.loads(line)  # must not raise


if __name__ == "__main__":
    unittest.main()
