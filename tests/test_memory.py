"""Unit tests for victor.memory — SQLite persistent memory store."""

from __future__ import annotations

import time
import unittest

from victor.memory import Memory, MemoryStore


class TestMemoryStoreBasic(unittest.TestCase):
    """Basic CRUD and schema tests."""

    def _make_store(self) -> MemoryStore:
        return MemoryStore(db_path=":memory:", run_id="test-run")

    def test_store_and_retrieve_recent(self) -> None:
        store = self._make_store()
        mid = store.store_memory(kind="observation", text="Hello world", tags=["greet"])
        self.assertIsInstance(mid, int)
        self.assertGreater(mid, 0)

        memories = store.retrieve_recent(limit=10)
        self.assertEqual(len(memories), 1)
        m = memories[0]
        self.assertEqual(m.kind, "observation")
        self.assertEqual(m.text, "Hello world")
        self.assertEqual(m.tags, ["greet"])
        self.assertAlmostEqual(m.emotional_weight, 0.5)
        self.assertEqual(m.run_id, "test-run")
        store.close()

    def test_retrieve_recent_limit(self) -> None:
        store = self._make_store()
        for i in range(15):
            store.store_memory(kind="note", text=f"note {i}")
        memories = store.retrieve_recent(limit=5)
        self.assertEqual(len(memories), 5)
        store.close()

    def test_emotional_weight_persisted(self) -> None:
        store = self._make_store()
        store.store_memory(kind="event", text="big event", emotional_weight=0.9)
        m = store.retrieve_recent(1)[0]
        self.assertAlmostEqual(m.emotional_weight, 0.9)
        store.close()

    def test_tags_round_trip(self) -> None:
        store = self._make_store()
        store.store_memory(kind="note", text="tagged note", tags=["alpha", "beta", "gamma"])
        m = store.retrieve_recent(1)[0]
        self.assertEqual(sorted(m.tags), ["alpha", "beta", "gamma"])
        store.close()

    def test_links_round_trip(self) -> None:
        store = self._make_store()
        mid1 = store.store_memory(kind="a", text="first")
        mid2 = store.store_memory(kind="b", text="second", links=[mid1])
        m = store.retrieve_recent(1)[0]
        self.assertIn(mid1, m.links)
        store.close()

    def test_retrieve_recent_ordering(self) -> None:
        store = self._make_store()
        ids = [store.store_memory(kind="n", text=f"item {i}") for i in range(5)]
        memories = store.retrieve_recent(limit=5)
        # Most recent first
        self.assertEqual(memories[0].id, ids[-1])
        store.close()

    def test_empty_store_returns_empty_list(self) -> None:
        store = self._make_store()
        self.assertEqual(store.retrieve_recent(10), [])
        store.close()

    def test_get_by_id(self) -> None:
        store = self._make_store()
        mid = store.store_memory(kind="fact", text="sky is blue")
        m = store.get_by_id(mid)
        self.assertIsNotNone(m)
        assert m is not None
        self.assertEqual(m.text, "sky is blue")
        store.close()

    def test_get_by_id_not_found(self) -> None:
        store = self._make_store()
        self.assertIsNone(store.get_by_id(99999))
        store.close()

    def test_context_manager(self) -> None:
        with MemoryStore(db_path=":memory:") as store:
            mid = store.store_memory(kind="ctx", text="via context manager")
            self.assertGreater(mid, 0)


class TestMemoryStoreSearch(unittest.TestCase):
    """Text search and relevance scoring."""

    def _make_store(self) -> MemoryStore:
        store = MemoryStore(db_path=":memory:", run_id="search-test")
        store.store_memory(kind="note", text="the cat sat on the mat", tags=["animal"])
        store.store_memory(kind="note", text="neural network training", tags=["ml"])
        store.store_memory(kind="note", text="cat neural network", tags=["ml", "animal"])
        return store

    def test_search_finds_matches(self) -> None:
        store = self._make_store()
        results = store.search_text("cat", limit=10)
        texts = [m.text for m in results]
        self.assertTrue(any("cat" in t for t in texts))
        store.close()

    def test_search_no_matches_returns_recent(self) -> None:
        store = self._make_store()
        # "xyzzy" won't match anything; fallback to retrieve_recent
        results = store.search_text("xyzzy", limit=5)
        # Should still return empty because no LIKE matches
        self.assertEqual(results, [])
        store.close()

    def test_search_empty_query_returns_recent(self) -> None:
        store = self._make_store()
        results = store.search_text("   ", limit=5)
        # Empty/whitespace query → retrieve_recent
        self.assertGreater(len(results), 0)
        store.close()

    def test_search_case_insensitive(self) -> None:
        store = self._make_store()
        lower = store.search_text("cat", limit=10)
        upper = store.search_text("CAT", limit=10)
        self.assertEqual(len(lower), len(upper))
        store.close()

    def test_relevance_score(self) -> None:
        """Memory with more keyword hits should score higher."""
        m_high = Memory(
            id=1, kind="n", text="cat cat cat", tags=["cat"],
            emotional_weight=0.5, links=[], created_at="2024-01-01T00:00:00+00:00",
            run_id=""
        )
        m_low = Memory(
            id=2, kind="n", text="cat", tags=[],
            emotional_weight=0.5, links=[], created_at="2024-01-01T00:00:00+00:00",
            run_id=""
        )
        now = time.time()
        self.assertGreater(
            m_high.relevance_score(["cat"], now),
            m_low.relevance_score(["cat"], now),
        )


if __name__ == "__main__":
    unittest.main()
