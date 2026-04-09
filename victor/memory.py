"""Persistent memory store backed by SQLite.

Schema
------
memories
  id               INTEGER PRIMARY KEY AUTOINCREMENT
  kind             TEXT    NOT NULL          -- e.g. 'observation', 'reflection'
  text             TEXT    NOT NULL
  tags             TEXT    NOT NULL DEFAULT ''   -- comma-separated
  emotional_weight REAL    NOT NULL DEFAULT 0.5
  links            TEXT    NOT NULL DEFAULT ''   -- comma-separated memory IDs
  created_at       TEXT    NOT NULL          -- UTC ISO-8601
  run_id           TEXT    NOT NULL DEFAULT ''

schema_version
  version          INTEGER PRIMARY KEY
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    """A single memory record."""

    id: int
    kind: str
    text: str
    tags: list[str]
    emotional_weight: float
    links: list[int]
    created_at: str
    run_id: str

    def relevance_score(self, query_terms: list[str], now_ts: float) -> float:
        """Compute a simple keyword + recency relevance score.

        Parameters
        ----------
        query_terms:
            Lower-cased tokens from the search query.
        now_ts:
            Current UNIX timestamp for recency decay calculation.

        Returns
        -------
        float
            Higher is more relevant.
        """
        text_lower = self.text.lower()
        tag_str = " ".join(self.tags).lower()

        # Keyword hit count (text + tags)
        keyword_score = sum(
            text_lower.count(t) + tag_str.count(t) for t in query_terms
        )

        # Recency decay: half-life of 1 hour
        try:
            created_ts = datetime.fromisoformat(self.created_at).timestamp()
        except ValueError:
            created_ts = now_ts
        age_hours = (now_ts - created_ts) / 3600.0
        recency = 1.0 / (1.0 + age_hours)

        return keyword_score + recency + self.emotional_weight * 0.5


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1

_DDL_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    kind             TEXT    NOT NULL,
    text             TEXT    NOT NULL,
    tags             TEXT    NOT NULL DEFAULT '',
    emotional_weight REAL    NOT NULL DEFAULT 0.5,
    links            TEXT    NOT NULL DEFAULT '',
    created_at       TEXT    NOT NULL,
    run_id           TEXT    NOT NULL DEFAULT ''
);
"""

_DDL_SCHEMA_VERSION = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


class MemoryStore:
    """SQLite-backed persistent memory store.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file.  Use ``":memory:"`` for
        ephemeral in-process storage (tests / inspection).
    run_id:
        Current run identifier stamped on every new memory record.
    """

    def __init__(self, db_path: str | Path = "victor_memory.db", run_id: str = "") -> None:
        self._db_path = str(db_path)
        self._run_id = run_id
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    # ------------------------------------------------------------------
    # Schema / migrations
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        """Create tables and apply any pending schema migrations."""
        cur = self._conn.cursor()
        cur.executescript(_DDL_SCHEMA_VERSION + _DDL_MEMORIES)
        # Persist current version
        cur.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
            (_SCHEMA_VERSION,),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_memory(
        self,
        *,
        kind: str,
        text: str,
        tags: Optional[list[str]] = None,
        emotional_weight: float = 0.5,
        links: Optional[list[int]] = None,
    ) -> int:
        """Insert a new memory record and return its row ID.

        Parameters
        ----------
        kind:
            Semantic category, e.g. ``'observation'``, ``'reflection'``,
            ``'directive'``.
        text:
            Free-form text content of the memory.
        tags:
            Optional list of tag strings for indexing / filtering.
        emotional_weight:
            Scalar in [0, 1] representing emotional salience.
        links:
            Optional list of related memory IDs (foreign-key convention).

        Returns
        -------
        int
            Row ID of the newly inserted memory.
        """
        tags_str = ",".join(tags or [])
        links_str = ",".join(str(i) for i in (links or []))
        created_at = datetime.now(timezone.utc).isoformat()

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO memories (kind, text, tags, emotional_weight, links, created_at, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (kind, text, tags_str, emotional_weight, links_str, created_at, self._run_id),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve_recent(self, limit: int = 10) -> list[Memory]:
        """Return the *limit* most recently created memories."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM memories ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_memory(r) for r in cur.fetchall()]

    def search_text(self, query: str, limit: int = 10) -> list[Memory]:
        """Full-text keyword search with relevance + recency scoring.

        Performs a case-insensitive LIKE scan across *text* and *tags*, then
        re-ranks results by the :meth:`Memory.relevance_score` heuristic.

        Parameters
        ----------
        query:
            Space-separated search terms.
        limit:
            Maximum number of results to return.

        Returns
        -------
        list[Memory]
            Memories ordered from most to least relevant.
        """
        terms = [t.strip().lower() for t in query.split() if t.strip()]
        if not terms:
            return self.retrieve_recent(limit)

        # Build a LIKE clause for each term (OR across terms for recall)
        like_clauses = " OR ".join(
            ["LOWER(text) LIKE ? OR LOWER(tags) LIKE ?"] * len(terms)
        )
        params: list[str] = []
        for t in terms:
            params.extend([f"%{t}%", f"%{t}%"])

        cur = self._conn.cursor()
        cur.execute(f"SELECT * FROM memories WHERE {like_clauses}", params)
        rows = cur.fetchall()

        memories = [self._row_to_memory(r) for r in rows]
        now_ts = time.time()
        memories.sort(key=lambda m: m.relevance_score(terms, now_ts), reverse=True)
        return memories[:limit]

    def get_by_id(self, memory_id: int) -> Optional[Memory]:
        """Retrieve a single memory by primary key."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cur.fetchone()
        return self._row_to_memory(row) if row else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        tags = [t for t in row["tags"].split(",") if t]
        links_raw = [t for t in row["links"].split(",") if t]
        links = [int(x) for x in links_raw if x.isdigit()]
        return Memory(
            id=row["id"],
            kind=row["kind"],
            text=row["text"],
            tags=tags,
            emotional_weight=row["emotional_weight"],
            links=links,
            created_at=row["created_at"],
            run_id=row["run_id"],
        )

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
