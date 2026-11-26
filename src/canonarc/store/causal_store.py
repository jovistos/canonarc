from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Self

from .episodes import _EpisodeStoreMixin
from .graph import _GraphStoreMixin
from .schema import SCHEMA_SQL
from .timelines import _TimelineStoreMixin


class CausalStore(_TimelineStoreMixin, _GraphStoreMixin, _EpisodeStoreMixin):
    """Manages SQLite storage, recursive lineage resolution, and time-travel operations."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._tx_depth = 0
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Reentrant atomic SQLite transaction guarded by an RLock."""
        with self._lock:
            is_outermost = self._tx_depth == 0
            if is_outermost and not self.conn.in_transaction:
                self.conn.execute("BEGIN")
            self._tx_depth += 1
            try:
                yield
            except BaseException:
                self._tx_depth -= 1
                if is_outermost and self.conn.in_transaction:
                    self.conn.rollback()
                raise
            else:
                self._tx_depth -= 1
                if is_outermost and self.conn.in_transaction:
                    self.conn.commit()

    def _init_db(self) -> None:
        self.conn.execute("PRAGMA foreign_keys = ON;")
        if self.db_path != ":memory:" and not self.db_path.startswith("file::memory:"):
            self.conn.execute("PRAGMA journal_mode = WAL;")
            self.conn.execute("PRAGMA busy_timeout = 5000;")
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)
            # Safe migration for existing DBs lacking believed_summary column
            cols = {
                row["name"]
                for row in self.conn.execute("PRAGMA table_info(episode_epistemics)").fetchall()
            }
            if "believed_summary" not in cols:
                self.conn.execute("ALTER TABLE episode_epistemics ADD COLUMN believed_summary TEXT")
            # Backfill FTS5 index for legacy databases created before episodes_fts existed
            if not self.conn.execute("SELECT 1 FROM episodes_fts LIMIT 1").fetchone():
                self.conn.execute(
                    "INSERT INTO episodes_fts(rowid, timeline_id, episode_id, summary) "
                    "SELECT rowid, timeline_id, episode_id, summary FROM episodes"
                )
            # Ensure default 'main' timeline exists
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM timelines WHERE timeline_id = 'main'")
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO timelines (timeline_id, parent_id, fork_step, metadata) VALUES (?, NULL, NULL, ?)",
                    ("main", json.dumps({"description": "Primary timeline"})),
                )

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
