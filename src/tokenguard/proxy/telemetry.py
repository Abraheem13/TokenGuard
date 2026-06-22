"""Day 6 — request telemetry persisted to SQLite.

Every routed request is logged with the chosen model, the cost charged, the
end-to-end latency, and (when a reward signal is available) the observed reward.
The same table doubles as a durable replay buffer: the online levels can later
read recent rows to continue adapting across proxy restarts.

The store is deliberately dependency-free (standard-library ``sqlite3`` only) so
it runs anywhere the proxy runs, including a single-file deployment.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


_SCHEMA = """
CREATE TABLE IF NOT EXISTS telemetry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    request_id    TEXT    NOT NULL,
    query_chars   INTEGER NOT NULL,
    chosen_model  TEXT    NOT NULL,
    predicted_q   REAL,
    cost_usd      REAL    NOT NULL,
    latency_ms    REAL    NOT NULL,
    reward        REAL,
    routed_by     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry(ts);
"""


@dataclass
class Record:
    """A single routed-request telemetry row."""
    request_id: str
    query_chars: int
    chosen_model: str
    cost_usd: float
    latency_ms: float
    predicted_q: float | None = None
    reward: float | None = None
    routed_by: str = "tokenguard"
    ts: float = 0.0


class TelemetryStore:
    """Thin SQLite wrapper for append + recent-window reads."""

    def __init__(self, db_path: str | Path = "experiments/telemetry.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    def log(self, rec: Record) -> int:
        """Append one record; returns its row id."""
        if rec.ts == 0.0:
            rec.ts = time.time()
        cur = self._conn.execute(
            """INSERT INTO telemetry
               (ts, request_id, query_chars, chosen_model, predicted_q,
                cost_usd, latency_ms, reward, routed_by)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (rec.ts, rec.request_id, rec.query_chars, rec.chosen_model,
             rec.predicted_q, rec.cost_usd, rec.latency_ms, rec.reward,
             rec.routed_by),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def set_reward(self, request_id: str, reward: float) -> None:
        """Attach a reward to a previously logged request (late feedback)."""
        self._conn.execute(
            "UPDATE telemetry SET reward=? WHERE request_id=?",
            (reward, request_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    def recent(self, limit: int = 1000) -> list[dict]:
        """Most-recent rows first (replay-buffer read)."""
        cur = self._conn.execute(
            "SELECT * FROM telemetry ORDER BY id DESC LIMIT ?", (limit,)
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def summary(self) -> dict:
        """Aggregate stats for the dashboard / gate report."""
        cur = self._conn.execute(
            """SELECT COUNT(*)          AS n,
                      AVG(cost_usd)      AS avg_cost,
                      SUM(cost_usd)      AS total_cost,
                      AVG(latency_ms)    AS avg_latency,
                      AVG(reward)        AS avg_reward
               FROM telemetry"""
        )
        row = cur.fetchone()
        cols = [c[0] for c in cur.description]
        out = dict(zip(cols, row))
        # per-model routing share
        cur = self._conn.execute(
            "SELECT chosen_model, COUNT(*) FROM telemetry GROUP BY chosen_model"
        )
        out["by_model"] = {m: c for m, c in cur.fetchall()}
        return out

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0])

    def close(self) -> None:
        self._conn.close()