"""SQLite-backed audit trail. Every arbitration run is persisted in full (as JSON)
so past verdicts can be retrieved, browsed, and mined for analytics."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from arbitration.models import ArbitrationRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS arbitrations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    original_prompt TEXT,
    original_output TEXT NOT NULL,
    overall_score INTEGER NOT NULL,
    confidence REAL NOT NULL,
    confirmed_issue_count INTEGER NOT NULL,
    dismissed_flag_count INTEGER NOT NULL,
    disagreement_count INTEGER NOT NULL,
    short_circuited INTEGER NOT NULL,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_arbitrations_created_at ON arbitrations(created_at);
"""


@contextmanager
def _connect(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_arbitration(db_path: str, record: ArbitrationRecord) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO arbitrations (
                id, created_at, original_prompt, original_output, overall_score, confidence,
                confirmed_issue_count, dismissed_flag_count, disagreement_count, short_circuited, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.created_at.isoformat(),
                record.original_prompt,
                record.original_output,
                record.verdict.overall_score,
                record.verdict.confidence,
                len(record.verdict.confirmed_issues),
                len(record.verdict.dismissed_flags),
                len(record.disagreements),
                int(record.verdict.short_circuited),
                record.model_dump_json(),
            ),
        )


def get_arbitration(db_path: str, arbitration_id: str) -> ArbitrationRecord | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT record_json FROM arbitrations WHERE id = ?", (arbitration_id,)).fetchone()
        if row is None:
            return None
        return ArbitrationRecord.model_validate(json.loads(row["record_json"]))


def list_arbitrations(db_path: str, limit: int = 50, offset: int = 0) -> list[ArbitrationRecord]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT record_json FROM arbitrations ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [ArbitrationRecord.model_validate(json.loads(row["record_json"])) for row in rows]


def count_arbitrations(db_path: str) -> int:
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM arbitrations").fetchone()[0]
