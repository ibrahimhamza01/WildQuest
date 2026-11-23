"""SQLite persistence for observations and the XP ledger (standard library only)."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from .models import (Observation, ObservationEvent, ObservationError, Status, XPOutcome,
                     from_utc_text, to_utc_text)

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE observations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id   TEXT    NOT NULL UNIQUE,
    user_id      TEXT    NOT NULL,
    class_id     INTEGER NOT NULL CHECK (class_id >= 0),
    confidence   REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    observed_at  TEXT    NOT NULL,
    latitude     REAL,
    longitude    REAL,
    status       TEXT    NOT NULL CHECK (status IN ('VERIFIED', 'NEEDS_REVIEW', 'REJECTED')),
    encounter_id INTEGER REFERENCES observations(id),
    image_ref    TEXT,
    created_at   TEXT    NOT NULL,
    reviewed_at  TEXT,
    CHECK ((latitude IS NULL) = (longitude IS NULL))
);
CREATE INDEX observations_user_species_time
    ON observations (user_id, class_id, observed_at);

-- Append-only ledger: one award per observation at most, one first discovery per species.
CREATE TABLE xp_awards (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER NOT NULL UNIQUE REFERENCES observations(id),
    user_id        TEXT    NOT NULL,
    class_id       INTEGER NOT NULL,
    reason         TEXT    NOT NULL CHECK (reason IN ('FIRST_DISCOVERY', 'NEW_ENCOUNTER')),
    points         INTEGER NOT NULL CHECK (points > 0),
    awarded_at     TEXT    NOT NULL
);
CREATE UNIQUE INDEX xp_awards_one_first_discovery
    ON xp_awards (user_id, class_id) WHERE reason = 'FIRST_DISCOVERY';
CREATE INDEX xp_awards_user_species_time ON xp_awards (user_id, class_id, awarded_at);
"""


class ObservationStore:
    """Thin data-access layer; XP rules live in ``wildquest.gamification``."""

    def __init__(self, db_path, class_names: Sequence[str]):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.class_names = tuple(class_names)
        # isolation_level=None: transactions are opened explicitly with BEGIN IMMEDIATE.
        self._db = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def _migrate(self) -> None:
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return
        if version > SCHEMA_VERSION:
            raise ObservationError(
                f"{self.db_path} uses schema v{version}; this code supports v{SCHEMA_VERSION}")
        with self.transaction():
            # Re-check inside the lock in case another process just created the schema.
            if self._db.execute("PRAGMA user_version").fetchone()[0] == 0:
                for statement in SCHEMA.split(";"):
                    if statement.strip():
                        self._db.execute(statement)
                self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Write lock for the whole read-decide-write sequence, so concurrent processes
        cannot both see "no award yet" and award twice."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    # ----- observations -------------------------------------------------------------

    def _row_to_observation(self, row: sqlite3.Row) -> Observation:
        return Observation(
            id=row["id"],
            request_id=row["request_id"],
            user_id=row["user_id"],
            class_id=row["class_id"],
            species=self.class_names[row["class_id"]],
            confidence=row["confidence"],
            observed_at=from_utc_text(row["observed_at"]),
            latitude=row["latitude"],
            longitude=row["longitude"],
            status=Status(row["status"]),
            encounter_id=row["encounter_id"],
            image_ref=row["image_ref"],
            created_at=from_utc_text(row["created_at"]),
            reviewed_at=from_utc_text(row["reviewed_at"]) if row["reviewed_at"] else None,
        )

    def insert_observation(self, event: ObservationEvent, status: Status,
                           created_at: datetime) -> Observation:
        cursor = self._db.execute(
            """INSERT INTO observations (request_id, user_id, class_id, confidence, observed_at,
                   latitude, longitude, status, image_ref, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event.request_id, event.user_id, event.class_id, event.confidence,
             to_utc_text(event.observed_at), event.latitude, event.longitude, status.value,
             event.image_ref, to_utc_text(created_at)))
        return self.get(cursor.lastrowid)

    def get(self, observation_id: int) -> Optional[Observation]:
        row = self._db.execute("SELECT * FROM observations WHERE id = ?",
                               (observation_id,)).fetchone()
        return self._row_to_observation(row) if row else None

    def get_by_request_id(self, request_id: str) -> Optional[Observation]:
        row = self._db.execute("SELECT * FROM observations WHERE request_id = ?",
                               (request_id,)).fetchone()
        return self._row_to_observation(row) if row else None

    def update_status(self, observation_id: int, status: Status, reviewed_at: datetime) -> None:
        self._db.execute("UPDATE observations SET status = ?, reviewed_at = ? WHERE id = ?",
                         (status.value, to_utc_text(reviewed_at), observation_id))

    def set_encounter(self, observation_id: int, encounter_id: int) -> None:
        self._db.execute("UPDATE observations SET encounter_id = ? WHERE id = ?",
                         (encounter_id, observation_id))

    def verified_between(self, user_id: str, class_id: int, start: datetime, end: datetime,
                         exclude_id: int) -> List[Observation]:
        rows = self._db.execute(
            """SELECT * FROM observations
               WHERE user_id = ? AND class_id = ? AND status = 'VERIFIED' AND id != ?
                 AND observed_at BETWEEN ? AND ?
               ORDER BY observed_at""",
            (user_id, class_id, exclude_id, to_utc_text(start), to_utc_text(end))).fetchall()
        return [self._row_to_observation(row) for row in rows]

    def history(self, user_id: str, limit: Optional[int] = None,
                status: Optional[Status] = None) -> List[Observation]:
        sql = "SELECT * FROM observations WHERE user_id = ?"
        params: list = [user_id]
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        sql += " ORDER BY observed_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._row_to_observation(row) for row in self._db.execute(sql, params)]

    def species_stats(self, user_id: str) -> List[sqlite3.Row]:
        """Per-species counts used to build the collection."""
        return self._db.execute(
            """SELECT class_id,
                      SUM(status = 'VERIFIED')                                AS verified,
                      SUM(status = 'NEEDS_REVIEW')                            AS pending,
                      MIN(CASE WHEN status = 'VERIFIED' THEN observed_at END) AS first_seen,
                      MAX(CASE WHEN status = 'VERIFIED' THEN observed_at END) AS last_seen,
                      MAX(CASE WHEN status = 'VERIFIED' THEN confidence END)  AS best_confidence,
                      COUNT(DISTINCT CASE WHEN status = 'VERIFIED' THEN encounter_id END)
                                                                              AS encounters
               FROM observations WHERE user_id = ? GROUP BY class_id""",
            (user_id,)).fetchall()

    def status_counts(self, user_id: str) -> dict:
        rows = self._db.execute(
            "SELECT status, COUNT(*) FROM observations WHERE user_id = ? GROUP BY status",
            (user_id,)).fetchall()
        counts = {status.value: 0 for status in Status}
        counts.update({row[0]: row[1] for row in rows})
        return counts

    # ----- XP ledger ----------------------------------------------------------------

    def insert_award(self, observation: Observation, reason: XPOutcome, points: int,
                     awarded_at: datetime) -> None:
        self._db.execute(
            """INSERT INTO xp_awards (observation_id, user_id, class_id, reason, points, awarded_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (observation.id, observation.user_id, observation.class_id, reason.value, points,
             to_utc_text(awarded_at)))

    def has_first_discovery(self, user_id: str, class_id: int) -> bool:
        return self._db.execute(
            """SELECT 1 FROM xp_awards
               WHERE user_id = ? AND class_id = ? AND reason = 'FIRST_DISCOVERY'""",
            (user_id, class_id)).fetchone() is not None

    def count_awards(self, user_id: str, class_id: int, reason: XPOutcome, start: datetime,
                     end: datetime) -> int:
        return self._db.execute(
            """SELECT COUNT(*) FROM xp_awards
               WHERE user_id = ? AND class_id = ? AND reason = ?
                 AND awarded_at >= ? AND awarded_at < ?""",
            (user_id, class_id, reason.value, to_utc_text(start), to_utc_text(end))).fetchone()[0]

    def total_xp(self, user_id: str) -> int:
        return self._db.execute("SELECT COALESCE(SUM(points), 0) FROM xp_awards WHERE user_id = ?",
                                (user_id,)).fetchone()[0]

    def awards(self, user_id: str) -> List[dict]:
        rows = self._db.execute(
            """SELECT observation_id, class_id, reason, points, awarded_at FROM xp_awards
               WHERE user_id = ? ORDER BY awarded_at, id""", (user_id,)).fetchall()
        return [{**dict(row), "species": self.class_names[row["class_id"]]} for row in rows]
