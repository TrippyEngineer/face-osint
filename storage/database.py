"""
storage/database.py
────────────────────
SQLite-based storage for all search metadata.

Why SQLite:
  • Zero install — built into Python
  • Zero RAM overhead — no separate process
  • Single file in data/ — trivial to backup, move, delete
  • Sufficient for thousands of searches

Schema:
  searches    — one row per search job
  matches     — one row per result from any scraper
  embeddings  — 512D vectors stored as BLOB for fast lookup
"""

import sqlite3
import json
import logging
import numpy as np
from datetime import datetime
from typing import Optional
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS searches (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    company         TEXT,
    location        TEXT,
    output_folder   TEXT,
    status          TEXT DEFAULT 'processing',
    verdict         TEXT,
    combined_score  REAL,
    created_at      TEXT NOT NULL,
    completed_at    TEXT,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id       TEXT NOT NULL REFERENCES searches(id),
    source          TEXT NOT NULL,
    name            TEXT,
    url             TEXT,
    username        TEXT,
    email           TEXT,
    company         TEXT,
    location        TEXT,
    face_score      REAL,
    name_score      REAL,
    combined_score  REAL,
    verdict         TEXT,
    raw_data        TEXT,   -- JSON blob
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_vectors (
    search_id   TEXT PRIMARY KEY REFERENCES searches(id),
    name        TEXT NOT NULL,
    vector_blob BLOB NOT NULL,   -- float32 numpy array, 512 floats = 2KB
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_searches_name    ON searches(name);
CREATE INDEX IF NOT EXISTS idx_matches_search   ON matches(search_id);
CREATE INDEX IF NOT EXISTS idx_matches_verdict  ON matches(verdict);

-- CIC crowd-sourced face captures (no FK to searches — independent pipeline)
CREATE TABLE IF NOT EXISTS cic_face_captures (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id    INTEGER NOT NULL,
    slot_id     INTEGER NOT NULL,
    zone_id     TEXT NOT NULL,
    zone_name   TEXT NOT NULL,
    vector_blob BLOB NOT NULL,
    captured_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cic_cap_track  ON cic_face_captures(track_id);
CREATE INDEX IF NOT EXISTS idx_cic_cap_slot   ON cic_face_captures(slot_id);
"""


class Database:
    """
    Thread-safe SQLite wrapper.
    Each method opens its own connection — safe for multi-threaded scraper use.
    """

    def __init__(self, path: Path = config.DB_PATH):
        self.path = path
        self._init_schema()
        logger.info(f"Database ready: {path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.debug("Database schema initialised")

    # ── Search CRUD ──────────────────────────────────────────────────────
    def create_search(
        self,
        search_id:     str,
        name:          str,
        company:       str = "",
        location:      str = "",
        output_folder: str = "",
    ):
        sql = """
            INSERT INTO searches (id, name, company, location, output_folder, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.execute(sql, (
                search_id, name, company, location,
                output_folder, _now(),
            ))
        logger.debug(f"Search created: {search_id} '{name}'")

    def complete_search(
        self,
        search_id:     str,
        verdict:       str,
        combined_score: float,
        error:         str = "",
    ):
        sql = """
            UPDATE searches
            SET status='done', verdict=?, combined_score=?,
                completed_at=?, error=?
            WHERE id=?
        """
        with self._connect() as conn:
            conn.execute(sql, (
                verdict, round(combined_score, 4),
                _now(), error, search_id,
            ))

    def get_search(self, search_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM searches WHERE id=?", (search_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_searches(self, limit: int = 50) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM searches ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Match CRUD ───────────────────────────────────────────────────────
    def insert_match(
        self,
        search_id:     str,
        source:        str,
        match_data:    dict,
    ):
        sql = """
            INSERT INTO matches
                (search_id, source, name, url, username, email, company,
                 location, face_score, name_score, combined_score, verdict,
                 raw_data, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with self._connect() as conn:
            conn.execute(sql, (
                search_id,
                source,
                match_data.get("name") or match_data.get("username"),
                match_data.get("url") or match_data.get("profile_url"),
                match_data.get("username"),
                match_data.get("email"),
                match_data.get("company") or match_data.get("affiliation"),
                match_data.get("location"),
                match_data.get("face_score"),
                match_data.get("name_score"),
                match_data.get("combined_score"),
                match_data.get("verdict"),
                json.dumps(match_data, default=str),
                _now(),
            ))

    def get_matches(self, search_id: str) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM matches WHERE search_id=? ORDER BY combined_score DESC",
                (search_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Face vector store ─────────────────────────────────────────────────
    def store_vector(self, search_id: str, name: str, vector: np.ndarray):
        blob = vector.astype(np.float32).tobytes()
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO face_vectors (search_id, name, vector_blob, created_at)
                   VALUES (?,?,?,?)""",
                (search_id, name, blob, _now()),
            )

    def get_all_vectors(self) -> list:
        """Return list of {search_id, name, vector} for similarity search."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT search_id, name, vector_blob FROM face_vectors"
            ).fetchall()
        result = []
        for row in rows:
            try:
                vec = np.frombuffer(row["vector_blob"], dtype=np.float32)
                result.append({
                    "search_id": row["search_id"],
                    "name":      row["name"],
                    "vector":    vec,
                })
            except Exception as e:
                logger.warning(f"Corrupt vector for {row['search_id']}: {e}")
        return result

    def find_similar_faces(
        self,
        query_vector: np.ndarray,
        top_k:        int = 5,
        threshold:    float = config.FACE_CONFIRMED,
    ) -> list:
        """
        Numpy cosine search across all stored face vectors.
        Fast enough for < 10,000 stored faces on limited hardware.
        Returns list of {search_id, name, score} sorted by score desc.
        """
        all_vecs = self.get_all_vectors()
        if not all_vecs:
            return []

        from embedding import cosine_similarity
        scored = []
        for entry in all_vecs:
            score = cosine_similarity(query_vector, entry["vector"])
            if score >= threshold:
                scored.append({
                    "search_id": entry["search_id"],
                    "name":      entry["name"],
                    "score":     round(score, 4),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ── CIC crowd face captures ───────────────────────────────────────────
    def store_cic_capture(
        self,
        track_id:  int,
        slot_id:   int,
        zone_id:   str,
        zone_name: str,
        vector:    "np.ndarray",
    ):
        blob = vector.astype(np.float32).tobytes()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO cic_face_captures
                   (track_id, slot_id, zone_id, zone_name, vector_blob, captured_at)
                   VALUES (?,?,?,?,?,?)""",
                (track_id, slot_id, zone_id, zone_name, blob, _now()),
            )

    def find_cic_captures(
        self,
        query_vector: "np.ndarray",
        top_k:        int = 5,
        threshold:    float = 0.45,
    ) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, track_id, slot_id, zone_id, zone_name,
                          vector_blob, captured_at
                   FROM cic_face_captures"""
            ).fetchall()
        if not rows:
            return []

        from embedding import cosine_similarity
        scored = []
        for row in rows:
            try:
                vec   = np.frombuffer(row["vector_blob"], dtype=np.float32)
                score = cosine_similarity(query_vector, vec)
                if score >= threshold:
                    scored.append({
                        "id":         row["id"],
                        "track_id":   row["track_id"],
                        "slot_id":    row["slot_id"],
                        "zone_id":    row["zone_id"],
                        "zone_name":  row["zone_name"],
                        "captured_at": row["captured_at"],
                        "score":      round(score, 4),
                    })
            except Exception:
                pass
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_cic_capture_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM cic_face_captures").fetchone()[0]


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
