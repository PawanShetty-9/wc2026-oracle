"""
data/cache.py — SQLite Caching Layer
=====================================
Caches expensive API responses in a local SQLite database to:
  - Reduce API usage (The Odds API has a 500 req/month free limit)
  - Make the app load instantly on revisit (no re-fetching)
  - Decouple the Streamlit render cycle from network calls

Architecture:
  - One DB file: betting/data/raw/betting_cache.db (gitignored)
  - Thread-safe: uses threading.local() for SQLite connections
    (Streamlit runs in a multi-thread environment)
  - TTL-based invalidation: cache entries expire after a configurable duration
  - Schema auto-initialised on first import

CACHE TTLs:
  - Live odds:      30 minutes  (protect 500 req/month limit)
  - Match results:  30 minutes  (scores don't change once a match ends)
  - Team metadata:  24 hours    (FIFA rankings rarely change)

HOW TO DEBUG:
  - If the app shows stale data, call clear_cache() or delete the .db file
  - Check cache hit rates with: SELECT key, fetched_at FROM api_cache
  - The DB is at betting/data/raw/betting_cache.db — open with DB Browser for SQLite
  - If you see "OperationalError: database is locked", multiple processes are
    writing simultaneously — this is rare on Streamlit Cloud (single worker)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# DB location (gitignored — ephemeral on each deploy)
_DB_PATH = Path(__file__).parent / "raw" / "betting_cache.db"

# Thread-local storage: each Streamlit worker thread gets its own connection
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it if needed."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row  # dict-like row access
        _ensure_schema(conn)
        _local.conn = conn
    return _local.conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist yet (idempotent)."""
    conn.executescript("""
        -- General-purpose key-value cache for API responses
        CREATE TABLE IF NOT EXISTS api_cache (
            key          TEXT PRIMARY KEY,
            value        TEXT NOT NULL,    -- JSON blob
            fetched_at   REAL NOT NULL,    -- Unix timestamp
            ttl_seconds  INTEGER NOT NULL  -- how long the entry is valid
        );

        -- Bet tracking (persisted across sessions on same machine)
        CREATE TABLE IF NOT EXISTS bet_tracker (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id     TEXT NOT NULL,
            home_team    TEXT NOT NULL,
            away_team    TEXT NOT NULL,
            market       TEXT NOT NULL,
            outcome_label TEXT NOT NULL,
            decimal_odds REAL NOT NULL,
            stake        REAL NOT NULL,
            model_prob   REAL NOT NULL,
            ev           REAL NOT NULL,
            placed_at    TEXT NOT NULL,    -- ISO datetime
            result       TEXT,             -- NULL=open, "WIN", "LOSS", "VOID"
            pnl          REAL             -- profit/loss in currency units
        );
    """)
    conn.commit()


# ─── Cache read / write ──────────────────────────────────────────────────────

def get_cached(key: str) -> dict | list | None:
    """Retrieve a cached value if it exists and hasn't expired.

    Parameters
    ----------
    key : str — cache key (e.g. "wc_odds_2026-06-18")

    Returns
    -------
    Deserialized JSON object, or None if cache miss / expired.
    """
    conn = _get_conn()
    now = time.time()

    row = conn.execute(
        "SELECT value, fetched_at, ttl_seconds FROM api_cache WHERE key = ?",
        (key,),
    ).fetchone()

    if row is None:
        return None  # cache miss

    age = now - row["fetched_at"]
    if age > row["ttl_seconds"]:
        # Entry exists but is expired — delete it
        conn.execute("DELETE FROM api_cache WHERE key = ?", (key,))
        conn.commit()
        logger.debug("Cache expired for key=%s (age=%.0fs)", key, age)
        return None  # cache miss (expired)

    logger.debug("Cache hit for key=%s (age=%.0fs)", key, age)
    return json.loads(row["value"])


def set_cached(key: str, value: dict | list, ttl_seconds: int = 1800) -> None:
    """Store a value in the cache.

    Parameters
    ----------
    key         : str  — cache key
    value       : dict | list — JSON-serialisable value
    ttl_seconds : int  — time to live in seconds (default: 30 minutes)
    """
    conn = _get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO api_cache (key, value, fetched_at, ttl_seconds)
        VALUES (?, ?, ?, ?)
        """,
        (key, json.dumps(value), time.time(), ttl_seconds),
    )
    conn.commit()
    logger.debug("Cached key=%s (ttl=%ds)", key, ttl_seconds)


def clear_cache(key_prefix: str | None = None) -> int:
    """Delete cache entries.

    Parameters
    ----------
    key_prefix : str | None — if given, only delete keys starting with this prefix;
                              if None, delete ALL cache entries.

    Returns
    -------
    int — number of entries deleted.
    """
    conn = _get_conn()
    if key_prefix:
        cursor = conn.execute(
            "DELETE FROM api_cache WHERE key LIKE ?",
            (f"{key_prefix}%",),
        )
    else:
        cursor = conn.execute("DELETE FROM api_cache")
    conn.commit()
    return cursor.rowcount


# ─── Bet tracker CRUD ────────────────────────────────────────────────────────

def add_bet(
    match_id: str,
    home_team: str,
    away_team: str,
    market: str,
    outcome_label: str,
    decimal_odds: float,
    stake: float,
    model_prob: float,
    ev: float,
) -> int:
    """Insert a new bet into the tracker. Returns the new row ID."""
    from datetime import datetime
    conn = _get_conn()
    cursor = conn.execute(
        """
        INSERT INTO bet_tracker
          (match_id, home_team, away_team, market, outcome_label,
           decimal_odds, stake, model_prob, ev, placed_at, result, pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        (match_id, home_team, away_team, market, outcome_label,
         decimal_odds, stake, model_prob, ev,
         datetime.utcnow().isoformat()),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def settle_bet(bet_id: int, result: str) -> float:
    """Mark a bet as won/lost. Returns the P&L.

    Parameters
    ----------
    bet_id : int — row ID from add_bet()
    result : str — "WIN", "LOSS", or "VOID"

    Returns P&L in currency units (positive = profit).
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT stake, decimal_odds FROM bet_tracker WHERE id = ?", (bet_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"Bet ID {bet_id} not found in tracker")

    pnl: float
    if result == "WIN":
        pnl = round(row["stake"] * (row["decimal_odds"] - 1.0), 2)
    elif result == "LOSS":
        pnl = -row["stake"]
    else:  # VOID — stake returned, no profit
        pnl = 0.0

    conn.execute(
        "UPDATE bet_tracker SET result = ?, pnl = ? WHERE id = ?",
        (result, pnl, bet_id),
    )
    conn.commit()
    return pnl


def get_all_bets() -> list[dict]:
    """Return all bets as a list of dicts (most recent first)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM bet_tracker ORDER BY placed_at DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_pnl_summary() -> dict:
    """Return aggregate P&L statistics.

    Returns
    -------
    dict with keys: total_bets, open_bets, settled_bets,
                    total_staked, total_pnl, roi_pct
    """
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM bet_tracker").fetchall()
    bets = [dict(r) for r in rows]

    settled = [b for b in bets if b["result"] is not None]
    open_bets = [b for b in bets if b["result"] is None]

    total_staked = sum(b["stake"] for b in settled)
    total_pnl = sum(b["pnl"] for b in settled if b["pnl"] is not None)
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0

    return {
        "total_bets":    len(bets),
        "open_bets":     len(open_bets),
        "settled_bets":  len(settled),
        "total_staked":  round(total_staked, 2),
        "total_pnl":     round(total_pnl, 2),
        "roi_pct":       round(roi, 1),
    }
