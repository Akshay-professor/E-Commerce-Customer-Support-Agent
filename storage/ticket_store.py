"""
Support-ticket persistence (SQLite).

Previously tickets were appended to a flat `tickets.json`. They now live in the
shared `storage/app.db` so the admin dashboard can query open/escalated counts
and issue frequencies. The `create_ticket_document` shape is preserved so
callers in `tools.py` don't change beyond passing an optional user_id/category.
"""
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")

_initialized = False


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _ensure_table() -> None:
    global _initialized
    if _initialized:
        return
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id   TEXT PRIMARY KEY,
                user_id     TEXT,
                issue       TEXT,
                category    TEXT,
                status      TEXT,
                created_at  TEXT
            )
            """
        )
    _initialized = True


def create_ticket_document(issue: str, ticket_id: str,
                           user_id: str = "unknown",
                           category: str = "general") -> dict:
    return {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "issue": issue,
        "category": category,
        "status": "open",
        "created_at": datetime.utcnow().isoformat(),
    }


def save_ticket(ticket: dict) -> None:
    _ensure_table()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO tickets
                (ticket_id, user_id, issue, category, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ticket["ticket_id"],
                ticket.get("user_id", "unknown"),
                ticket.get("issue", ""),
                ticket.get("category", "general"),
                ticket.get("status", "open"),
                ticket.get("created_at", datetime.utcnow().isoformat()),
            ),
        )


def get_all_tickets() -> list[dict]:
    _ensure_table()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT ticket_id, user_id, issue, category, status, created_at "
            "FROM tickets ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_tickets_for_user(user_id: str) -> list[dict]:
    _ensure_table()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT ticket_id, user_id, issue, category, status, created_at "
            "FROM tickets WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]
