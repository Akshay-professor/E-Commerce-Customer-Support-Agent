"""
Enhanced memory (SQLite), keyed by user_id.

Goes beyond LangGraph's per-thread chat checkpoint by persisting cross-thread
knowledge about a customer:
  - a profile (derived from their real orders + demographics),
  - rolling conversation summaries,
  - lightweight preferences.

The agent node reads `build_memory_context(user_id)` and prepends it to the
system prompt so it serves the customer with continuity across conversations.
Previous support tickets are pulled live from `ticket_store`.
"""
import os
import sqlite3
from datetime import datetime

# NOTE: sibling modules are imported lazily inside the functions that use them.
# Importing them at module load caused an import-order failure on Streamlit
# Cloud (memory_store is imported while the storage package is still loading).

DB_PATH = os.path.join(os.path.dirname(__file__), "app.db")

_initialized = False


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _ensure_tables() -> None:
    global _initialized
    if _initialized:
        return
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS customer_profiles (
                user_id     TEXT PRIMARY KEY,
                name        TEXT,
                tier        TEXT,
                order_count INTEGER,
                return_count INTEGER,
                updated_at  TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                user_id    TEXT,
                thread_id  TEXT,
                summary    TEXT,
                updated_at TEXT,
                PRIMARY KEY (user_id, thread_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id    TEXT,
                key        TEXT,
                value      TEXT,
                updated_at TEXT,
                PRIMARY KEY (user_id, key)
            )
            """
        )
    _initialized = True


# --- Profile ---------------------------------------------------------------

def upsert_profile(user_id: str, name: str = "", tier: str = "") -> dict:
    """Refresh the cached profile from the customer's real order data."""
    from storage import orders_store
    _ensure_tables()
    stats = orders_store.get_user_stats(user_id)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO customer_profiles
                (user_id, name, tier, order_count, return_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                name = excluded.name,
                tier = excluded.tier,
                order_count = excluded.order_count,
                return_count = excluded.return_count,
                updated_at = excluded.updated_at
            """,
            (
                user_id, name, tier,
                stats["order_count"], stats["return_count"],
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )
    return {"user_id": user_id, "name": name, "tier": tier, **stats}


def get_profile(user_id: str) -> dict | None:
    _ensure_tables()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM customer_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


# --- Summaries -------------------------------------------------------------

def save_summary(user_id: str, thread_id: str, summary: str) -> None:
    _ensure_tables()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_summaries (user_id, thread_id, summary, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, thread_id) DO UPDATE SET
                summary = excluded.summary,
                updated_at = excluded.updated_at
            """,
            (user_id, thread_id, summary,
             datetime.utcnow().isoformat(timespec="seconds")),
        )


def get_recent_summaries(user_id: str, limit: int = 3) -> list[str]:
    _ensure_tables()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT summary FROM conversation_summaries "
            "WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [r[0] for r in rows if r[0]]


# --- Preferences -----------------------------------------------------------

def set_preference(user_id: str, key: str, value: str) -> None:
    _ensure_tables()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_preferences (user_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (user_id, key, value,
             datetime.utcnow().isoformat(timespec="seconds")),
        )


def get_preferences(user_id: str) -> dict:
    _ensure_tables()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT key, value FROM user_preferences WHERE user_id = ?", (user_id,)
        ).fetchall()
    return {k: v for k, v in rows}


# --- Context assembly ------------------------------------------------------

def build_memory_context(user_id: str) -> str:
    """
    Compact "what we know about this customer" block, injected into the system
    prompt. Returns "" if we know nothing yet, so the prompt stays clean.
    """
    try:
        _ensure_tables()
        profile = get_profile(user_id)
        prefs = get_preferences(user_id)
        summaries = get_recent_summaries(user_id, limit=2)
        try:
            from storage import ticket_store
            tickets = ticket_store.get_tickets_for_user(user_id)
        except Exception:
            tickets = []

        lines: list[str] = []
        if profile:
            desc = f"- Customer {user_id}"
            if profile.get("name"):
                desc += f" ({profile['name']})"
            if profile.get("tier"):
                desc += f", {profile['tier']} tier"
            desc += (f": {profile.get('order_count', 0)} orders, "
                     f"{profile.get('return_count', 0)} returns on record.")
            lines.append(desc)
        if tickets:
            open_t = [t for t in tickets if t.get("status") == "open"]
            if open_t:
                lines.append(f"- Has {len(open_t)} open support ticket(s); "
                             f"most recent issue: \"{open_t[0]['issue'][:120]}\".")
        if prefs:
            pref_str = ", ".join(f"{k}={v}" for k, v in prefs.items())
            lines.append(f"- Known preferences: {pref_str}.")
        if summaries:
            lines.append(f"- Recent history: {summaries[0][:240]}")

        if not lines:
            return ""

        return (
            "### WHAT WE KNOW ABOUT THIS CUSTOMER\n"
            "Use this context naturally; do not read it back verbatim.\n"
            + "\n".join(lines)
            + "\n"
        )
    except Exception:
        # Memory is an enhancement, never a hard dependency for a reply.
        return ""
