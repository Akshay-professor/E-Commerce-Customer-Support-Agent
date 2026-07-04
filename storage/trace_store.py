"""
Conversation tracing (SQLite).

One row per user turn is written by the `validate` node once the turn is
complete and every signal is known. Powers the trace-inspection page and the
admin dashboard.
"""
import json
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
            CREATE TABLE IF NOT EXISTS conversation_traces (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                TEXT,
                thread_id         TEXT,
                user_id           TEXT,
                scenario          TEXT,
                tools_used        TEXT,
                retrieved_docs    TEXT,
                llm_provider      TEXT,
                latency_ms        INTEGER,
                validation_result TEXT,
                confidence        REAL,
                final_answer      TEXT
            )
            """
        )
    _initialized = True


def log_trace(*, thread_id: str, user_id: str, scenario: str,
              tools_used: list, retrieved_docs: list, llm_provider: str,
              latency_ms: int, validation_result: str, confidence: float,
              final_answer: str) -> None:
    _ensure_table()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO conversation_traces
                (ts, thread_id, user_id, scenario, tools_used, retrieved_docs,
                 llm_provider, latency_ms, validation_result, confidence, final_answer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                thread_id,
                user_id,
                scenario,
                json.dumps(tools_used or []),
                json.dumps(retrieved_docs or []),
                llm_provider or "",
                int(latency_ms or 0),
                validation_result or "",
                float(confidence or 0.0),
                final_answer or "",
            ),
        )


def get_recent_traces(limit: int = 100) -> list[dict]:
    _ensure_table()
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM conversation_traces ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    traces = []
    for r in rows:
        d = dict(r)
        d["tools_used"] = json.loads(d["tools_used"] or "[]")
        d["retrieved_docs"] = json.loads(d["retrieved_docs"] or "[]")
        traces.append(d)
    return traces


def get_dashboard_metrics() -> dict:
    """Aggregate trace-derived metrics for the admin dashboard."""
    _ensure_table()
    from collections import Counter

    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT thread_id, scenario, tools_used, latency_ms FROM conversation_traces"
        ).fetchall()

    threads = set()
    scenario_counts: Counter = Counter()
    tool_counts: Counter = Counter()
    latencies: list[int] = []

    for r in rows:
        threads.add(r["thread_id"])
        if r["scenario"]:
            scenario_counts[r["scenario"]] += 1
        for t in json.loads(r["tools_used"] or "[]"):
            tool_counts[t] += 1
        if r["latency_ms"]:
            latencies.append(r["latency_ms"])

    return {
        "total_conversations": len(threads),
        "total_turns": len(rows),
        "scenario_counts": dict(scenario_counts),
        "tool_counts": dict(tool_counts),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
    }
