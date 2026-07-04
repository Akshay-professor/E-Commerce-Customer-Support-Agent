"""
Per-turn observability side-channel.

The LangGraph loop and the tools produce signals we want for tracing and the
confidence score (which LLM provider answered, how long it took, which policy
docs were retrieved and how well they matched). Threading all of that through
`ChatState` reducers is noisy, so instead each user turn gets a small scratch
dict that the agent node and the RAG tools write to, and the `validate` node
reads, persists, and clears.

Keyed by `thread_id` in a plain module-level dict: LangGraph may execute nodes
in worker threads, and a process-global dict is shared across them (unlike a
ContextVar, which does not propagate across threads). Each caller passes the
thread_id it reads from the graph config.
"""
from threading import Lock
from typing import Any

_store: dict[str, dict] = {}
_lock = Lock()


def _fresh() -> dict:
    return {
        "provider": None,        # which LLM provider answered ("groq"/"gemini")
        "latency_ms": 0,         # cumulative agent-node LLM latency for the turn
        "retrieval_scores": [],  # best FAISS similarity score per RAG call
        "retrieved_docs": [],    # short snippets of retrieved policy text
    }


def _key(thread_id: str | None) -> str:
    return thread_id or "default"


def _ctx(thread_id: str | None) -> dict:
    k = _key(thread_id)
    with _lock:
        if k not in _store:
            _store[k] = _fresh()
        return _store[k]


def reset(thread_id: str | None) -> None:
    """Start a fresh scratch dict for a new user turn."""
    with _lock:
        _store[_key(thread_id)] = _fresh()


def set_provider(thread_id: str | None, name: str) -> None:
    _ctx(thread_id)["provider"] = name


def add_latency(thread_id: str | None, ms: float) -> None:
    _ctx(thread_id)["latency_ms"] += int(ms)


def record_retrieval(thread_id: str | None, best_score: float, snippets: list[str]) -> None:
    """Called by a RAG tool: log its best similarity score and doc snippets."""
    ctx = _ctx(thread_id)
    ctx["retrieval_scores"].append(best_score)
    ctx["retrieved_docs"].extend(snippets)


def get(thread_id: str | None) -> dict[str, Any]:
    """Read the current turn's accumulated signals."""
    return dict(_ctx(thread_id))


def clear(thread_id: str | None) -> None:
    with _lock:
        _store.pop(_key(thread_id), None)
