"""
Conversation Traces — inspect every logged turn (Feature 2).

Reads `conversation_traces` written by the validate node and shows a table plus
per-turn detail (tools used, retrieved docs, provider, latency, validation,
confidence, final answer).
"""
import streamlit as st

from storage import trace_store

st.set_page_config(page_title="Conversation Traces", page_icon="🧾", layout="wide")

st.title("🧾 Conversation Traces")
st.caption("Every agent turn is logged for observability and debugging.")

traces = trace_store.get_recent_traces(limit=200)

if not traces:
    st.info("No conversations traced yet. Chat with the assistant on the main page, then come back.")
    st.stop()

# Summary row
col1, col2, col3 = st.columns(3)
col1.metric("Turns logged", len(traces))
col2.metric("Conversations", len({t["thread_id"] for t in traces}))
avg_conf = sum(t["confidence"] for t in traces) / len(traces)
col3.metric("Avg confidence", f"{avg_conf:.0f}%")

st.divider()

# Compact table
table_rows = [
    {
        "time": t["ts"],
        "user": t["user_id"],
        "scenario": t["scenario"],
        "provider": t["llm_provider"],
        "latency_ms": t["latency_ms"],
        "validation": t["validation_result"],
        "confidence": t["confidence"],
        "tools": ", ".join(t["tools_used"]) or "—",
    }
    for t in traces
]
st.dataframe(table_rows, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Turn detail")

for t in traces:
    label = (f"#{t['id']} · {t['scenario']} · {t['validation_result']} · "
             f"{t['confidence']:.0f}% · {t['ts']}")
    with st.expander(label):
        st.markdown(f"**User:** {t['user_id']}  |  **Thread:** `{t['thread_id']}`")
        st.markdown(f"**Provider:** {t['llm_provider'] or '—'}  |  "
                    f"**Latency:** {t['latency_ms']} ms  |  "
                    f"**Validation:** {t['validation_result']}  |  "
                    f"**Confidence:** {t['confidence']:.0f}%")
        st.markdown(f"**Tools used:** {', '.join(t['tools_used']) or '—'}")
        if t["retrieved_docs"]:
            st.markdown("**Retrieved documents:**")
            for d in t["retrieved_docs"]:
                st.text(d)
        st.markdown("**Final answer:**")
        st.write(t["final_answer"])
