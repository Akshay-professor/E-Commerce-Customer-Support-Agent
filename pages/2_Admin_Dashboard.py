"""
Admin Dashboard — operational metrics (Feature 4).

Read-only aggregation over the SQLite trace + ticket tables:
total conversations, open/escalated tickets, most common issues,
tool-usage frequency, and average response time.
"""
import streamlit as st

from storage import trace_store, ticket_store

st.set_page_config(page_title="Admin Dashboard", page_icon="📊", layout="wide")

st.title("📊 Support Admin Dashboard")
st.caption("Operational overview of the AI support agent.")

metrics = trace_store.get_dashboard_metrics()
tickets = ticket_store.get_all_tickets()

open_tickets = [t for t in tickets if t["status"] == "open"]
escalated = [t for t in tickets if t["category"] == "escalation"]

# --- KPI cards ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total conversations", metrics["total_conversations"])
c2.metric("Turns handled", metrics["total_turns"])
c3.metric("Open tickets", len(open_tickets))
c4.metric("Escalated tickets", len(escalated))

c5, c6 = st.columns(2)
c5.metric("Avg response time", f"{metrics['avg_latency_ms']:.0f} ms")
c6.metric("Total tickets", len(tickets))

st.divider()

# --- Charts ---
left, right = st.columns(2)

with left:
    st.subheader("Tool usage frequency")
    if metrics["tool_counts"]:
        st.bar_chart(metrics["tool_counts"])
    else:
        st.info("No tool usage recorded yet.")

with right:
    st.subheader("Conversations by scenario")
    if metrics["scenario_counts"]:
        st.bar_chart(metrics["scenario_counts"])
    else:
        st.info("No scenarios recorded yet.")

st.divider()

# --- Most common issues (from tickets) ---
st.subheader("Most common issues (tickets)")
if tickets:
    from collections import Counter
    issue_counts = Counter(
        (t["issue"][:60] + "…") if len(t["issue"]) > 60 else t["issue"]
        for t in tickets
    )
    st.bar_chart(dict(issue_counts.most_common(8)))

    st.subheader("Recent tickets")
    st.dataframe(
        [
            {
                "ticket": t["ticket_id"],
                "user": t["user_id"],
                "category": t["category"],
                "status": t["status"],
                "issue": t["issue"],
                "created": t["created_at"],
            }
            for t in tickets[:25]
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No tickets created yet.")
