import streamlit as st
import uuid
import re
from langchain_core.messages import HumanMessage, AIMessage
from main import chatbot, retrieve_all_threads
from storage import orders_store, memory_store


def confidence_caption(score) -> str:
    """Small explainable label shown under each assistant answer."""
    if score is None:
        return ""
    if score >= 80:
        emoji, word = "🟢", "High"
    elif score >= 60:
        emoji, word = "🟡", "Medium"
    else:
        emoji, word = "🔴", "Low"
    return f"{emoji} Confidence: {word} ({score:.0f}%)"

# =====================================================
# 1. PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="E-Commerce Customer Support",
    page_icon="🛍️",
    layout="wide"
)

# =====================================================
# 2. GLOBAL THEME & STYLING
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* App background */
.stApp {
    background: #f8fafc;
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit default UI */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* Global text */
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
    font-weight: 700;
}
[data-testid="stSubheader"] {
    color: #0f172a !important;
}
p, li, span, div {
    color: #334155;
}

* {
    transition: all 0.2s ease;
}

/* ================= SIDEBAR ================= */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
    border-right: 1px solid #e2e8f0;
}

[data-testid="stSidebar"] .sidebar-title-wrap {
    margin-top: 8px;
    margin-bottom: 10px;
}

[data-testid="stSidebar"] .sidebar-title {
    color: #0f172a !important;
    font-size: 1.9rem;
    font-weight: 700;
    margin-bottom: 6px;
}

[data-testid="stSidebar"] .sidebar-accent {
    width: 62px;
    height: 3px;
    border-radius: 999px;
    background: linear-gradient(90deg, #60a5fa, #6366f1);
    box-shadow: 0 0 12px rgba(96, 165, 250, 0.75);
    margin-bottom: 8px;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
    color: #475569 !important;
}

[data-testid="stSidebar"] hr.sidebar-divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 12px 0 14px 0;
}

[data-testid="stSidebar"] .stButton button {
    border-radius: 12px;
    border: 1px solid #dbe3f0;
    background: #ffffff;
    color: #334155;
    text-align: left;
}

[data-testid="stSidebar"] .stButton button:hover {
    border-color: #6366f1;
    box-shadow: 0 6px 14px rgba(99, 102, 241, 0.18);
    color: #111827;
}

[data-testid="stSidebar"] .stButton:first-of-type button {
    background: linear-gradient(90deg, #8b5cf6, #6366f1);
    border: 0;
    color: #ffffff;
    box-shadow: 0 10px 20px rgba(99, 102, 241, 0.35);
    font-weight: 600;
}

[data-testid="stSidebar"] .stButton:first-of-type button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 24px rgba(99, 102, 241, 0.45);
}

[data-testid="stSidebar"] .chat-preview {
    font-size: 0.75rem;
    color: #64748b !important;
    margin: -6px 0 10px 10px;
}

[data-testid="stSidebar"] .chat-active {
    border-left: 3px solid #60a5fa;
    padding-left: 8px;
    margin-left: -8px;
}

/* ================= HEADER ================= */
.support-header {
    background: linear-gradient(135deg, #8b5cf6, #6366f1);
    padding: 22px 26px;
    border-radius: 20px;
    margin-bottom: 26px;
    box-shadow: 0 16px 30px rgba(99, 102, 241, 0.28);
}
.support-header h2 {
    color: #fb923c !important;
    font-weight: 700;
    margin-bottom: 6px;
}
.support-header p {
    color: #E0E7FF !important;
    font-size: 15px;
    margin: 0;
}

/* ================= LEFT DASHBOARD (NEW) ================= */
.user-dashboard {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 14px 28px rgba(15, 23, 42, 0.08);
    margin-bottom: 20px;
    border: 1px solid #eef2ff;
}

.user-dashboard:hover {
    transform: translateY(-3px);
    box-shadow: 0 18px 34px rgba(15, 23, 42, 0.12);
}

.user-profile {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 15px;
}
.avatar {
    width: 58px;
    height: 58px;
    background: linear-gradient(135deg, #8b5cf6, #4f46e5);
    color: #ffffff !important;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.35rem;
}

.member-row {
    display: flex;
    align-items: center;
    gap: 6px;
}

.tier-badge {
    background: #fff7d6;
    color: #7c5a00 !important;
    border: 1px solid #f7d977;
    font-size: 0.68rem;
    font-weight: 700;
    border-radius: 999px;
    padding: 2px 8px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-top: 12px;
}

.stat-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 10px 8px;
    text-align: center;
}

.stat-card h4 {
    margin: 0;
    color: #1e293b !important;
}

.stat-card span {
    font-size: 0.78rem;
    color: #64748b;
}

/* Order Status Card */
.order-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px;
    margin-top: 10px;
}
.order-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #64748B;
    margin-bottom: 8px;
}
.order-item {
    font-weight: 600;
    color: #0F172A !important;
    margin-bottom: 4px;
}
.status-badge {
    background: #DEF7EC;
    color: #03543F !important;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* ================= CHAT ================= */
.chat-container {
    background: #ffffff;
    border-radius: 0 0 20px 20px;
    padding: 12px 14px;
    padding-top: 14px;
    border-top: 0;
    border: 1px solid #e2e8f0;
    box-shadow: 0 14px 28px rgba(15, 23, 42, 0.06);
}

/* Chat title bar */
.chat-title {
    background: linear-gradient(90deg, #111827, #1f2937);
    color: white !important;
    padding: 13px 18px;
    border-radius: 20px 20px 0 0;
    font-weight: 700;
    margin-bottom: 0;
    border-bottom: 2px solid #fb923c;
}

.chat-title-main {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #fff !important;
}

.online-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 10px rgba(34, 197, 94, 0.8);
}

.chat-subtitle {
    font-size: 0.78rem;
    color: rgba(255, 255, 255, 0.82) !important;
    margin-top: 2px;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 10px 8px;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    border-radius: 16px;
    padding: 10px 12px;
    max-width: 86%;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    justify-content: flex-end;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    margin-left: auto;
    background: #1e1e2e;
    color: #f8fafc !important;
    border-bottom-right-radius: 4px;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    background: #ffffff;
    color: #0f172a !important;
    border-bottom-left-radius: 4px;
    border-left: 3px solid #fb923c;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) span {
    color: #f8fafc !important;
}

/* ================= QUICK REPLIES ================= */
.quick-replies {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding-bottom: 8px;
}
.quick-replies button {
    background-color: #ffffff;
    color: #4B4ACD;
    border: 1px solid #E0E0E0;
    border-radius: 14px;
    padding: 8px 16px;
    font-size: 0.85rem;
    font-weight: 600;
    white-space: nowrap;
    box-shadow: 0 3px 6px rgba(0,0,0,0.08);
}
.quick-replies button:hover {
    background-color: #F0F0FF;
    border-color: #4B4ACD;
}

/* Typing indicator */
.typing {
    color: #64748b !important;
    font-style: italic;
}

/* ChatGPT-style input */
[data-testid="stChatInput"] {
    margin-top: 10px;
}

[data-testid="stChatInput"] > div {
    border-radius: 999px !important;
    border: 2px solid #fb923c !important;
    background: #111827 !important;
    box-shadow: 0 10px 20px rgba(15, 23, 42, 0.12);
}

[data-testid="stChatInput"] input {
    color: #f8fafc !important;
}

[data-testid="stChatInput"] input::placeholder {
    color: #cbd5e1 !important;
}

[data-testid="stChatInput"] button {
    background: #fb923c !important;
    color: #111827 !important;
    border-radius: 999px !important;
    border: 0 !important;
}

/* Quick Actions */
.quick-actions-grid .stButton > button {
    background: #ffffff !important;
    color: #111827 !important;
    border-radius: 12px !important;
    border: 1px solid #fed7aa !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    text-align: left;
}

.quick-actions-grid .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 20px rgba(15, 23, 42, 0.12);
    background: #fff7ed !important;
    color: #111827 !important;
}

.quick-actions-grid .stButton:nth-of-type(1) > button { border-left: 4px solid #3b82f6 !important; }
.quick-actions-grid .stButton:nth-of-type(2) > button { border-left: 4px solid #f97316 !important; }
.quick-actions-grid .stButton:nth-of-type(3) > button { border-left: 4px solid #8b5cf6 !important; }
.quick-actions-grid .stButton:nth-of-type(4) > button { border-left: 4px solid #ef4444 !important; }

/* ================= BLACK / WHITE / ORANGE THEME OVERRIDES ================= */
.stApp {
    background: #f8f8f8 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0b0b 0%, #151515 100%) !important;
    border-right: 1px solid #27272a !important;
}

[data-testid="stSidebar"] .sidebar-title,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #f5f5f5 !important;
}

[data-testid="stSidebar"] .sidebar-accent {
    background: linear-gradient(90deg, #fb923c, #f97316) !important;
    box-shadow: 0 0 10px rgba(249, 115, 22, 0.65) !important;
}

[data-testid="stSidebar"] .stButton:first-of-type button {
    background: linear-gradient(90deg, #fb923c, #f97316) !important;
    color: #0b0b0b !important;
    font-weight: 700 !important;
}

.support-header,
.chat-title {
    background: linear-gradient(90deg, #0f0f0f, #1a1a1a) !important;
    border: 1px solid #27272a !important;
}

.support-header {
    box-shadow: 0 14px 26px rgba(0, 0, 0, 0.24) !important;
}

.chat-title {
    border-bottom: 2px solid #fb923c !important;
}

.support-header h2,
.chat-title-main,
.chat-subtitle {
    color: #ffffff !important;
}

.user-dashboard,
.chat-container,
.stat-card {
    background: #ffffff !important;
    border-color: #e5e7eb !important;
}

.user-dashboard:hover {
    box-shadow: 0 18px 30px rgba(0, 0, 0, 0.16) !important;
}

.avatar {
    background: linear-gradient(135deg, #0f0f0f, #262626) !important;
    color: #ffffff !important;
}

.tier-badge {
    background: #fff7ed !important;
    color: #9a3412 !important;
    border-color: #fdba74 !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background: #0f0f0f !important;
    color: #ffffff !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    background: #ffffff !important;
    border-left: 3px solid #fb923c !important;
    color: #111827 !important;
}

[data-testid="stChatInput"] > div {
    border: 2px solid #fb923c !important;
    background: #0f0f0f !important;
}

[data-testid="stChatInput"] input,
[data-testid="stChatInput"] input::placeholder {
    color: #f9fafb !important;
}

[data-testid="stChatInput"] button {
    background: #fb923c !important;
    color: #111827 !important;
}

.quick-actions-grid .stButton > button {
    background: #ffffff !important;
    color: #ffffff !important;
    border: 1px solid #fed7aa !important;
}

.quick-actions-grid .stButton > button:hover {
    background: #fff7ed !important;
    color: #ffffff !important;
}

.quick-actions-grid .stButton > button * {
    color: #ffffff !important;
}

.quick-actions-grid .stButton:nth-of-type(1) > button,
.quick-actions-grid .stButton:nth-of-type(2) > button,
.quick-actions-grid .stButton:nth-of-type(3) > button,
.quick-actions-grid .stButton:nth-of-type(4) > button {
    border-left: 4px solid #fb923c !important;
}

/* ================= FINAL HARD OVERRIDES ================= */
.support-header h2 {
    color: #fb923c !important;
}

[data-testid="stAppViewContainer"] .main .stButton > button {
    background: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid #fb923c !important;
}

[data-testid="stAppViewContainer"] .main .stButton > button * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: #ffffff !important;
    color: #334155 !important;
    border: 1px solid #dbe3f0 !important;
}

[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: linear-gradient(90deg, #fb923c, #f97316) !important;
    color: #0b0b0b !important;
    border: 0 !important;
}

/* ================= ULTIMATE OVERRIDES (DO NOT MOVE) ================= */
.support-header h2,
.support-header h2 * {
    color: #fb923c !important;
}

[data-testid="stAppViewContainer"] .main .quick-actions-grid .stButton > button {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #fed7aa !important;
}

[data-testid="stAppViewContainer"] .main .quick-actions-grid .stButton > button * {
    color: #111827 !important;
}

[data-testid="stAppViewContainer"] .main .quick-actions-grid .stButton > button:hover {
    background: #fff7ed !important;
    color: #111827 !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. SESSION STATE
# =====================================================
def init_state():
    defaults = {
        "chat_history": [],
        "thread_id": str(uuid.uuid4()),
        "chat_threads": retrieve_all_threads(),
        "pending_user_input": None,
        "is_typing": False,
        "user_id": None,   # set once the customer logs in with their User ID
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =====================================================
# 3b. LOGIN GATE — customer types their User ID, then sees only their own data
# =====================================================
def logout():
    st.session_state.user_id = None
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.pending_user_input = None
    st.session_state.is_typing = False
    st.rerun()

if not st.session_state.user_id:
    st.markdown(
        """
        <div class="support-header">
            <h2>🛍️ ShopSmart Customer Support</h2>
            <p>Please sign in with your User ID to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("login_form"):
        typed = st.text_input("Your User ID", placeholder="e.g. USER-105")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        candidate = (typed or "").strip()
        if candidate and orders_store.user_exists(candidate):
            st.session_state.user_id = orders_store._normalize_user_id(candidate)
            # Seed/refresh this customer's memory profile on login.
            try:
                memory_store.upsert_profile(st.session_state.user_id)
            except Exception:
                pass
            st.rerun()
        else:
            st.error("That User ID wasn't found. Try one like USER-105 (any number 0–9999).")
    st.caption("Demo login — no password. Your session only exposes your own order and returns.")
    st.stop()

USER_ID = st.session_state.user_id

# =====================================================
# 4. HELPERS
# =====================================================
def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    messages = []
    if state.values and "messages" in state.values:
        for msg in state.values["messages"]:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                messages.append({"role": "assistant", "content": msg.content})
    return messages

def reset_chat():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.pending_user_input = None
    st.session_state.is_typing = False
    st.rerun()


def _content_to_text(content) -> str:
    """Coerce an LLM message's content to a plain string.

    LangChain message `.content` may be a string OR a list of content blocks
    (dicts like {"type": "text", "text": ...}). Streamlit's .strip()/append
    downstream needs a string, so normalize both shapes here.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if text:
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def format_agent_error(error: Exception) -> str:
    message = str(error)
    if "RateLimitError" in message or "rate_limit" in message.lower() or "429" in message:
        wait_match = re.search(r"(try again in|retry in)\s*([^\.\n]+)", message, flags=re.IGNORECASE)
        if "limit: 0" in message.lower() or "resource_exhausted" in message.lower():
            return (
                "This API key currently has no usable quota for the selected model. "
                "Please enable billing or use another key/project, then retry."
            )
        if wait_match:
            wait_for = wait_match.group(2).strip()
            return (
                f"LLM API limit reached. Please try again in about {wait_for}. "
                "You can switch provider/model in your .env using LLM_PROVIDER and model variables "
                "(for example: LLM_PROVIDER=gemini, GEMINI_MODEL=gemini-2.0-flash)."
            )
        return (
            "LLM API limit reached for now. Please retry shortly, or switch provider/model in your .env "
            "using LLM_PROVIDER with GEMINI_MODEL or GROQ_MODEL."
        )

    return "I hit a temporary error while generating a response. Please try again."

# =====================================================
# 5. SIDEBAR
# =====================================================
st.sidebar.markdown(
    """
    <div class="sidebar-title-wrap">
        <div class="sidebar-title">🛍️ Help Center</div>
        <div class="sidebar-accent"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("E-Commerce Support Assistant")

if st.sidebar.button("➕ New Support Chat", use_container_width=True):
    reset_chat()

st.sidebar.markdown('<hr class="sidebar-divider"/>', unsafe_allow_html=True)

# =====================================================
# 6. HEADER
# =====================================================
st.markdown("""
<div class="support-header">
    <h2>🛍️ E-Commerce Customer Support</h2>
    <p>Orders, shipping, returns, and refunds</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 7. MAIN LAYOUT
# =====================================================
chat_col, info_col = st.columns([2, 1], gap="large")

# ---------------- LEFT COLUMN (CHAT APP) ----------------
with chat_col:
    st.markdown(
        '''
        <div class="chat-title">
            <div class="chat-title-main"><span>💬 Sammy</span><span class="online-dot"></span></div>
            <div class="chat-subtitle">AI Support Agent</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    chat_box = st.container(height=320)

    with chat_box:
        if not st.session_state.chat_history:
            st.info(f"👋 Hi {USER_ID}! I'm your support assistant. How can I help you today?")
        
        for msg in st.session_state.chat_history:
            avatar = "🛍️" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                cap = confidence_caption(msg.get("confidence")) if msg["role"] == "assistant" else ""
                if cap:
                    st.caption(cap)

        if st.session_state.is_typing:
            with st.chat_message("assistant", avatar="🛍️"):
                st.markdown("<span class='typing'>Support agent is typing…</span>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Message Sammy")
    if user_input and user_input.strip():
        clean_input = user_input.strip()
        st.session_state.chat_history.append(
            {"role": "user", "content": clean_input}
        )
        st.session_state.pending_user_input = clean_input
        st.session_state.is_typing = True
        st.rerun()

# ---------------- RIGHT COLUMN (USER STATUS + ACTIONS) ----------------
with info_col:
    # --- 1. User Profile Card (real stats for the signed-in customer) ---
    stats = orders_store.get_user_stats(USER_ID)
    initials = "".join([p[0] for p in re.findall(r"[A-Za-z]+|\d+", USER_ID)][:2]).upper() or "U"
    st.markdown(f"""
    <div class="user-dashboard">
        <div class="user-profile">
            <div class="avatar">{initials}</div>
            <div>
                <div style="font-weight:600; font-size:1.1rem; color:#0f172a;">{USER_ID}</div>
                <div class="member-row">
                    <div style="font-size:0.85rem; color:#64748B;">Signed-in customer</div>
                    <span class="tier-badge">VERIFIED</span>
                </div>
            </div>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <h4>{stats['order_count']}</h4>
                <span>Orders</span>
            </div>
            <div class="stat-card">
                <h4>{stats['return_count']}</h4>
                <span>Returns</span>
            </div>
            <div class="stat-card">
                <h4>${stats['total_spend']:.0f}</h4>
                <span>Spend</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Switch user / Log out", use_container_width=True):
        logout()

    # --- 3. Smart Actions ---
    st.markdown("##### ⚡ Quick Actions")
    st.markdown('<div class="quick-actions-grid">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    # These buttons act as triggers to inject text into the chat loop
    with col1:
        if st.button("📍 Track Order", use_container_width=True):
            st.session_state.pending_user_input = "I want to track my order ?"
            st.session_state.is_typing = True
            st.rerun()
    with col2:
        if st.button("↩️ Return Item", use_container_width=True):
            st.session_state.pending_user_input = "I want to return order"
            st.session_state.is_typing = True
            st.rerun()
    with col1:
        if st.button("📃 Policies", use_container_width=True):
            st.session_state.pending_user_input = "What is your return policy?"
            st.session_state.is_typing = True
            st.rerun()
    with col2:
        if st.button("☎️ Support", use_container_width=True):
            st.session_state.pending_user_input = "I need to speak to a human"
            st.session_state.is_typing = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 8. ASSISTANT RESPONSE LOGIC
# =====================================================
if st.session_state.pending_user_input:
    user_input = st.session_state.pending_user_input
    # Clear pending state immediately to prevent loops
    st.session_state.pending_user_input = None
    
    CONFIG = {"configurable": {"thread_id": st.session_state.thread_id, "user_id": USER_ID}}

    ai_response = ""
    confidence = None
    try:
        with chat_col:
            with chat_box:
                with st.chat_message("assistant"):
                    def stream_response():
                        # Stream only textual assistant chunks from the agent node.
                        for chunk, meta in chatbot.stream(
                            {"messages": [HumanMessage(content=user_input)], "user_id": USER_ID},
                            config=CONFIG,
                            stream_mode="messages"
                        ):
                            if meta.get("langgraph_node") != "agent":
                                continue
                            content = chunk.content
                            if isinstance(content, str) and content:
                                yield content
                            elif isinstance(content, list):
                                for part in content:
                                    if isinstance(part, dict) and part.get("type") == "text":
                                        text = part.get("text", "")
                                        if text:
                                            yield text

                    # Write stream to UI and capture full response
                    result = st.write_stream(stream_response())
                    if isinstance(result, str):
                        ai_response = result
                    elif result is not None:
                        ai_response = str(result)
    except Exception as err:
        ai_response = format_agent_error(err)
    finally:
        # Always clear typing indicator, even if generation fails.
        st.session_state.is_typing = False

    # After the graph settles (agent -> validate, plus any single regeneration),
    # read the authoritative final answer + confidence from persisted state.
    # This keeps saved history clean even if a regeneration streamed twice.
    try:
        final_state = chatbot.get_state(CONFIG).values
        msgs = final_state.get("messages", [])
        if msgs and isinstance(msgs[-1], AIMessage) and msgs[-1].content:
            ai_response = _content_to_text(msgs[-1].content)
        confidence = final_state.get("confidence")
    except Exception:
        pass

    # Defensive: the graph/stream can hand back list-of-blocks content.
    ai_response = _content_to_text(ai_response)

    # Save complete response to history
    if ai_response.strip():
        st.session_state.chat_history.append(
            {"role": "assistant", "content": ai_response, "confidence": confidence}
        )
    else:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "I am here. Please share your issue and I will help you."}
        )
    
    # Save thread ID to session if new
    if st.session_state.thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(st.session_state.thread_id)


    st.rerun()
