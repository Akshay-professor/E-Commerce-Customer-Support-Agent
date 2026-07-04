from typing import TypedDict, Annotated
import sqlite3
import os
import time
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import (
    BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage,
)
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import TOOLS
from prompt import SYSTEM_PROMPT, GRADER_PROMPT, REGENERATE_NUDGE, SUMMARY_PROMPT
from storage import turn_context, trace_store, memory_store

load_dotenv()

# Model Setup - preferred provider (LLM_PROVIDER) tried first, other kept as fallback.
# We keep two views of each provider: tool-bound (for the agent) and plain (for
# the grader/summary calls, which must return text, not tool calls).
available_models: dict[str, object] = {}
plain_models_by_name: dict[str, object] = {}

# max_retries=0 + a short timeout are important: if a provider is rate-limited
# (e.g. Gemini's small free-tier quota), we want to fail over to the other
# provider INSTANTLY rather than let the client sit through 30s backoff retries,
# which otherwise stacks to minutes across the agent + grader + summary calls.
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
if groq_api_key:
    groq_chat = ChatGroq(
        model=groq_model, temperature=0, api_key=groq_api_key,
        max_retries=0, timeout=30,
    )
    available_models["groq"] = groq_chat.bind_tools(TOOLS)
    plain_models_by_name["groq"] = groq_chat

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
if gemini_api_key:
    gemini_chat = ChatGoogleGenerativeAI(
        model=gemini_model, temperature=0, google_api_key=gemini_api_key,
        max_retries=0, timeout=30,
    )
    available_models["gemini"] = gemini_chat.bind_tools(TOOLS)
    plain_models_by_name["gemini"] = gemini_chat

preferred_provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
provider_order = [preferred_provider] + [p for p in ("groq", "gemini") if p != preferred_provider]
tool_enabled_models: list[tuple[str, object]] = [
    (name, available_models[name]) for name in provider_order if name in available_models
]
# Auxiliary LLM calls (validation grader + memory summarizer) are observability
# features, not the customer's answer. We pin them to Groq ONLY (fast, per-minute
# quota that resets) and never let them fall through to Gemini, whose free tier is
# only ~20 requests/DAY. This reserves Gemini's scarce daily quota for real
# answers during the live demo. If Groq is unavailable/rate-limited, the aux call
# is skipped gracefully (grader -> N/A, summary -> skipped) rather than punished.
aux_models: list[tuple[str, object]] = (
    [("groq", plain_models_by_name["groq"])] if "groq" in plain_models_by_name else []
)

DEFAULT_USER_ID = "guest"

# Shown when every provider fails; validate scores it low (it's not a real answer).
PROVIDER_DOWN_MESSAGE = (
    "I am here to help, but the AI providers are temporarily unavailable right now. "
    "Please try again in a minute, and I can continue from where we left off."
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str                 # signed-in customer (from config each turn)
    validation_attempts: int     # drives the single-regeneration loop
    confidence: float            # last turn's confidence score (0-100)
    validation_result: str       # last turn's grader verdict


# --------------------------------------------------------------------------
# Agent node
# --------------------------------------------------------------------------

def agent_node(state: ChatState, config=None):
    input_messages = state["messages"]
    user_id = state.get("user_id") or DEFAULT_USER_ID
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    is_new_turn = bool(input_messages) and isinstance(input_messages[-1], HumanMessage)

    # A fresh user turn: reset per-turn observability.
    if is_new_turn:
        turn_context.reset(thread_id)

    # Enhanced memory: prepend what we know about this customer.
    memory_block = memory_store.build_memory_context(user_id)
    system_text = SYSTEM_PROMPT + (("\n\n" + memory_block) if memory_block else "")

    prompt_messages: list[BaseMessage] = [SystemMessage(content=system_text)]

    # If validation just failed, we're regenerating: nudge toward grounding.
    if state.get("validation_attempts", 0) >= 1 and isinstance(input_messages[-1], AIMessage):
        prompt_messages.append(SystemMessage(content=REGENERATE_NUDGE))

    prompt_messages += input_messages

    for provider_name, model in tool_enabled_models:
        try:
            t0 = time.perf_counter()
            response = model.invoke(prompt_messages)
            turn_context.add_latency(thread_id, (time.perf_counter() - t0) * 1000)
            turn_context.set_provider(thread_id, provider_name)
            print(f"[agent_node] responded via provider: {provider_name}")
            out = {"messages": [response]}
            if is_new_turn:
                # Reset per-turn routing/observability fields.
                out["validation_attempts"] = 0
                out["validation_result"] = ""
            return out
        except Exception as exc:
            print(f"[agent_node] provider '{provider_name}' failed: {exc}")
            continue

    fallback = AIMessage(
        content=PROVIDER_DOWN_MESSAGE
    )
    return {"messages": [fallback], "validation_attempts": 0}


def route_after_agent(state: ChatState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "validate"


# --------------------------------------------------------------------------
# Turn-analysis helpers (used by the validate node)
# --------------------------------------------------------------------------

_ERROR_MARKERS = ("error", "couldn't find", "not available", "missing")


def _current_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Messages produced since (and including) the latest human message."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def _last_human_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _turn_tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    return [m for m in _current_turn_messages(messages) if isinstance(m, ToolMessage)]


def _turn_tools_used(messages: list[BaseMessage]) -> list[str]:
    used: list[str] = []
    for m in _current_turn_messages(messages):
        for call in getattr(m, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                used.append(name)
    return used


_SCENARIO_BY_TOOL = {
    "check_order_status": "order_status",
    "check_return_details": "returns",
    "initiate_return": "returns",
    "search_return_policy": "returns",
    "search_shipping_policy": "shipping",
    "search_general_faq": "faq",
    "escalate_to_human": "escalation",
    "create_support_ticket": "escalation",
}


def _derive_scenario(tools_used: list[str]) -> str:
    for t in tools_used:
        if t in _SCENARIO_BY_TOOL:
            return _SCENARIO_BY_TOOL[t]
    return "general"


def _collect_evidence(messages: list[BaseMessage], ctx: dict) -> str:
    parts: list[str] = []
    for tm in _turn_tool_messages(messages):
        content = tm.content if isinstance(tm.content, str) else str(tm.content)
        parts.append(f"[tool:{getattr(tm, 'name', 'tool')}] {content}")
    for doc in ctx.get("retrieved_docs", []):
        parts.append(f"[doc] {doc}")
    return "\n".join(parts).strip()


def _grade(question: str, answer: str, evidence: str) -> str:
    """Return 'PASS', 'FAIL', or 'N/A' (no factual claims / no grader available)."""
    if not answer.strip():
        return "N/A"
    if not evidence:
        # No tool/doc evidence gathered -> the reply is a greeting/clarifying
        # question; nothing to hallucinate against.
        return "N/A"

    prompt = GRADER_PROMPT.format(evidence=evidence, question=question, answer=answer)
    for _, model in aux_models:  # Groq-only; reserve Gemini's daily quota for answers
        try:
            # Gemini rejects a lone SystemMessage ("contents are required"), so
            # send the task as a HumanMessage.
            verdict = model.invoke([HumanMessage(content=prompt)]).content
            text = (verdict if isinstance(verdict, str) else str(verdict)).strip().upper()
            if text.startswith("FAIL"):
                return "FAIL"
            if text.startswith("PASS"):
                return "PASS"
        except Exception as exc:
            print(f"[validate] grader via plain model failed: {exc}")
            continue
    return "N/A"  # grader unavailable -> don't punish the answer


def _compute_confidence(messages: list[BaseMessage], ctx: dict, validation_label: str) -> float:
    # Retrieval quality: best normalized score this turn, neutral if no RAG used.
    scores = ctx.get("retrieval_scores", [])
    retrieval_q = max(scores) if scores else 1.0

    # Tool execution: fraction of tool outputs without an error marker.
    tool_msgs = _turn_tool_messages(messages)
    if tool_msgs:
        ok = sum(
            1 for tm in tool_msgs
            if not any(mark in (tm.content or "").lower() for mark in _ERROR_MARKERS)
        )
        tool_q = ok / len(tool_msgs)
    else:
        tool_q = 1.0

    validation_q = {
        "PASS": 1.0, "N/A": 1.0, "REGENERATED_PASS": 0.7, "FAIL": 0.3,
    }.get(validation_label, 1.0)

    score = 0.35 * retrieval_q + 0.30 * tool_q + 0.35 * validation_q
    return round(score * 100, 1)


def _maybe_update_memory(state: ChatState, thread_id: str, user_id: str):
    """Refresh the customer profile each turn; summarize periodically."""
    try:
        memory_store.upsert_profile(user_id)
    except Exception as exc:
        print(f"[memory] profile upsert failed: {exc}")

    human_turns = sum(1 for m in state["messages"] if isinstance(m, HumanMessage))
    if human_turns == 0 or human_turns % 3 != 0:
        return

    # Summarize the recent conversation (best-effort, one plain LLM call).
    transcript = []
    for m in state["messages"][-12:]:
        if isinstance(m, HumanMessage):
            transcript.append(f"Customer: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            transcript.append(f"Agent: {m.content}")
    convo = "\n".join(transcript)
    prompt = SUMMARY_PROMPT.format(conversation=convo)

    for _, model in aux_models:  # Groq-only; reserve Gemini's daily quota for answers
        try:
            out = model.invoke([HumanMessage(content=prompt)]).content
            text = out if isinstance(out, str) else str(out)
            summary_line, pref = text, None
            for line in text.splitlines():
                if line.strip().upper().startswith("PREFERENCE:"):
                    pref = line.split(":", 1)[1].strip()
                    summary_line = text.replace(line, "").strip()
                    break
            memory_store.save_summary(user_id, thread_id, summary_line.strip())
            if pref and "=" in pref:
                k, v = pref.split("=", 1)
                memory_store.set_preference(user_id, k.strip(), v.strip())
            return
        except Exception as exc:
            print(f"[memory] summary failed: {exc}")
            continue


# --------------------------------------------------------------------------
# Validation node
# --------------------------------------------------------------------------

def validate_node(state: ChatState, config=None):
    messages = state["messages"]
    final = messages[-1]
    answer = final.content if isinstance(final.content, str) else str(final.content)
    user_id = state.get("user_id") or DEFAULT_USER_ID
    attempts = state.get("validation_attempts", 0)
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
    ctx = turn_context.get(thread_id)

    # Provider-down fallback isn't a real answer: mark it low, skip grading/regen.
    if answer.strip() == PROVIDER_DOWN_MESSAGE.strip():
        thread_id_pd = thread_id
        try:
            trace_store.log_trace(
                thread_id=thread_id_pd, user_id=user_id, scenario="general",
                tools_used=[], retrieved_docs=[], llm_provider="",
                latency_ms=ctx.get("latency_ms", 0),
                validation_result="PROVIDER_DOWN", confidence=0.0, final_answer=answer,
            )
        except Exception:
            pass
        turn_context.clear(thread_id)
        return {"confidence": 0.0, "validation_result": "PROVIDER_DOWN"}

    evidence = _collect_evidence(messages, ctx)
    question = _last_human_text(messages)
    verdict = _grade(question, answer, evidence)

    # Failed and we still have a regeneration budget -> loop back to agent once.
    if verdict == "FAIL" and attempts < 1:
        return {"validation_attempts": attempts + 1}

    # Terminal for this turn: settle the label, confidence, trace, and memory.
    if verdict == "PASS" and attempts >= 1:
        label = "REGENERATED_PASS"
    else:
        label = verdict  # PASS / FAIL / N/A

    confidence = _compute_confidence(messages, ctx, label)
    tools_used = _turn_tools_used(messages)
    scenario = _derive_scenario(tools_used)

    updated_messages = None
    if not answer.strip():
        # Guard: never surface a blank reply (e.g. a weak model returning empty
        # content after a tool call).
        answer = ("I'm sorry, I wasn't able to put together a full answer just now. "
                  "Could you rephrase, or would you like me to create a support ticket?")
        updated_messages = [AIMessage(content=answer, id=getattr(final, "id", None))]
    elif label == "FAIL":
        # Retry exhausted and still ungrounded: soft-escalate rather than assert.
        note = ("\n\n_I want to make sure this is accurate — let me have a human "
                "teammate confirm the details for you._")
        answer = answer + note
        updated_messages = [AIMessage(content=answer, id=getattr(final, "id", None))]

    try:
        trace_store.log_trace(
            thread_id=thread_id, user_id=user_id, scenario=scenario,
            tools_used=tools_used, retrieved_docs=ctx.get("retrieved_docs", []),
            llm_provider=ctx.get("provider", ""), latency_ms=ctx.get("latency_ms", 0),
            validation_result=label, confidence=confidence, final_answer=answer,
        )
    except Exception as exc:
        print(f"[validate] trace logging failed: {exc}")

    _maybe_update_memory(state, thread_id, user_id)
    turn_context.clear(thread_id)

    out = {"confidence": confidence, "validation_result": label}
    if updated_messages is not None:
        out["messages"] = updated_messages
    return out


def route_after_validate(state: ChatState):
    # validate_node leaves validation_result empty ONLY when it wants a
    # regeneration; a settled turn always sets a non-empty label.
    if not state.get("validation_result"):
        return "agent"
    return END


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

workflow = StateGraph(ChatState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.add_node("validate", validate_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent", route_after_agent, {"tools": "tools", "validate": "validate"}
)
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "validate", route_after_validate, {"agent": "agent", END: END}
)

chatbot = workflow.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    # Helper to fetch unique thread IDs from DB
    try:
        with sqlite3.connect("chatbot.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            return [row[0] for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        # Table doesn't exist yet on first run
        return []
