from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
import uuid
from storage.ticket_store import save_ticket, create_ticket_document
from storage.orders_store import get_order, get_return_details
from storage import turn_context


def _user_id_from_config(config: RunnableConfig | None) -> str:
    """Read the signed-in user from the graph config; default to the demo user."""
    if not config:
        return "USER-109"
    return (config.get("configurable", {}) or {}).get("user_id", "USER-109")

@tool
def check_order_status(order_id: str, config: RunnableConfig = None) -> str:
    """
    Check the signed-in customer's order status using their order ID.
    Use this when the user asks where their package is or for tracking info.
    """
    clean_id = order_id.strip()
    if not clean_id:
        return "Error: No Order ID provided. Please ask the user for their Order ID."

    user_id = _user_id_from_config(config)
    order = get_order(clean_id, user_id)
    if order:
        price = order.get("product_price")
        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "an unlisted price"
        qty = order.get("order_quantity") or 1
        return (
            f"Order {order['order_id']}: category '{order['product_category']}', "
            f"product ID {order.get('product_id', 'N/A')}, quantity {qty}, {price_str}. "
            f"Status: {order['shipping_status']}. "
            f"Ordered on {order['order_date']} via {order['shipping_method']} shipping. "
            f"(No specific product name, item list, or delivery date is on record — "
            f"only the category and the fields above.)"
        )
    # Not found OR belongs to another account -> same message, no existence leak.
    return (
        f"I couldn't find order '{clean_id}' on your account. "
        f"Please double-check the ID (it should look like ORD-105)."
    )


@tool
def check_return_details(order_id: str, config: RunnableConfig = None) -> str:
    """
    Check the return/refund details of the signed-in customer's own order:
    whether it was returned, the reason, the return date, and days taken.
    """
    clean_id = order_id.strip()
    if not clean_id:
        return "Error: No Order ID provided. Please ask the user for their Order ID."

    user_id = _user_id_from_config(config)
    order = get_return_details(clean_id, user_id)
    if not order:
        return (
            f"I couldn't find order '{clean_id}' on your account. "
            f"Please double-check the ID (it should look like ORD-105)."
        )

    if (order.get("return_status") or "").lower() != "returned":
        return (
            f"Order {order['order_id']} ({order['product_category']}) has NOT been returned. "
            f"It's currently: {order['shipping_status']}."
        )

    reason = order.get("return_reason") or "not specified"
    ret_date = order.get("return_date") or "an unspecified date"
    days = order.get("days_to_return")
    days_str = f" ({int(days)} days after ordering)" if isinstance(days, (int, float)) else ""
    return (
        f"Order {order['order_id']} ({order['product_category']}) was RETURNED on "
        f"{ret_date}{days_str}. Reason: {reason}. The refund has been processed."
    )

@tool
def initiate_return(order_id: str, reason: str) -> str:
    """
    Initiate a return. REQUIRE both 'order_id' and 'reason' from the user.
    """
    if not order_id or not reason:
        return "Error: Missing order_id or reason. Ask the user for details."

    return (
        f"✅ Return successfully initiated for {order_id}.\n"
        f"Reason recorded: '{reason}'.\n"
        f"A return label has been emailed to you."
    )

@tool
def create_support_ticket(issue: str, config: RunnableConfig = None) -> str:
    """
    Create and store a support ticket for unresolved issues.
    """
    user_id = _user_id_from_config(config)
    ticket_id = str(uuid.uuid4())[:8]

    ticket_doc = create_ticket_document(
        issue=issue,
        ticket_id=ticket_id,
        user_id=user_id,
        category="general",
    )

    save_ticket(ticket_doc)

    return (
        f"🎫 Support ticket created successfully!\n"
        f"Ticket ID: {ticket_id}\n"
        f"Issue: {issue}\n"
        f"Our support team will contact you within 24 hours."
    )


@tool
def escalate_to_human(reason: str = "", config: RunnableConfig = None) -> str:
    """
    Escalate the conversation to a human support agent. Provide a short 'reason'
    summarizing the unresolved issue so the human has context.
    """
    user_id = _user_id_from_config(config)
    issue = reason or "Customer requested a human support agent."

    # Log the escalation as a ticket so the admin dashboard can count it.
    ticket_id = str(uuid.uuid4())[:8]
    save_ticket(create_ticket_document(
        issue=issue, ticket_id=ticket_id, user_id=user_id, category="escalation",
    ))

    return (
        f"📞 I’m escalating this to a human support agent (ticket {ticket_id}).\n"
        f"I’ve logged this for our team. You’ll be contacted shortly. "
        f"Thank you for your patience."
    )


# --- RAG search tools ---
# Each retrieves from its category's FAISS store via `search_with_scores`, then
# records the best similarity score + doc snippets to turn_context so the
# confidence calculation and trace can use them.

def _thread_id_from_config(config: RunnableConfig | None) -> str:
    if not config:
        return "default"
    return (config.get("configurable", {}) or {}).get("thread_id", "default")


def _rag_search(category: str, query: str, unavailable_msg: str,
                config: RunnableConfig | None) -> str:
    try:
        from rag.retriever import search_with_scores
        docs, best_quality = search_with_scores(category, query)
    except Exception:
        # Keep support usable even when optional RAG deps/stores are missing.
        return unavailable_msg

    if not docs:
        return unavailable_msg

    snippets = [d.page_content[:200] for d in docs]
    turn_context.record_retrieval(_thread_id_from_config(config), best_quality, snippets)
    return "\n\n".join(d.page_content for d in docs)


@tool
def search_return_policy(query: str, config: RunnableConfig = None) -> str:
    """
    Useful for questions about refunds, returning items, money back, or exchange policies.
    """
    return _rag_search("returns", query, "Return policy documents are not available.", config)

@tool
def search_shipping_policy(query: str, config: RunnableConfig = None) -> str:
    """
    ONLY use this for delivery, tracking, shipping costs, or carrier info.
    DO NOT use for payments or returns.
    """
    return _rag_search("shipping", query, "Shipping policy documents are not available.", config)

@tool
def search_general_faq(query: str, config: RunnableConfig = None) -> str:
    """
    Useful for general questions about the company, hours of operation, or contact info.
    """
    return _rag_search("general", query, "FAQ documents are not available.", config)



TOOLS = [check_order_status,
         check_return_details,
         initiate_return,
         create_support_ticket,
         escalate_to_human,
         search_return_policy,
         search_shipping_policy,
         search_general_faq]

