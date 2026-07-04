"""
Order + returns data store (SQLite), backed by the 10,000-row
`ecommerce_returns_synthetic_data.csv` dataset.

Each user owns exactly one order. Lookups are scoped by user_id so a signed-in
customer can only ever see their own order/return; an order that belongs to
someone else is reported identically to a nonexistent one (no existence leak).
"""
import csv
import os
import re
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "orders.db")
SEED_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "ecommerce_returns.csv")

_initialized = False

_COLUMNS = [
    "order_id", "product_id", "user_id", "order_date", "return_date",
    "product_category", "product_price", "order_quantity", "return_reason",
    "return_status", "days_to_return", "user_age", "user_gender",
    "user_location", "payment_method", "shipping_method", "discount_applied",
    "shipping_status",
]

_SHIPPING_CHOICES = [
    "Delivered - Left at Front Porch",
    "In Transit - Arriving Soon",
    "Processing - Preparing Shipment",
]


def _normalize_id(value: str, prefix: str) -> str:
    """Map friendly forms (USER-105, 'user 105', 105) to canonical PREFIX00000105."""
    compact = re.sub(r"[^A-Za-z0-9]", "", value or "").upper()
    m = re.match(rf"^{prefix}(\d+)$", compact)
    if m:
        return f"{prefix}{int(m.group(1)):08d}"
    m = re.match(r"^(\d+)$", compact)  # bare number
    if m:
        return f"{prefix}{int(m.group(1)):08d}"
    return compact


def _normalize_user_id(user_id: str) -> str:
    return _normalize_id(user_id, "USER")


def _normalize_order_id(order_id: str) -> str:
    return _normalize_id(order_id, "ORD")


def _derive_shipping_status(order_id: str, return_status: str) -> str:
    if (return_status or "").strip().lower() == "returned":
        return "Returned - Refund Processed"
    digits = re.sub(r"\D", "", order_id) or "0"
    return _SHIPPING_CHOICES[int(digits) % 3]


def _num(value, cast, default=None):
    try:
        if value is None or value == "":
            return default
        return cast(value)
    except (ValueError, TypeError):
        return default


def _ensure_table() -> None:
    global _initialized
    if _initialized:
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        needs_seed = False
        try:
            cursor.execute("SELECT COUNT(*) FROM orders")
            needs_seed = cursor.fetchone()[0] == 0
        except sqlite3.OperationalError:
            cursor.execute(
                """
                CREATE TABLE orders (
                    order_id TEXT PRIMARY KEY,
                    product_id TEXT,
                    user_id TEXT,
                    order_date TEXT,
                    return_date TEXT,
                    product_category TEXT,
                    product_price REAL,
                    order_quantity INTEGER,
                    return_reason TEXT,
                    return_status TEXT,
                    days_to_return REAL,
                    user_age INTEGER,
                    user_gender TEXT,
                    user_location TEXT,
                    payment_method TEXT,
                    shipping_method TEXT,
                    discount_applied REAL,
                    shipping_status TEXT
                )
                """
            )
            cursor.execute("CREATE INDEX idx_orders_user ON orders(user_id)")
            needs_seed = True

        if needs_seed:
            _seed(cursor)
            conn.commit()
    finally:
        conn.close()

    _initialized = True


def _seed(cursor) -> None:
    rows = []
    with open(SEED_CSV_PATH, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            order_id = _normalize_order_id(r["Order_ID"])
            rows.append((
                order_id,
                r.get("Product_ID", ""),
                _normalize_user_id(r["User_ID"]),
                r.get("Order_Date", ""),
                r.get("Return_Date", "") or None,
                r.get("Product_Category", ""),
                _num(r.get("Product_Price"), float, 0.0),
                _num(r.get("Order_Quantity"), int, 0),
                r.get("Return_Reason", "") or None,
                r.get("Return_Status", ""),
                _num(r.get("Days_to_Return"), float, None),
                _num(r.get("User_Age"), int, None),
                r.get("User_Gender", ""),
                r.get("User_Location", ""),
                r.get("Payment_Method", ""),
                r.get("Shipping_Method", ""),
                _num(r.get("Discount_Applied"), float, 0.0),
                _derive_shipping_status(order_id, r.get("Return_Status", "")),
            ))
    cursor.executemany(
        f"INSERT OR IGNORE INTO orders ({', '.join(_COLUMNS)}) "
        f"VALUES ({', '.join(['?'] * len(_COLUMNS))})",
        rows,
    )


def _row_to_dict(row) -> dict:
    return dict(zip(_COLUMNS, row))


def user_exists(user_id: str) -> bool:
    _ensure_table()
    uid = _normalize_user_id(user_id)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM orders WHERE user_id = ? LIMIT 1", (uid,))
        return cur.fetchone() is not None
    finally:
        conn.close()


def get_order(order_id: str, user_id: str) -> dict | None:
    """Look up an order, scoped to its owner. None if not found OR not owned by user_id."""
    _ensure_table()
    oid = _normalize_order_id(order_id)
    uid = _normalize_user_id(user_id)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM orders WHERE order_id = ? AND user_id = ?",
            (oid, uid),
        )
        row = cur.fetchone()
    finally:
        conn.close()
    return _row_to_dict(row) if row else None


def get_return_details(order_id: str, user_id: str) -> dict | None:
    """Same scoping as get_order; caller reads the return_* fields."""
    return get_order(order_id, user_id)


def get_user_orders(user_id: str) -> list[dict]:
    _ensure_table()
    uid = _normalize_user_id(user_id)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM orders WHERE user_id = ? ORDER BY order_date DESC",
            (uid,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    return [_row_to_dict(r) for r in rows]


def get_user_stats(user_id: str) -> dict:
    """Real per-user stats for the sidebar / memory profile."""
    orders = get_user_orders(user_id)
    returned = sum(1 for o in orders if (o["return_status"] or "").lower() == "returned")
    total_spend = sum((o["product_price"] or 0) * (o["order_quantity"] or 0) for o in orders)
    credit = sum(o["discount_applied"] or 0 for o in orders)
    return {
        "order_count": len(orders),
        "return_count": returned,
        "total_spend": round(total_spend, 2),
        "credit": round(credit, 2),
        "recent_products": [o["product_category"] for o in orders[:3]],
    }
