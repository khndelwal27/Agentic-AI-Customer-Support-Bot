import json
import pickle
import datetime
import pandas as pd
import re
from rag_engine import retrieve_similar

from sklearn.metrics.pairwise import cosine_similarity

# ================================
# LOAD CONFIG + KNOWLEDGE BASE
# ================================

with open("config.json") as f:
    CONFIG = json.load(f)

with open("knowledge.json") as f:
    KNOWLEDGE = json.load(f)

ESCALATION_FILE = "escalations.json"
FEEDBACK_FILE = "feedback.json"

# ================================
# LOAD TRAINED MODEL + VECTORIZER
# ================================

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("\n✅ Model Loaded Successfully!")
print("Intent Classes:", model.classes_)

# ================================
# LOAD FULL DATASET FOR RETRIEVAL
# ================================

tickets_df = pd.read_csv("data/customer_support_tickets.csv").dropna()

# AUTO DETECT TEXT COLUMN
possible_text_cols = ["ticket_text", "text", "description", "Ticket Description"]

TEXT_COLUMN = None
for col in possible_text_cols:
    if col in tickets_df.columns:
        TEXT_COLUMN = col
        break

if TEXT_COLUMN is None:
    raise ValueError("❌ No valid text column found in Kaggle dataset!")

print("✅ Using Text Column:", TEXT_COLUMN)

ticket_texts = tickets_df[TEXT_COLUMN].astype(str).tolist()
ticket_vectors = vectorizer.transform(ticket_texts)

print("✅ Retrieval Dataset Loaded:", len(ticket_texts), "tickets")

# ================================
# LOAD REALISTIC ORDERS DATABASE
# ================================

orders_db = pd.read_csv("data/orders.csv")
print("✅ Orders Dataset Loaded:", len(orders_db), "orders")

# ================================
# MEMORY AGENT
# ================================

memory = {
    "awaiting_order_id": False,
    "last_intent": None
}

# ================================
# RETRIEVAL AGENT
# ================================

def retrieve_similar_tickets(user_query, top_k=3):
    query_vec = vectorizer.transform([user_query])
    sims = cosine_similarity(query_vec, ticket_vectors).flatten()

    top_indices = sims.argsort()[-top_k:][::-1]

    return [ticket_texts[i] for i in top_indices]

# ================================
# TOOL AGENT: ORDER STATUS CHECK
# ================================

def track_order(order_id, query=None):

    import pandas as pd

    order_id = order_id.upper()

    # AUTO DETECT ORDER COLUMN
    order_col = None
    for col in orders_db.columns:
        if "order" in col and "id" in col:
            order_col = col
            break

    if order_col is None:
        return f"❌ Order ID column not found. Columns: {list(orders_db.columns)}"

    # FILTER DATA
    result = orders_db[orders_db[order_col].astype(str).str.upper() == order_id]

    # ✅ VERY IMPORTANT CHECK
    if result.empty:
        return "❌ Order ID not found."

    # NOW SAFE
    row = result.iloc[0]

    delivery = row.get("estimated_delivery", "Not Available")
    if pd.isna(delivery):
        delivery = "Not Available"

    return f"""
📦 Order Details:

👤 Customer: {row.get('customer_name', 'N/A')}
📌 Product: {row.get('product', 'N/A')} ({row.get('category', 'N/A')})
💳 Payment: {row.get('payment_method', 'N/A')}
💰 Amount: ₹{row.get('total_amount', 'N/A')}

🚚 Status: {row.get('status', 'N/A')}
📅 Delivery: {delivery}
💸 Refund: {row.get('refund_status', 'N/A')}
📍 City: {row.get('shipping_city', 'N/A')}
"""
def search_orders_by_text(query):

    query = query.lower()

    # search in multiple columns
    filtered = orders_db[
        orders_db.apply(
            lambda row: query in str(row).lower(), axis=1
        )
    ]

    if filtered.empty:
        return None

    # take first match
    row = filtered.iloc[0]

    delivery = row.get("estimated_delivery", "Not Available")
    if pd.isna(delivery):
        delivery = "Not Available"

    return f"""
📦 Order Found:

👤 Customer: {row.get('customer_name', 'N/A')}
📌 Product: {row.get('product', 'N/A')} ({row.get('category', 'N/A')})
💳 Payment: {row.get('payment_method', 'N/A')}
💰 Amount: ₹{row.get('total_amount', 'N/A')}

🚚 Status: {row.get('status', 'N/A')}
📅 Delivery: {delivery}
💸 Refund: {row.get('refund_status', 'N/A')}
📍 City: {row.get('shipping_city', 'N/A')}
"""
def search_orders_by_text(query):

    query = query.lower()

    # search in multiple columns
    filtered = orders_db[
        orders_db.apply(
            lambda row: query in str(row).lower(), axis=1
        )
    ]

    if filtered.empty:
        return None

    # take first match
    row = filtered.iloc[0]

    delivery = row.get("estimated_delivery", "Not Available")
    if pd.isna(delivery):
        delivery = "Not Available"

    return f"""
📦 Order Found:

👤 Customer: {row.get('customer_name', 'N/A')}
📌 Product: {row.get('product', 'N/A')} ({row.get('category', 'N/A')})
💳 Payment: {row.get('payment_method', 'N/A')}
💰 Amount: ₹{row.get('total_amount', 'N/A')}

🚚 Status: {row.get('status', 'N/A')}
📅 Delivery: {delivery}
💸 Refund: {row.get('refund_status', 'N/A')}
📍 City: {row.get('shipping_city', 'N/A')}
"""
def search_orders_by_text(query):

    query = query.lower()

    # search in multiple columns
    filtered = orders_db[
        orders_db.apply(
            lambda row: query in str(row).lower(), axis=1
        )
    ]

    if filtered.empty:
        return None

    # take first match
    row = filtered.iloc[0]

    delivery = row.get("estimated_delivery", "Not Available")
    if pd.isna(delivery):
        delivery = "Not Available"

    return f"""
📦 Order Found:

👤 Customer: {row.get('customer_name', 'N/A')}
📌 Product: {row.get('product', 'N/A')} ({row.get('category', 'N/A')})
💳 Payment: {row.get('payment_method', 'N/A')}
💰 Amount: ₹{row.get('total_amount', 'N/A')}

🚚 Status: {row.get('status', 'N/A')}
📅 Delivery: {delivery}
💸 Refund: {row.get('refund_status', 'N/A')}
📍 City: {row.get('shipping_city', 'N/A')}
"""
def search_orders_by_text(query):

    query = query.lower()

    # search in multiple columns
    filtered = orders_db[
        orders_db.apply(
            lambda row: query in str(row).lower(), axis=1
        )
    ]

    if filtered.empty:
        return None

    # take first match
    row = filtered.iloc[0]

    delivery = row.get("estimated_delivery", "Not Available")
    if pd.isna(delivery):
        delivery = "Not Available"

    return f"""
📦 Order Found:

👤 Customer: {row.get('customer_name', 'N/A')}
📌 Product: {row.get('product', 'N/A')} ({row.get('category', 'N/A')})
💳 Payment: {row.get('payment_method', 'N/A')}
💰 Amount: ₹{row.get('total_amount', 'N/A')}

🚚 Status: {row.get('status', 'N/A')}
📅 Delivery: {delivery}
💸 Refund: {row.get('refund_status', 'N/A')}
📍 City: {row.get('shipping_city', 'N/A')}
"""
# ================================
# ESCALATION AGENT
# ================================

def escalate_to_human(query, intent, confidence):
    with open(ESCALATION_FILE, "r") as f:
        data = json.load(f)

    ticket_id = f"TKT{len(data)+1:04d}"

    ticket = {
        "ticket_id": ticket_id,
        "query": query,
        "intent": intent,
        "confidence": float(confidence),
        "time": str(datetime.datetime.now())
    }

    data.append(ticket)

    with open(ESCALATION_FILE, "w") as f:
        json.dump(data, f, indent=2)

    return ticket_id

# ================================
# FEEDBACK AGENT
# ================================

def save_feedback(intent, success):
    with open(FEEDBACK_FILE, "r") as f:
        data = json.load(f)

    data.append({
        "intent": intent,
        "success": success,
        "time": str(datetime.datetime.now())
    })

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ================================
# HYBRID INTENT AGENT
# ================================

def detect_intent(user_input):

    text = user_input.lower()

    keyword_map = {
        "billing": "Billing inquiry",
        "payment": "Billing inquiry",
        "invoice": "Billing inquiry",

        "refund": "Refund request",
        "return": "Refund request",

        "cancel": "Cancellation request",

        "product": "Product inquiry",
        "replacement": "Product inquiry",

        "error": "Technical issue",
        "bug": "Technical issue",
        "app": "Technical issue"
    }

    for word, label in keyword_map.items():
        if word in text and label in model.classes_:
            return label, 0.95

    X = vectorizer.transform([user_input])
    intent = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])

    return intent, confidence

# ================================
# RESPONSE GENERATION AGENT
# ================================

def generate_dynamic_response(intent, user_query):

    examples = retrieve_similar_tickets(user_query)

    reply = f"🧠 Detected Issue Type: **{intent}**\n\n"
    reply += "📌 Similar customer queries:\n"

    for ex in examples:
        reply += f"• {ex}\n"

    reply += "\n✅ Support Response:\n"
    reply += KNOWLEDGE.get(intent, "Our support team will contact you soon.")

    reply += "\n\n🤝 Please share more details if needed."

    return reply

# ================================
# MAIN AUTONOMOUS LOOP
# ================================

print("\n🤖 Autonomous Agentic AI Support Bot Started")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye 👋")
        break

    # ============================================
    # ✅ ORDER ID DETECTION ANYTIME (Regex Safe)
    # ============================================

    match = re.search(r"(ORD\d+)", user_input.upper())
    # SEARCH BY NAME / PRODUCT / CITY
    result = search_orders_by_text(user_input)

    if result:
        print("\nBot: 🔍 Matching order found!")
        print(result)
        print()
        continue

    if match:
        order_id = match.group(1)

        print("\nBot: ✅ Order ID detected!")
        print(track_order(order_id, user_input))
        print()
        continue

    # ============================================
    # INTENT DETECTION
    # ============================================

    intent, confidence = detect_intent(user_input)

    print(f"\n[Intent: {intent} | Confidence: {confidence:.2f}]")
    # ============================
    # ============================
# RAG RESPONSE
# ============================
    retrieved = retrieve_similar(user_input)

    print("\n📚 Relevant Past Issues:")
    for i, r in enumerate(retrieved, 1):
        print(f"{i}. {r[:100]}...")

    print("\nBot: Based on similar cases, here's what you can do:")
    print("👉 Please follow standard resolution steps or contact support if issue persists.\n")

    # ============================================
    # ESCALATION
    # ============================================

    if confidence < CONFIG["confidence_threshold"]:
        print("Bot:", CONFIG["escalation_message"])

        ticket_id = escalate_to_human(user_input, intent, confidence)
        print("📌 Ticket Generated:", ticket_id)

        save_feedback(intent, False)
        print()
        continue

    # ============================================
    # DYNAMIC RESPONSE
    # ============================================

    response = generate_dynamic_response(intent, user_input)
    print("\nBot:", response)

    save_feedback(intent, True)
    print()
