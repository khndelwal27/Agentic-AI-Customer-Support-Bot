import json
import pickle
import datetime

# ---------------- LOAD KNOWLEDGE + CONFIG ---------------- #

with open("knowledge.json") as f:
    KNOWLEDGE = json.load(f)

with open("config.json") as f:
    CONFIG = json.load(f)

FEEDBACK_FILE = "feedback.json"

# ---------------- LOAD TRAINED MODEL ---------------- #

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- MEMORY (MODEL-BASED AGENT) ---------------- #

memory = {
    "last_intent": None
}

# ---------------- ML INTENT DETECTION (REACTIVE AGENT) ---------------- #

def detect_intent(user_input):
    X = vectorizer.transform([user_input])
    intent = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])
    return intent, confidence

# ---------------- UTILITY DECISION AGENT ---------------- #

def utility_decision(confidence):
    if confidence >= CONFIG["confidence_threshold"]:
        return "auto"
    return "escalate"

# ---------------- RESPONSE TOOL AGENT ---------------- #

def get_response(intent):
    return KNOWLEDGE.get(intent, "Sorry, I cannot handle this query right now.")

# ---------------- LEARNING AGENT ---------------- #

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

# ---------------- MAIN LOOP ---------------- #

print("\n🤖 Agentic AI Customer Support Bot (Final-Year Version)")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Thank you! Goodbye 😊")
        break

    # Step 1: Predict intent using ML model
    intent, confidence = detect_intent(user_input)

    # Step 2: Update memory
    memory["last_intent"] = intent

    # Step 3: Decision making
    decision = utility_decision(confidence)

    print(f"\n[Intent: {intent} | Confidence: {confidence:.2f}]")

    # Step 4: Action
    if decision == "auto":
        response = get_response(intent)
        print("Bot:", response)
        save_feedback(intent, True)
    else:
        print("Bot:", CONFIG["escalation_message"])
        save_feedback(intent, False)

    print()
