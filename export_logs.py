import json
import pandas as pd

# -----------------------------
# Export Feedback Logs
# -----------------------------
with open("feedback.json") as f:
    feedback_data = json.load(f)

feedback_df = pd.DataFrame(feedback_data)
feedback_df.to_csv("powerbi_feedback.csv", index=False)

print("✅ Exported: powerbi_feedback.csv")


# -----------------------------
# Export Escalation Tickets
# -----------------------------
with open("escalations.json") as f:
    escalation_data = json.load(f)

escalation_df = pd.DataFrame(escalation_data)
escalation_df.to_csv("powerbi_escalations.csv", index=False)

print("✅ Exported: powerbi_escalations.csv")

