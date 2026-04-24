import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("\n🚀 Training Model Using Kaggle Support Ticket Dataset...\n")

# Step 1: Load Kaggle dataset
data = pd.read_csv("data/customer_support_tickets.csv")
print("Dataset Loaded:", data.shape)
print("\nColumns in dataset:\n")
print(data.columns)

# Step 2: Clean dataset
data = data.dropna()
data = data.drop_duplicates()

print("Dataset Loaded:", data.shape)

# Step 3: Select correct columns
# Step 3: Automatically detect correct text + label columns

print("\nAvailable Columns:\n")
print(data.columns)

# Detect text column
possible_text_cols = [
    "text",
    "ticket_text",
    "description",
    "issue",
    "query",
    "Ticket Description",
    "ticket_description"
]

text_col = None
for col in possible_text_cols:
    if col in data.columns:
        text_col = col
        break

if text_col is None:
    raise ValueError("❌ No valid text column found in dataset!")

print("\n✅ Using TEXT column:", text_col)

# Detect label/intent column
possible_label_cols = [
    "intent",
    "category",
    "Category",
    "issue_type",
    "Ticket Type"
]

label_col = None
for col in possible_label_cols:
    if col in data.columns:
        label_col = col
        break

if label_col is None:
    raise ValueError("❌ No valid label/intent column found in dataset!")

print("✅ Using LABEL column:", label_col)

# Final dataset assignment
X = data[text_col]
y = data[label_col]


# Step 4: Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2)
)

X_vec = vectorizer.fit_transform(X)

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Step 6: Train classifier
model = LogisticRegression(max_iter=2000, class_weight="balanced")

model.fit(X_train, y_train)

# Step 7: Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model Accuracy:", accuracy)
print("\n📌 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Step 8: Save model + vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\n🎉 Model trained and saved successfully!")
print("Saved: model.pkl and vectorizer.pkl\n")
