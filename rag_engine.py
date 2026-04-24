import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("data/customer_support_tickets.csv")

# Detect text column automatically
text_col = None
for col in df.columns:
    if "text" in col.lower() or "description" in col.lower():
        text_col = col
        break

if text_col is None:
    raise ValueError("No text column found!")

texts = df[text_col].astype(str).tolist()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(texts, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("✅ RAG Engine Ready!")

# ============================
# SEARCH FUNCTION
# ============================
def retrieve_similar(query, top_k=3):

    query_vec = model.encode([query], convert_to_numpy=True)

    distances, indices = index.search(query_vec, top_k)

    results = []
    for i in indices[0]:
        results.append(texts[i])

    return results