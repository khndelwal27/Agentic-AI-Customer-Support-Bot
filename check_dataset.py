import pandas as pd
df = pd.read_csv("data/customer_support_tickets.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nSample Rows:\n")
print(df.head())
