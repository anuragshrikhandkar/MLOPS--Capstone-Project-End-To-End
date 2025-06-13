import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

# Step 1: Load the cleaned training data
df = pd.read_csv("data/raw/train.csv")

# Step 2: Extract text column (adjust column name if needed)
texts =  df["review"]   # Replace "text" if your column has a different name

# Step 3: Create and fit the CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(texts)

# Step 4: Save vectorizer to models/vectorizer.pkl
os.makedirs("models", exist_ok=True)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Vectorizer created and saved at models/vectorizer.pkl")