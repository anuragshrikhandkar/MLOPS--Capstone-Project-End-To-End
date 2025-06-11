# File: train_model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import mlflow
import mlflow.sklearn
import os
import pickle

# ------------------------------
# Step 1: Prepare Dataset
# ------------------------------
data = pd.DataFrame({
    'text': [
        "I love this product!",
        "This is amazing.",
        "Absolutely fantastic experience!",
        "I am very happy with the results.",
        "I hate it.",
        "Worst experience ever.",
        "It was really bad.",
        "Terrible service."
    ],
    'label': [1, 1, 1, 1, 0, 0, 0, 0]
})

# ------------------------------
# Step 2: Vectorizer + Model
# ------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

model = LogisticRegression()
model.fit(X, y)

# ------------------------------
# Step 3: Save Vectorizer
# ------------------------------
os.makedirs("models", exist_ok=True)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("[INFO] Vectorizer saved to models/vectorizer.pkl")

# ------------------------------
# Step 4: Log to MLflow
# ------------------------------
mlflow.set_tracking_uri("https://dagshub.com/anuragshrikhandkar/MLOPS--Capstone-Project-End-To-End.mlflow")
mlflow.set_experiment("SentimentAnalysis")

with mlflow.start_run():
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="my_model")
    print("[INFO] Model logged and registered as 'my_model'")
