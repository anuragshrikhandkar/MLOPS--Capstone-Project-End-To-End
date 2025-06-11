import mlflow
from mlflow.tracking import MlflowClient

# Set the MLflow tracking URI
mlflow.set_tracking_uri('https://dagshub.com/anuragshrikhandkar/MLOPS--Capstone-Project-End-To-End.mlflow')

# Initialize client
client = MlflowClient()

# List all registered models
models = client.list_registered_models()

if not models:
    print("❌ No models registered.")
else:
    print("✅ Registered Models:")
    for m in models:
        print(f"- {m.name}")