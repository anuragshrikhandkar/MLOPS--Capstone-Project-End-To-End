# from flask import Flask, render_template, request

# # -------------------------------
# # Production MLflow + DagsHub Setup (with token-based auth)
# # -------------------------------
# import os
# import mlflow
# import dagshub

# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "anuragshrikhandkar"
# repo_name = "MLOPS--Capstone-Project-End-To-End"
# mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)



from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# -------------------------------
# Text Preprocessing Functions
# -------------------------------

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    return re.sub(r'\s+', ' ', text).strip()

def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

# -------------------------------
# MLflow + DagsHub Setup

# # -------------------------------
# os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
# os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "anuragshrikhandkar"
repo_name = "MLOPS--Capstone-Project-End-To-End"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)


# mlflow.set_tracking_uri("https://dagshub.com/anuragshrikhandkar/MLOPS--Capstone-Project-End-To-End.mlflow")
# dagshub.init(repo_owner="anuragshrikhandkar", repo_name="MLOPS--Capstone-Project-End-To-End", mlflow=True) 

# mlflow.set_tracking_uri('https://dagshub.com/anuragshrikhandkar/MLOPS--Capstone-Project-End-To-End.mlflow')
# dagshub.init(repo_owner='anuragshrikhandkar', repo_name='MLOPS--Capstone-Project-End-To-End', mlflow=True)

# -------------------------------
# Flask App
# -------------------------------

app = Flask(__name__)

# -------------------------------
# Prometheus Metrics Setup
# -------------------------------

registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Prediction class count", ["prediction"], registry=registry)

# -------------------------------
# Load MLflow Model
# -------------------------------

model_name = "my_model"
model = None

try:
    client = mlflow.MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])

    if not latest_versions:
        print(f"[ERROR] No model versions found for '{model_name}' in MLflow Registry.")
        print("👉 Please register a model via MLflow with the name 'my_model'.")
    else:
        model_version = latest_versions[0].version
        model_uri = f"models:/{model_name}/{model_version}"
        print(f"[INFO] Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    print(f"[EXCEPTION] Failed to load MLflow model '{model_name}': {e}")
    model = None

# -------------------------------
# Load Vectorizer
# -------------------------------

vectorizer = None
try:
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    print("[INFO] Vectorizer loaded successfully.")
except Exception as e:
    print(f"[EXCEPTION] Failed to load vectorizer: {e}")

# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    input_text = request.form["text"]
    print("Input text:", input_text)

    cleaned_text = normalize_text(input_text)
    print("Cleaned text:", cleaned_text)

    try:
        features = vectorizer.transform([cleaned_text])
        print("Feature vector shape:", features.shape)

        features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

        result = model.predict(features_df)
        prediction = result[0]
        print("Model prediction:", prediction)

        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

        if str(prediction) == "1":
            output = "😊 Positive Sentiment"
        else:
            output = "😞 Negative Sentiment"

    except Exception as e:
        output = f"Prediction failed: {e}"

    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
    return render_template("index.html", result=output)

@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# -------------------------------
# Run App
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

# from flask import Flask, render_template, request
# import mlflow
# import pickle
# import os
# import pandas as pd
# from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
# import time
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# import string
# import re
# import dagshub

# import warnings
# warnings.simplefilter("ignore", UserWarning)
# warnings.filterwarnings("ignore")

# def lemmatization(text):
#     """Lemmatize the text."""
#     lemmatizer = WordNetLemmatizer()
#     text = text.split()
#     text = [lemmatizer.lemmatize(word) for word in text]
#     return " ".join(text)

# def remove_stop_words(text):
#     """Remove stop words from the text."""
#     stop_words = set(stopwords.words("english"))
#     text = [word for word in str(text).split() if word not in stop_words]
#     return " ".join(text)

# def removing_numbers(text):
#     """Remove numbers from the text."""
#     text = ''.join([char for char in text if not char.isdigit()])
#     return text

# def lower_case(text):
#     """Convert text to lower case."""
#     text = text.split()
#     text = [word.lower() for word in text]
#     return " ".join(text)

# def removing_punctuations(text):
#     """Remove punctuations from the text."""
#     text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
#     text = text.replace('؛', "")
#     #text = re.sub('\s+', ' ', text).strip()
#     return text

# def removing_urls(text):
#     """Remove URLs from the text."""
#     url_pattern = re.compile(r'https?://\S+|www\.\S+')
#     return url_pattern.sub(r'', text)

# def remove_small_sentences(df):
#     """Remove sentences with less than 3 words."""
#     for i in range(len(df)):
#         if len(df.text.iloc[i].split()) < 3:
#             df.text.iloc[i] = np.nan

# def normalize_text(text):
#     text = lower_case(text)
#     text = remove_stop_words(text)
#     text = removing_numbers(text)
#     text = removing_punctuations(text)
#     text = removing_urls(text)
#     text = lemmatization(text)

#     return text

# # Below code block is for local use
# # -------------------------------------------------------------------------------------
# # mlflow.set_tracking_uri('https://dagshub.com/vikashdas770/YT-Capstone-Project.mlflow')
# # dagshub.init(repo_owner='vikashdas770', repo_name='YT-Capstone-Project', mlflow=True)
# # -------------------------------------------------------------------------------------

# # Below code block is for production use
# # -------------------------------------------------------------------------------------
# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "vikashdas770"
# repo_name = "YT-Capstone-Project"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# # -------------------------------------------------------------------------------------


# # Initialize Flask app
# app = Flask(__name__)

# # from prometheus_client import CollectorRegistry

# # Create a custom registry
# registry = CollectorRegistry()

# # Define your custom metrics using this registry
# REQUEST_COUNT = Counter(
#     "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
# )
# REQUEST_LATENCY = Histogram(
#     "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
# )
# PREDICTION_COUNT = Counter(
#     "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
# )

# # ------------------------------------------------------------------------------------------
# # Model and vectorizer setup
# model_name = "my_model"
# def get_latest_model_version(model_name):
#     client = mlflow.MlflowClient()
#     latest_version = client.get_latest_versions(model_name, stages=["Production"])
#     if not latest_version:
#         latest_version = client.get_latest_versions(model_name, stages=["None"])
#     return latest_version[0].version if latest_version else None

# model_version = get_latest_model_version(model_name)
# model_uri = f'models:/{model_name}/{model_version}'
# print(f"Fetching model from: {model_uri}")
# model = mlflow.pyfunc.load_model(model_uri)
# vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# # Routes
# @app.route("/")
# def home():
#     REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
#     start_time = time.time()
#     response = render_template("index.html", result=None)
#     REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
#     return response

# @app.route("/predict", methods=["POST"])
# def predict():
#     REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
#     start_time = time.time()

#     text = request.form["text"]
#     # Clean text
#     text = normalize_text(text)
#     # Convert to features
#     features = vectorizer.transform([text])
#     features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

#     # Predict
#     result = model.predict(features_df)
#     prediction = result[0]

#     # Increment prediction count metric
#     PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

#     # Measure latency
#     REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

#     return render_template("index.html", result=prediction)

# @app.route("/metrics", methods=["GET"])
# def metrics():
#     """Expose only custom Prometheus metrics."""
#     return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# if __name__ == "__main__":
#     # app.run(debug=True) # for local use
#     app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker