# %%
import os
import json
import boto3
import json
import mlflow
import mlflow.sklearn
import requests
from pymongo import MongoClient
from matplotlib import collections
# %%
os.environ['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']
os.environ['AWS_DEFAULT_REGION']


def load_model():

    RUN_ID =  '500ac111537445bdb89e5642819f6333'
    logged_model = f"s3://xxxxx/1/{RUN_ID}/artifacts/model"
    model = mlflow.sklearn.load_model(logged_model)
    return model 


def predict(event, model):
    y_pred = model.predict_proba(event)[0, 1]
    return y_pred


def save_to_db(record, prediction):
    MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://xxxxxx.eu-west-1.compute.amazonaws.com:27017")
    mongo_client = MongoClient(MONGODB_ADDRESS)
    db = mongo_client.get_database("prediction_service")
    collections = db.get_collection("data")
    rec = record.copy()
    rec["prediction"] = prediction
    collections.insert_one(rec)

def send_to_evidently_service(record, prediction):
    EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://xxxxxx.eu-west-1.compute.amazonaws.com:8085")
    rec = record.copy()
    rec["prediction"] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/chd", json=[rec])


def lambda_handler(event, context):
    model = load_model()
    y_pred = predict(event, model)
    risk = y_pred >= 0.5
    result = {
        "Patients CHD risk probability": float(y_pred),
        "Patient CHD risk result": bool(risk),
    }
    save_to_db(event, float(y_pred))
    send_to_evidently_service(event, float(y_pred))
    return json.dumps(result)