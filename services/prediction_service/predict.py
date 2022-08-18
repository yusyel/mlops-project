# %%
import os
import json

import mlflow
import requests
import mlflow.sklearn
from flask import Flask, jsonify, request
from prefect import flow, task, get_run_logger

# pylint disable for pymonto (import-error)
from pymongo import MongoClient
from matplotlib import collections

# %%


@task
def load_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_name = "chd_risk_model"
    stage = "Production"

    model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
    return model


app = Flask("chd")


@task
def save_to_db(record, prediction):
    MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")
    mongo_client = MongoClient(MONGODB_ADDRESS)
    db = mongo_client.get_database("prediction_service")
    collections = db.get_collection("data")
    rec = record.copy()
    rec["prediction"] = prediction
    collections.insert_one(rec)


@task
def send_to_evidently_service(record, prediction):
    EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:5000")
    rec = record.copy()
    rec["prediction"] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/chd", json=[rec])


@app.route("/predict", methods=["POST"])
@flow
def predict():
    log = get_run_logger()
    model = load_model()
    record = request.get_json()
    y_pred = model.predict_proba(record)[0, 1]
    risk = y_pred >= 0.5
    result = {
        "Patients CHD risk probability": float(y_pred),
        "Patient CHD risk result": bool(risk),
    }
    save_to_db(record, float(y_pred))
    send_to_evidently_service(record, float(y_pred))
    log.info(f"result is: {result}")
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
