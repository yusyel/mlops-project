# %%
import json
from flask import Flask, request, jsonify
from matplotlib import collections
import mlflow
import mlflow.sklearn
import requests
from pymongo import MongoClient
import os
from prefect import flow, task, get_run_logger
# %%





# %%

def load_model():

    best_model = "285dda4dcdca4cccbac6e8a1f4959e33"
    logged_model = f"./1/{best_model}/artifacts/model"
    model = mlflow.sklearn.load_model(logged_model)
    return model



def evidently():
    EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
    return EVIDENTLY_SERVICE_ADDRESS



def mongo():
    MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")
    return MONGODB_ADDRESS


app = Flask('chd')
MONGODB_ADDRESS = mongo()
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collections = db.get_collection('data')



@app.route('/predict', methods=['POST'])
@flow
def predict():
    model = load_model()
    record = request.get_json()
    logger = get_run_logger()
    y_pred = model.predict_proba(record)[0, 1]
    risk = y_pred >= 0.5
    result = {
        'Patients CHD risk probability': float(y_pred),
        'Patient CHD risk result': bool(risk)
    }
    save_to_db(record, float(y_pred))
    send_to_evidently_service(record, float(y_pred))
    logger.info(f"Result: {result}")
    return jsonify(result)

@task
def save_to_db(record, prediction):
    MONGODB_ADDRESS = mongo()
    rec = record.copy()
    rec['prediction'] = prediction
    collections.insert_one(rec)


@task
def send_to_evidently_service(record, prediction):
    EVIDENTLY_SERVICE_ADDRESS = evidently()
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/chd", json=[rec])




if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
# %%
