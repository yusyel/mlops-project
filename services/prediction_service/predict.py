# %%
import json
from flask import Flask, request, jsonify
from matplotlib import collections
import mlflow
import mlflow.sklearn
import requests
from pymongo import MongoClient
import os
# %%



best_model = "8c805ce1a2ac4e3198ab41e420396ce7"
logged_model = f"./1/{best_model}/artifacts/model"
model = mlflow.sklearn.load_model(logged_model)

EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTLY_SERVICE', 'http://127.0.0.1:5000')
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")



# %%
app = Flask('chd')
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collections = db.get_collection('data')


@app.route('/predict', methods=['POST'])
def predict():
    record = request.get_json()

    y_pred = model.predict_proba(record)[0, 1]
    risk = y_pred >= 0.5
    result = {
        'Patients CHD risk probability': float(y_pred),
        'Patient CHD risk result': bool(risk)
    }
    save_to_db(record, float(y_pred))
    send_to_evidently_service(record, float(y_pred))
    return jsonify(result)


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collections.insert_one(rec)

def send_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/chd", json=[rec])


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
# %%
