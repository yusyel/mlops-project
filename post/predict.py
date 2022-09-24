# %%
import os
import json
import mlflow
import requests
import mlflow.sklearn
from flask import Flask, jsonify, request

# %%
os.environ['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']
os.environ['AWS_DEFAULT_REGION']

RUN_ID =  '500ac111537445bdb89e5642819f6333'


logged_model = f"s3://mlops-project-yus/1/{RUN_ID}/artifacts/model"
model = mlflow.sklearn.load_model(logged_model)
app = Flask("chd")

@app.route("/predict", methods=["POST"])
def predict():
    record = request.get_json()
    y_pred = model.predict_proba(record)[0, 1]
    risk = y_pred >= 0.5
    result = {
        "Patients CHD risk probability": float(y_pred),
        "Patient CHD risk result": bool(risk),
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
