# %%
import os
import json
import boto3
import json
import mlflow
import mlflow.sklearn
# %%
os.environ['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']
os.environ['AWS_DEFAULT_REGION']


def load_model():

    RUN_ID =  '500ac111537445bdb89e5642819f6333'
    logged_model = f"s3://mlops-project-yus/1/{RUN_ID}/artifacts/model"
    model = mlflow.sklearn.load_model(logged_model)
    return model 


def predict(event, model):
    y_pred = model.predict_proba(event)[0, 1]
    return y_pred

def lambda_handler(event, context):
    model = load_model()
    y_pred = predict(event, model)
    risk = y_pred >= 0.5
    result = {
        "Patients CHD risk probability": float(y_pred),
        "Patient CHD risk result": bool(risk),
    }
    return json.dumps(result)