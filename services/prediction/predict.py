#%%
import json
from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
import requests
#%%
logged_model = '../../mlruns/1/d4d213067afe450a8878a63e70ea0fa5/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)


#%%
app = Flask('chd')
@app.route('/predict', methods=['POST'])


def predict():
    x = request.get_json()

    pred = model.predict(x)
    sonuc = pred >= 0.5
    result = {
        'Patients CHD risk probability': float(pred),
        'Patient CHD risk result': bool(sonuc)
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    # %%
