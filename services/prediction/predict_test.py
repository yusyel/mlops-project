import requests


url = 'http://localhost:9696/predict'


patient =  {"male": "0",
     "age": 94,
     "education": "1.0",
     "currentsmoker": "1",
     "cigsperday": 500.0,
     "bpmeds": "0.0",
     "prevalentstroke": "1",
     "prevalenthyp": "1",
     "diabetes": "1",
     "totchol": 415.0,
     "sysbp": 276.0,
     "diabp": 150.0,
     "bmi": 109.23,
     "heartrate": 100.0,
     "glucose": 105.0}



response = requests.post(url, json=patient).json()
print(response)
