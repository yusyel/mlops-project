import requests
test= {'male': '0',
 'age': 54,
 'education': '1.0',
 'currentsmoker': '0',
 'cigsperday': 0.0,
 'bpmeds': '0.0',
 'prevalentstroke': '1',
 'prevalenthyp': '1',
 'diabetes': '0',
 'totchol': 315.0,
 'sysbp': 176.0,
 'diabp': 87.0,
 'bmi': 29.23,
 'heartrate': 82.0,
 'glucose': 72.0}



url = 'http://localhost:9696/predict'
response = requests.post(url, json=test)
print(response.json())
