#%%
import json
import uuid
from datetime import datetime
from time import sleep
import pyarrow.parquet as pq
import requests
from datetime import timedelta

#%%

table = pq.read_table("output.parquet")
data = table.to_pylist()


#%%
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


with open("target.csv", 'w') as f_target:
    for row in data:
        row['id'] = str(uuid.uuid4())
        tenyearchd = (row['tenyearchd'])
        f_target.write(f"{row['id']},{tenyearchd}\n")
        response = requests.post("http://127.0.0.1:9696/predict",
                            headers={"Content-Type": "application/json"},
                            data=json.dumps(row, cls=DateTimeEncoder)).json()
        print(f"prediction: {response}")
        sleep(1)