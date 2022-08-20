#%%
import json
import uuid
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import requests
from prefect import flow, task, get_run_logger

#%%


def read(path):
    table = pq.read_table(path)
    data = table.to_pylist()
    return data


#%%
class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


def send(data):
    with open("target.csv", "w", encoding="utf-8") as f_target:
        for row in data:
            row["id"] = str(uuid.uuid4())
            tenyearchd = row["tenyearchd"]
            f_target.write(f"{row['id']},{tenyearchd}\n")
            response = requests.post(
                "http://127.0.0.1:9696/predict",
                headers={"Content-Type": "application/json"},
                data=json.dumps(row, cls=DateTimeEncoder),
            ).json()



def main():
    data = read(path="./output.parquet")
    response = send(data)


main()
