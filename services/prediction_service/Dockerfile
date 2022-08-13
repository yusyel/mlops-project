FROM python:3.9.7-slim

RUN pip install packaging
RUN python3 -m pip install --upgrade pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py", "./" ]
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
#WORKDIR /app

#COPY requirements.txt requirements.txt

#RUN pip3 install -r requirements.txt


#COPY [ "predict.py", "./" ]
#ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]