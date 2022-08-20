# syntax=docker/dockerfile:1
#RUN pip install -U pip
#RUN pip install pipenv

#WORKDIR /app

#COPY [ "Pipfile", "Pipfile.lock", "./" ]

#RUN pipenv install --system --deploy --skip-lock
#COPY [ "app.py", "./" ]

#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8085"]

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN pip3 install evidently==0.1.56.dev0

COPY app.py .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8085"]
