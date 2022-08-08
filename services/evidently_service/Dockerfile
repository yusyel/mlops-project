# syntax=docker/dockerfile:1

FROM python:3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy
COPY [ "app.py", "./" ]

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8085"]

