FROM python:3.9.7-slim

RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py", "./" ]
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
