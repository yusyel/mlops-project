FROM python:3.9.7-slim

RUN pip install --upgrade pip
RUN pip install pipenv
WORKDIR /app
COPY ["train.py", "framingham.csv", "./" ]
COPY ["Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

CMD ["python3", "-m", "train"]
