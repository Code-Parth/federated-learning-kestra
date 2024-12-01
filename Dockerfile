FROM python:3.9-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/logs /app/results /app/outputs

ENV PYTHONPATH=/app

CMD ["python", "-m", "federated_learning"]