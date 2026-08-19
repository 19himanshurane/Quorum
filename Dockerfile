FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY arbitration/ ./arbitration/
COPY api/ ./api/
COPY ui/ ./ui/

RUN mkdir -p /app/data

EXPOSE 8000 8501
