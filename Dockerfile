FROM python:3.11-slim

WORKDIR /app

# System deps (psycopg2 build + runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Instance dir for SQLite fallback / scratch
RUN mkdir -p instance

ENV PYTHONUNBUFFERED=1

# Railway injects $PORT; default to 5000 locally. Shell form so $PORT expands.
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --timeout 120 app:app
