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

# Railway injects $PORT; gunicorn.conf.py reads it in Python (no shell
# expansion needed). Exec form is fine since the config handles the port.
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
