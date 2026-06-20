# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Default: run the heuristic eval
CMD ["python", "run_eval.py"]
