FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Refresh the catalog at build time; falls back to data/catalog_sample.json
# if the upstream host is unreachable during the build.
RUN python scripts/fetch_catalog.py || true

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
