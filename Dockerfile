# FastAPI service. The model is trained at build time from the synthetic
# generator (deterministic, seed 42), so the image is self-contained.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN useradd --create-home --uid 10001 appuser

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY src ./src
RUN python -m src.train --out-dir /app/models \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

USER appuser
ENV MODEL_DIR=/app/models \
    DB_PATH=/app/data/fraud_logs.db \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
