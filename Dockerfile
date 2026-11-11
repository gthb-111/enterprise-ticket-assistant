FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVICEFLOW_HOST=0.0.0.0 \
    SERVICEFLOW_DATA_DIR=/app/.data

WORKDIR /app

RUN useradd --create-home --uid 10001 serviceflow

COPY pyproject.toml README.md ./
COPY serviceflow ./serviceflow
RUN python -m pip install --no-cache-dir .

COPY web ./web
COPY knowledge ./knowledge
COPY samples ./samples
COPY data ./data
COPY docs/architecture.md docs/evaluation-results.json ./docs/
COPY run.py ./

RUN mkdir -p /app/.data && chown -R serviceflow:serviceflow /app
USER serviceflow

EXPOSE 8877

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8877/api/health', timeout=3)"

CMD ["python", "run.py", "serve"]

