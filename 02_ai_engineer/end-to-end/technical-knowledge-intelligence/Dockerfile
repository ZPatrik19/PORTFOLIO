FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl graphviz \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml build_backend.py README.md LICENSE ./
COPY 03_pipeline ./03_pipeline
RUN python -m pip install --upgrade pip \
    && python -m pip install .

COPY config.yaml .env.example ./
COPY 00_setup ./00_setup
COPY 01_data/reference_docs ./01_data/reference_docs
COPY 04_api ./04_api
COPY 05_ui ./05_ui
COPY 07_results ./07_results
COPY 09_docs ./09_docs

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/01_data/user_library /app/01_data/indexes /app/01_data/interim /app/01_data/processed /app/08_logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "04_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
