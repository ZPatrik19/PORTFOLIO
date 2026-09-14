FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PROMPT_BENCHMARK_ROOT=/app

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --create-home app

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY 03_src ./03_src

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install . --no-deps

COPY 00_setup ./00_setup
COPY 01_data ./01_data
COPY 02_notebooks ./02_notebooks
COPY configs ./configs
COPY 05_scripts ./05_scripts
COPY 06_deployment/docker-entrypoint.sh ./06_deployment/docker-entrypoint.sh
COPY docs ./docs
COPY 07_outputs/reports ./07_outputs/reports

# Normalize Windows CRLF line endings and make the entrypoint executable.
RUN sed -i 's/\r$//' /app/06_deployment/docker-entrypoint.sh \
    && chmod 755 /app/06_deployment/docker-entrypoint.sh

RUN mkdir -p \
      /app/07_outputs/results \
      /app/07_outputs/logs \
      /app/configs/prompts/custom \
      /app/configs/prompts/playground \
    && chown -R app:app /app

USER app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()" || exit 1

ENTRYPOINT ["/app/06_deployment/docker-entrypoint.sh"]