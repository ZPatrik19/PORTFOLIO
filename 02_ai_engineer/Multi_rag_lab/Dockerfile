FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    HF_HUB_ETAG_TIMEOUT=30 \
    HF_XET_NUM_CONCURRENT_RANGE_GETS=4

WORKDIR /app

# Pull current Debian security fixes before installing runtime packages.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Install the application, then explicitly raise security-sensitive transitive
# packages to versions that include the fixes enforced by the Trivy gate.
RUN python -m pip install --no-cache-dir --upgrade pip "setuptools>=78.1.1" wheel \
    && python -m pip install --no-cache-dir ".[cpu]" \
    && python -m pip install --no-cache-dir --upgrade \
       "msgpack>=1.2.1" \
       "urllib3>=2.8.0" \
       "setuptools>=78.1.1" \
    && python -m pip check

# Default image build is self-contained for the RAG retrieval path: public HF
# embedding/reranker models, medical corpus and the persistent FAISS index are
# prepared once at build time. Set --build-arg PREBUILD_RAG_ASSETS=0 for an
# offline/lightweight image build.
ARG PREBUILD_RAG_ASSETS=1
RUN if [ "$PREBUILD_RAG_ASSETS" = "1" ]; then \
      python scripts/prepare_runtime_assets.py --retries 5 --verify && \
      python scripts/prepare_medical_corpus.py; \
    fi

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
