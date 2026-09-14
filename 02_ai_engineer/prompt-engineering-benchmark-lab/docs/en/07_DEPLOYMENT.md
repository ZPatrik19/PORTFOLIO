# Deployment

## Local / Windows
`run_project.bat` creates or reuses `.venv`, installs/updates project dependencies as needed, validates the environment, and starts the single Streamlit UI. `RUN_UI.bat` remains a compatibility alias.

## Linux/macOS
```bash
chmod +x run_project.sh run_tests.sh
./run_project.sh
```

## Docker
```bash
docker build -t prompt-benchmark .
docker run --rm -p 8501:8501 --env-file .env prompt-benchmark
```
The image runs as a non-root user and uses Streamlit's `/_stcore/health` endpoint.

## Kubernetes
Deployment assets live in `06_deployment/kubernetes/`. The Deployment defines readiness/liveness probes, resource requests/limits, ConfigMap settings, and optional secrets from a Secret object. Never commit real secrets into YAML.

## Production note
This repository is a portfolio/experiment application. For multi-user production use, move filesystem history to a transactional store, add authentication, centralized observability, concurrency/rate-limit controls, and provider-specific quota management.
