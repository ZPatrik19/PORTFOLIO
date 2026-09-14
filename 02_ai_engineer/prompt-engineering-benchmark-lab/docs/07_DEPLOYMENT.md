# Deployment

## Local Windows

```text
run_project.bat
```

`RUN_UI.bat` remains a backward-compatible alias.

## Local Linux/macOS

```bash
chmod +x run_project.sh
./run_project.sh
```

## Docker

```bash
docker build -t prompt-engineering-benchmark-lab:latest .
docker run --rm -p 8501:8501 prompt-engineering-benchmark-lab:latest
```

For cloud providers pass secrets at runtime, never in the image:

```bash
docker run --rm -p 8501:8501 \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  prompt-engineering-benchmark-lab:latest
```

Health endpoint:

```text
GET /_stcore/health
```

## Kubernetes
See `06_deployment/kubernetes/README.md`. The deployment uses the same health endpoint for readiness and liveness probes and has CPU/memory requests and limits.

## Persistence
The sample Kubernetes manifest uses `emptyDir` for results. Production multi-user use should mount a PVC/object store/database depending on retention and concurrency requirements.
