# Deployment

## Local UI

```bat
run_project.bat
```

The daily runner starts Streamlit only. The UI calls the shared `tkip` core directly, so running FastAPI at the same time is unnecessary.

## Local API

```bat
run_project.bat api
```

Use this when another client needs HTTP endpoints.

## Docker

```bash
docker build -t technical-knowledge-intelligence .
docker compose up --build
```

Docker Compose intentionally runs separate UI/API services. Personal documents and `.env` are not baked into the image; local data is mounted at runtime.

## Kubernetes

See `10_deployment/kubernetes/README.md` for separate UI/API Deployments and Services, shared data storage, probes, resource limits, ConfigMap and secret example.
