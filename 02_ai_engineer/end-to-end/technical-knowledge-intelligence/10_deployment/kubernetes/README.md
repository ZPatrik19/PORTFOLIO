# Kubernetes deployment

These manifests are a reference deployment for the two real application adapters: FastAPI and Streamlit. They intentionally do not add Kafka/Redis/microservices.

## Before deploy

1. Build and push `technical-knowledge-intelligence:latest` to your registry; update image names if required.
2. Provision storage compatible with the PVC access mode, or adapt `pvc.yaml` to your cluster.
3. Create the Gemini secret without committing it:

```bash
kubectl apply -f namespace.yaml
kubectl -n tki create secret generic tki-secrets --from-literal=GEMINI_API_KEY='<key>'
```

## Apply

```bash
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f api-deployment.yaml
kubectl apply -f ui-deployment.yaml
kubectl -n tki get pods
kubectl -n tki get services
```

## Logs

```bash
kubectl -n tki logs deployment/tki-api
kubectl -n tki logs deployment/tki-ui
```

## Remove

```bash
kubectl delete -f ui-deployment.yaml
kubectl delete -f api-deployment.yaml
kubectl delete -f pvc.yaml
kubectl delete -f configmap.yaml
```

The example secret file contains no credential and is never intended to be applied unchanged in production.
