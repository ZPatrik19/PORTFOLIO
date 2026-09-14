# Kubernetes deployment

This deployment runs the Streamlit application directly. Streamlit exposes
`/_stcore/health`, which is used for readiness and liveness probes; a separate
API service would add complexity without improving this project's runtime model.

```bash
docker build -t prompt-engineering-benchmark-lab:latest .
kubectl apply -f 06_deployment/kubernetes/namespace.yaml
kubectl apply -f 06_deployment/kubernetes/configmap.yaml
# Create a real secret separately if a cloud provider is required.
kubectl apply -f 06_deployment/kubernetes/deployment.yaml
kubectl apply -f 06_deployment/kubernetes/service.yaml
kubectl -n prompt-benchmark get pods,svc
kubectl -n prompt-benchmark logs deployment/prompt-benchmark
kubectl -n prompt-benchmark port-forward svc/prompt-benchmark 8501:8501
```

Open `http://localhost:8501` after port-forwarding.

For credentials, create a Secret outside Git (for example with
`kubectl create secret generic ... --from-literal=...`).
Do not apply `secret.example.yaml` with real values committed to source control.

Remove resources:

```bash
kubectl delete namespace prompt-benchmark
```
