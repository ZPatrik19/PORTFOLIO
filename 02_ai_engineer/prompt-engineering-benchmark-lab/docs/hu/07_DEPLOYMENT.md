# Deployment

## Lokális / Windows
A `run_project.bat` létrehozza vagy újrahasználja a `.venv` környezetet, szükség esetén telepíti/frissíti a dependencyket, validálja az environmentet és elindítja az egyetlen Streamlit UI-t. A `RUN_UI.bat` kompatibilitási alias marad.

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
Az image non-root userrel fut, és a Streamlit `/_stcore/health` végpontját használja.

## Kubernetes
A deployment assetek a `06_deployment/kubernetes/` alatt találhatók. A Deployment readiness/liveness probe-ot, resource request/limitet, ConfigMap beállításokat és opcionális Secret referenciát tartalmaz. Valódi secret soha ne kerüljön YAML-ba/Gitbe.

## Production megjegyzés
Ez a repository portfólió/experiment alkalmazás. Többfelhasználós production környezetben a filesystem historyt tranzakciós tárolóra kell cserélni, és authentication, centralizált observability, concurrency/rate-limit kontroll és provider quota management szükséges.
