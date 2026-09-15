from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
K8S_DIR = ROOT / "10_deployment" / "kubernetes"


def _documents(path: Path) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]


def test_docker_files_exist_and_do_not_contain_secrets() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "GEMINI_API_KEY=" not in dockerfile
    assert "GEMINI_API_KEY:" not in compose
    assert "COPY pyproject.toml build_backend.py README.md LICENSE ./" in dockerfile
    assert "COPY config.yaml .env.example ./" in dockerfile
    assert "USER appuser" in dockerfile


def test_kubernetes_deployments_have_health_probes_and_resources() -> None:
    deployments = []
    for filename in ("api-deployment.yaml", "ui-deployment.yaml"):
        deployments.extend(
            doc for doc in _documents(K8S_DIR / filename) if doc["kind"] == "Deployment"
        )

    assert len(deployments) == 2
    for deployment in deployments:
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert "readinessProbe" in container
        assert "livenessProbe" in container
        assert "requests" in container["resources"]
        assert "limits" in container["resources"]


def test_secret_manifest_is_example_only() -> None:
    secret_text = (K8S_DIR / "secret.example.yaml").read_text(encoding="utf-8")

    assert "replace_me" in secret_text
    assert "AQ." not in secret_text
