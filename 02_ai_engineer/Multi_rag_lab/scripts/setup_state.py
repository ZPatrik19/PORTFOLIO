from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".cache" / "setup_state.json"
TRACKED = [
    ROOT / "pyproject.toml",
    ROOT / ".env.example",
    ROOT / "config" / "models.yaml",
    ROOT / "config" / "llm_profiles.yaml",
    ROOT / "config" / "faiss_profiles.yaml",
    ROOT / "config" / "cuda_profiles.yaml",
    ROOT / "SETUP.bat",
    ROOT / "SETUP.sh",
    ROOT / "INFRASTRUCTURE.bat",
    ROOT / "INFRASTRUCTURE.sh",
    ROOT / "EVALUATION.bat",
    ROOT / "EVALUATION.sh",
    ROOT / "scripts" / "infrastructure_cli.py",
]
SETUP_SCHEMA = "2026-10-02-v43-py314-ui-state-arrow"


def fingerprint() -> str:
    digest = hashlib.sha256(SETUP_SCHEMA.encode())
    for path in TRACKED:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_state() -> dict[str, str]:
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "write", "show"])
    args = parser.parse_args()
    fp = fingerprint()
    old = load_state()
    if args.action == "check":
        if old.get("fingerprint") == fp:
            print("UNCHANGED")
            return 0
        print("CHANGED")
        return 10
    if args.action == "write":
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"fingerprint": fp, "schema": SETUP_SCHEMA}, indent=2), encoding="utf-8")
        print(fp)
        return 0
    print(json.dumps({"current": fp, "stored": old}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
