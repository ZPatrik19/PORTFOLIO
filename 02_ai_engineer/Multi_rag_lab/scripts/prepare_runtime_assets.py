from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from rag_engine.platform.model_assets import (
    HF_HUB_CACHE,
    STATE_PATH,
    configure_hf_environment,
    runtime_model_ids,
)
from rag_engine.platform.config import load_settings


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _download_one(repo_id: str, *, retries: int, verify: bool) -> Path:
    # Hugging Face reads most environment variables at import time.
    configure_hf_environment()
    from huggingface_hub import HfApi, snapshot_download

    token = os.getenv("HF_TOKEN") or None
    revision = os.getenv("HF_MODEL_REVISION") or HfApi(token=token).model_info(repo_id).sha
    if not revision:
        raise RuntimeError(f"Nem sikerült feloldani a Hugging Face modell revisionjét: {repo_id}")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[HF] {repo_id} · próbálkozás {attempt}/{retries}")
            snapshot = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    cache_dir=str(HF_HUB_CACHE),
                    token=token,
                    max_workers=1,
                    etag_timeout=float(os.getenv("HF_HUB_ETAG_TIMEOUT", "30")),
                )
            )
            # This second call is network-free and verifies that the cached snapshot is complete.
            verified = Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    cache_dir=str(HF_HUB_CACHE),
                    local_files_only=True,
                    max_workers=1,
                )
            )
            if verify:
                if not (verified / "config.json").exists() and not (verified / "modules.json").exists():
                    raise RuntimeError(f"A letöltött modell snapshot nem tűnik teljesnek: {verified}")
            print(f"[OK] {repo_id} → {verified}")
            return verified
        except Exception as exc:  # network/backend dependent
            last_error = exc
            print(f"[WARN] {repo_id} letöltési hiba: {exc}")
            if attempt < retries:
                delay = min(2 ** (attempt - 1) * 3, 30)
                print(f"       Újrapróbálás {delay} másodperc múlva; a már letöltött byte-ok cache-ben maradnak.")
                time.sleep(delay)
    raise RuntimeError(f"A modell nem tölthető le {retries} próbálkozás után: {repo_id}: {last_error}")


def _verify_model_loading(items: dict[str, dict[str, str]]) -> None:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    for role, item in items.items():
        path = item["absolute_snapshot_path"]
        print(f"[VERIFY] {role}: {item['repo_id']}")
        if role == "reranker":
            model = CrossEncoder(path, device="cpu")
        else:
            model = SentenceTransformer(path, device="cpu")
            _ = model.get_sentence_embedding_dimension()
        del model
        print("         betöltés cache-ből: OK")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A RAG runtime Hugging Face modelljeinek robusztus előtöltése és ellenőrzése."
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--verify", action="store_true", help="A letöltés után CPU-n ténylegesen töltsd is be a modelleket."
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Csak a már cache-elt modelleket ellenőrizd, hálózat nélkül."
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    configure_hf_environment()
    HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    model_ids = runtime_model_ids(settings)
    if not os.getenv("HF_TOKEN"):
        print(
            "[INFO] HF_TOKEN nincs beállítva. A modellek publikusak, ezért a letöltés működik, de kisebb Hub rate limit mellett."
        )
        print("       Opcionális: add meg a HF_TOKEN értékét a .env fájlban.")

    items: dict[str, dict[str, str]] = {}
    if args.verify_only:
        if not STATE_PATH.exists():
            raise SystemExit("Nincs runtime model cache state. Futtasd előbb ezt a scriptet --verify kapcsolóval.")
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        for role, item in payload.get("models", {}).items():
            raw = Path(str(item["snapshot_path"]))
            absolute = raw if raw.is_absolute() else ROOT / raw
            if not absolute.exists():
                raise SystemExit(f"Hiányzó cache snapshot: {absolute}")
            items[str(role)] = {
                "repo_id": str(item["repo_id"]),
                "snapshot_path": str(item["snapshot_path"]),
                "absolute_snapshot_path": str(absolute),
            }
    else:
        for role, repo_id in model_ids.items():
            snapshot = _download_one(repo_id, retries=max(1, args.retries), verify=True)
            items[role] = {
                "repo_id": repo_id,
                "snapshot_path": _relative_or_absolute(snapshot),
                "absolute_snapshot_path": str(snapshot.resolve()),
            }

        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "hf_home": _relative_or_absolute(HF_HUB_CACHE.parent),
                    "authenticated": bool(os.getenv("HF_TOKEN")),
                    "models": {
                        role: {k: v for k, v in item.items() if k != "absolute_snapshot_path"}
                        for role, item in items.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[OK] Modell cache state: {STATE_PATH}")

    if args.verify or args.verify_only:
        _verify_model_loading(items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
