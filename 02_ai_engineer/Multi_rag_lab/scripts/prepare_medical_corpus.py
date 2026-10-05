from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_engine.platform.config import load_settings
from rag_engine.ingestion.medical import download_medical_corpus, load_medical_corpus_config
from rag_engine.indexing.embedding_factory import create_embedding_provider
from rag_engine.evaluation.medical_dataset import build_medical_evaluation_dataset
from rag_engine.models import ChunkingConfig
from rag_engine.ingestion.pipeline import ingest_paths
from rag_engine.indexing.vector_factory import create_vector_store

SUPPORTED = {".pdf", ".txt", ".md", ".markdown", ".html", ".htm", ".docx"}


def _configure_console_encoding() -> None:
    """Use UTF-8 for Windows console/pipe output without crashing on symbols."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except OSError, ValueError:
                pass


def _fingerprint(paths: list[Path], config: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(str(path.stat().st_mtime_ns).encode("ascii"))
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if sidecar.exists():
            digest.update(sidecar.read_bytes())
    digest.update(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def _load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return {}


def main() -> int:
    _configure_console_encoding()
    parser = argparse.ArgumentParser(
        description="100 magyar Egészségvonal orvosi cikk automatikus letöltése, darabolása és FAISS indexelése."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "medical_corpus.yaml")
    parser.add_argument(
        "--target", type=int, default=None, help="Letöltendő dokumentumok száma; alapértelmezés a YAML-ból."
    )
    parser.add_argument("--refresh", action="store_true", help="A már meglévő HTML cikkek újraletöltése.")
    parser.add_argument("--force-index", action="store_true", help="Index újraépítése változatlan bemenet esetén is.")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--semantic-threshold", type=float, default=None)
    parser.add_argument("--embedding-device", default=None, choices=["cpu", "cuda", "auto"])
    parser.add_argument("--vector-device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--hashing", action="store_true", help="Offline deterministic hashing embedding használata.")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    corpus = load_medical_corpus_config(args.config)
    target = args.target or corpus.target_documents
    indexing_defaults = corpus.indexing
    strategy = args.strategy or str(
        indexing_defaults.get("strategy", indexing_defaults.get("chunking_strategy", "recursive"))
    )
    chunk_size = args.chunk_size or int(indexing_defaults.get("chunk_size", 700))
    overlap = (
        args.overlap
        if args.overlap is not None
        else int(indexing_defaults.get("overlap", indexing_defaults.get("chunk_overlap", 100)))
    )
    semantic_threshold = (
        args.semantic_threshold
        if args.semantic_threshold is not None
        else float(indexing_defaults.get("semantic_threshold", 0.72))
    )
    embedding_device = (
        args.embedding_device
        or os.getenv("RAG_EMBEDDING_DEVICE")
        or str(indexing_defaults.get("embedding_device", "auto"))
    )
    vector_device = (
        args.vector_device or os.getenv("RAG_VECTOR_DEVICE") or str(indexing_defaults.get("vector_device", "cpu"))
    )
    raw_dir = ROOT / corpus.output_dir
    chunks_path = ROOT / corpus.processed_chunks
    vector_dir = ROOT / corpus.vectorstore_dir
    state_path = ROOT / corpus.state_path
    evaluation_defaults = corpus.evaluation
    eval_target = int(evaluation_defaults.get("target_questions", 80))
    eval_path = ROOT / str(evaluation_defaults.get("dataset_path", "artifacts/evaluations/medical_rag_eval.jsonl"))

    print("=" * 78)
    print("Magyar orvosi RAG korpusz előkészítése")
    print("=" * 78)
    print(f"Forrás       : {corpus.organization}")
    print(f"Cél          : {target} magyar egészségügyi cikk")
    print(f"Nyers adatok : {raw_dir}")
    print(f"Index        : {vector_dir}")
    print()

    def progress(current: int, total: int, title: str) -> None:
        print(f"[LETÖLTÉS {current:03d}/{total:03d}] {title}")

    manifest = download_medical_corpus(
        corpus,
        raw_dir,
        target=target,
        refresh=args.refresh,
        progress_callback=progress,
    )
    print()
    print(
        f"Dokumentumok: {manifest['available_documents']} elérhető · "
        f"{manifest['downloaded_now']} új · {manifest['reused_existing']} újrahasznált"
    )
    if manifest.get("failures"):
        print(f"Figyelmeztetés: {len(manifest['failures'])} letöltés hibás.")
    if int(manifest["available_documents"]) < target:
        raise SystemExit("A kért dokumentumszám nem áll rendelkezésre; indexelés megszakítva.")
    if args.download_only:
        return 0

    print()
    print("[EVAL] Forrásolt magyar orvosi evaluation dataset építése...")
    evaluation_dataset = build_medical_evaluation_dataset(
        raw_dir=raw_dir,
        manifest_path=raw_dir / "manifest.json",
        output_path=eval_path,
        target_questions=eval_target,
    )
    print(
        f"       {len(evaluation_dataset.items)} kérdés · {len({item.source_id for item in evaluation_dataset.items})} forráscikk"
    )
    print(f"       Dataset: {eval_path}")

    paths = [path for path in sorted(raw_dir.iterdir()) if path.is_file() and path.suffix.lower() in SUPPORTED]
    if not paths:
        raise SystemExit("Nincs indexelhető orvosi dokumentum.")

    settings = load_settings()
    configured_model = indexing_defaults.get("embedding_model")
    model_name = (
        "hashing" if args.hashing else (args.model or configured_model or settings.multilingual_embedding_model)
    )
    index_config = {
        "target": target,
        "strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "semantic_threshold": semantic_threshold,
        "embedding_model": model_name,
        "embedding_device": embedding_device,
        "vector_device": vector_device,
    }
    current_fp = _fingerprint(paths, index_config)
    old_state = _load_state(state_path)
    if (
        not args.force_index
        and old_state.get("fingerprint") == current_fp
        and chunks_path.exists()
        and vector_dir.exists()
    ):
        print("[INDEX] A dokumentumok és az indexkonfiguráció nem változott; újraindexelés kihagyva.")
        print(f"        Fingerprint: {current_fp[:16]}...")
        return 0

    print()
    print("[INGEST] Parsing → cleaning → chunking...")
    chunking = ChunkingConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        semantic_threshold=semantic_threshold,
    )
    semantic_embedder = None
    if strategy == "semantic":
        semantic_embedder = create_embedding_provider(
            model_name,
            device=embedding_device,
            fallback_to_hashing=args.hashing,
        )
    result = ingest_paths(paths, chunking, embedder=semantic_embedder)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps([chunk.model_dump() for chunk in result.chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    index_chunks = [chunk for chunk in result.chunks if chunk.metadata.get("role") != "parent"]
    print(f"         {len(paths)} fájl → {len(index_chunks)} indexelendő chunk")

    print("[EMBED] Magyar multilingual embeddingek előállítása...")
    embedder = create_embedding_provider(
        model_name,
        device=embedding_device,
        fallback_to_hashing=args.hashing,
    )
    vectors = embedder.embed_documents([chunk.text for chunk in index_chunks])
    print(f"        Modell: {getattr(embedder, 'model_name', model_name)}")
    print(f"        Device: {getattr(embedder, 'device', args.embedding_device)}")

    print("[INDEX] FAISS / vector backend építése és mentése...")
    store = create_vector_store(device=vector_device, fallback_to_numpy=True)
    store.add(vectors, index_chunks)
    vector_dir.mkdir(parents=True, exist_ok=True)
    store.save(vector_dir)

    state = {
        "fingerprint": current_fp,
        "documents": len(paths),
        "chunks": len(index_chunks),
        "embedding_model": getattr(embedder, "model_name", model_name),
        "embedding_device": getattr(embedder, "device", embedding_device),
        "vector_backend": getattr(store, "backend_name", "unknown"),
        "vector_device": getattr(store, "device", "unknown"),
        "source_files": [
            {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path in paths
        ],
        "config": index_config,
        "chunks_path": str(chunks_path),
        "vectorstore_path": str(vector_dir),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("ORVOSI KORPUSZ KÉSZ")
    print("=" * 78)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    print()
    print("Figyelem: a korpusz oktatási/RAG-demó célú; nem helyettesít orvosi vizsgálatot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
