from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_engine.evaluation.device_benchmark import benchmark_embedding, benchmark_metadata
from rag_engine.platform.config import load_settings
from rag_engine.indexing.hashing import HashingEmbeddingProvider
from rag_engine.indexing.sentence_transformer import SentenceTransformerEmbeddingProvider
from rag_engine.platform.device import cuda_available


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=int, default=500)
    parser.add_argument("--hashing", action="store_true", help="Use dependency-light hashing baseline")
    args = parser.parse_args()
    texts = [f"RAG benchmark document chunk {i} about retrieval, embeddings and grounded generation." for i in range(args.workload)]
    settings = load_settings()
    devices = ["cpu"] if args.hashing else ["cpu"] + (["cuda"] if cuda_available() else [])
    records = []
    for device in devices:
        if args.hashing:
            factory = lambda _device: HashingEmbeddingProvider()
        else:
            factory = lambda d: SentenceTransformerEmbeddingProvider(settings.embedding_model, device=d)
        result = benchmark_embedding(factory, texts, device)
        records.append({**result.to_dict(), "metadata": benchmark_metadata(device)})
    if len(records) == 2 and {record["device"] for record in records} == {"cpu", "cuda"}:
        records[1]["speedup_vs_cpu"] = records[0]["total_ms"] / max(records[1]["total_ms"], 1e-9)
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
