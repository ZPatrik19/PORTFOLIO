from __future__ import annotations

import sys


def main() -> int:
    try:
        import faiss
    except Exception as exc:
        print(f"[FAIL] import faiss: {exc}")
        return 1

    print(f"FAISS version: {getattr(faiss, '__version__', 'unknown')}")
    print(f"FAISS module:  {getattr(faiss, '__file__', 'unknown')}")

    has_gpu_api = hasattr(faiss, "StandardGpuResources")
    print(f"GPU API:       {has_gpu_api}")
    if not has_gpu_api:
        print("[FAIL] CPU-only FAISS build is imported. StandardGpuResources is missing.")
        print("       Use WSL2/Linux and install faiss-gpu; remove faiss-cpu from that environment.")
        return 2

    gpu_count = int(faiss.get_num_gpus()) if hasattr(faiss, "get_num_gpus") else -1
    print(f"Visible GPUs:  {gpu_count}")
    if gpu_count <= 0:
        print("[FAIL] FAISS GPU build is installed, but no CUDA GPU is visible.")
        return 3

    dimension = 384
    resources = faiss.StandardGpuResources()
    cpu_index = faiss.IndexFlatIP(dimension)
    gpu_index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
    print(f"GPU index:     {type(gpu_index).__name__}")
    print("[OK] FAISS GPU is operational.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
