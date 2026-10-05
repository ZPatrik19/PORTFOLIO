# FAISS GPU

## Miért nem működik általában natív Windows Pythonból?

A `faiss-cpu` csomag nem tartalmazza a FAISS GPU API-kat (`StandardGpuResources`, `index_cpu_to_gpu`). Ezért attól még, hogy PyTorch/Cross-Encoder CUDA-n fut, a FAISS maradhat CPU-n.

## Javasolt környezet

```text
Windows
  ↓
WSL2 / Ubuntu
  ↓
NVIDIA CUDA passthrough
  ↓
Conda environment
  ↓
faiss-gpu
```

## Telepítés

```bash
conda env create -f environment-gpu.yml
conda activate multi-rag-faiss-gpu
pip install -e . --no-deps
python scripts/check_faiss_gpu.py
```

Sikeres ellenőrzésnél a scriptnek GPU API-t és legalább egy látható GPU-t kell jelentenie.

## Fallback

A `cuda_preferred` profil CPU/NumPy fallbackot enged. Benchmark vagy szigorú validáció esetén a `cuda_strict` profilt használd, hogy a rendszer ne mérjen véletlenül CPU fallbackot GPU eredményként.
