# Artifacts

Az `artifacts/` kizárólag újragenerálható futási outputokat tartalmaz:

- `indexes/`: FAISS/NumPy indexek és metadata;
- `evaluations/`: evaluation datasetek és mérési kimenetek;
- `experiments/`: experiment registry, benchmark és export fájlok.

Ezeket a Git repository nem verziózza. A cél, hogy minden felhasználó ugyanazzal a kóddal és konfigurációval a saját CPU/GPU környezetében reprodukálja a méréseket.
