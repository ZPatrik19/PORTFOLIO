# Changelog

## 0.43.0

- Javítva a Streamlit `StreamlitWidgetAlreadyInstantiatedError`: az Infrastructure oldal többé nem módosít közvetlenül már létrehozott sidebar widget-state kulcsokat.
- Új deferred widget-state mechanizmus: a profilváltások a következő rerun elején, még a widgetek létrehozása előtt lépnek életbe.
- Bevezetve az egységes Arrow-safe dataframe réteg minden `st.dataframe` és `st.data_editor` híváshoz; a vegyes `object` oszlopok típushelyesen nullable string/numeric/bool oszlopokká normalizálódnak.
- Javítva az Infrastructure `Érték` oszlop `int`/`str` keveredése és ugyanennek a hibacsaládnak a többi UI táblában való lehetősége.
- Új regressziós tesztek a shared widget-state mutáció, az Arrow-safe táblák és az Infrastructure profilgomb AppTest útvonal ellen.

## 0.42.0

- Javítva a Streamlit/PyArrow rangtábla: a `Dense rank` és `BM25 rank` most nullable `Int64`, így nincs `ArrowInvalid` vegyes `int`/`—` típus miatt.
- A Python 3.14-es CUDA bootstrap determinisztikusan `torch==2.14.1+cu126` buildet telepít `--force-reinstall` módban, ezért a korábban telepített CPU-only PyTorch nem maradhat bent észrevétlenül.
- `SETUP.bat` / `SETUP.sh` NVIDIA driver észlelésekor automatikusan aktiválja és validálja a CUDA-s PyTorch runtime-ot.
- A setup fingerprint már az infrastructure installer változásait is követi.
- Új regressziós tesztek a CUDA wheel cserére és az Arrow-kompatibilis nullable rank oszlopokra.

## 0.41.0

- Javítva a Windows FAISS index mentés/betöltés ékezetes vagy más Unicode karaktert tartalmazó projektútvonalakon.
- A FAISS persistence most elsődlegesen `serialize_index` / `deserialize_index` útvonalat használ, így a C++ `FileIOWriter` nem kap Unicode fájlútvonalat.
- A Windows launcherek többé nem írják felül a rendszer `TEMP` / `TMP` változóit projektlokális `.cache\temp` útvonallal; ez csökkenti a natív könyvtárak Unicode-path hibáit.
- A GPU FAISS load ugyanazt a Unicode-safe persistence réteget használja.
- Új regressziós teszt ellenőrzi a FAISS persistence helper Unicode-safe viselkedését.

## 0.40.0

- A projekt runtime-ja egységesen CPython 3.14-re állt át: `pyproject.toml`, CI, Docker és launcher contractok.
- Visszaállt a korábban működő, egyszerű négy-launcheres Windows flow: `SETUP.bat`, `INFRASTRUCTURE.bat`, `RUN.bat`, `EVALUATION.bat`.
- A `SETUP` újra csak Python/dependency/test felelősséget kezel; a CUDA/FAISS/Ollama gépspecifikus bootstrap külön infrastructure lépés.
- A `RUN` Python 3.14-et validál, setup-state változásnál frissít, majd infrastructure runtime check után indít Streamlitet.
- Új `EVALUATION.bat` és `EVALUATION.sh` került be a retrieval/RAG/measurement munkafolyamatokhoz.
- A V39 központi `bootstrap_setup.py` / `run_app.py` launcher absztrakció megszűnt, mert a bizonyított BAT-alapú flow egyszerűbb és Windows alatt átláthatóbb.

## 0.39.0

- A Windows `SETUP.bat` minimalis bootstrap launcherva egyszerusodott; a teljes setup logika a tesztelheto `scripts/bootstrap_setup.py` modulba kerult.
- A setup hiba eseten mindig megall es a `logs/setup/` naplora mutat.
- Hianyzo Python eseten `winget` segitsegevel Python 3.12 telepitest probal Windows alatt.
- A Windows `RUN.bat` es `INFRASTRUCTURE.bat` is vekonyabb, robusztusabb wrapper lett.
- A Linux/WSL `SETUP.sh` ugyanazt a Python bootstrapot hasznalja, igy a ket platform setup folyamata nem tud szetcsuszni.
- Uj `scripts/run_app.py` centralizalja a setup-state, runtime preflight es Streamlit inditasi logikat.

## 0.38.0

- Javítva az Ollama/FAISS/CUDA profilok hiányzó UI metaadatai (`display_name`, `description`, FAISS `output_dir`).
- A profile loader biztonságos default metaadatokat ad, így régi/hiányos YAML profil nem dönti el a Streamlit UI-t.
- Visszakerült a gyökérszintű `INFRASTRUCTURE.bat`, és készült `INFRASTRUCTURE.sh` párja.
- Az infrastructure manager képes CUDA-s PyTorch build automatikus javítására NVIDIA driver esetén, valamint Ollama telepítésére/indítására és Qwen profil létrehozására.
- A `SETUP` most a core Python telepítés után gépspecifikus infrastructure bootstrapot is futtat.
- A setup fingerprint figyeli a CUDA/FAISS/LLM profilokat és a launchereket is.

## 0.37.0

### Setup / runtime
- A Windows és Linux/WSL `SETUP` launcherek valódi Python-verzióellenőrzést, sérült `.venv` újraépítést, dependency-installt, FAISS CPU telepítést, konfigurációs könyvtár-előkészítést és compile/import smoke checket végeznek.
- Az Ollama opcionálisan a hivatalos telepítővel telepíthető; meglévő Ollama esetén a `balanced` Qwen profil automatikusan pull/create ellenőrzést kap.
- A `RUN` launcherek hiányzó `.venv` vagy megváltozott dependency/config fingerprint esetén automatikusan újrafuttatják a setupot.
- Új `runtime_preflight.py` ellenőrzi a demo adatot, FAISS/NumPy fallbackot, PyTorch CUDA állapotot és a konfigurált LLM runtime-ot.
- Megszűnt a már nem létező `infrastructure/cuda/CUDA_SETUP.bat` legacy hivatkozás; a CUDA setup most diagnosztikai, host-specifikus policyt követ.
- A setup fingerprint a `pyproject.toml` mellett az alap environment- és modellprofil-konfigurációkat is követi.

## 0.36.0

### Added
- Measurement Lab: concurrency/load benchmark, query robustness és paired bootstrap A/B összehasonlítás.
- Bootstrap confidence interval és latency coefficient-of-variation mérés.
- R-Precision, MRR@K és deterministic Context Precision@K retrieval metrikák.
- Retrieval/generation failure-diagnostics taxonomy.
- Offline `scripts/run_measurement_suite.py` és külön CI measurement-smoke job.

### Changed
- Retrieval/RAG benchmark sorok uncertainty és failure-rate mezőkkel bővültek.
- Experiment Registry schema v5 támogatja az új mérési mezőket.
- README és evaluation/CI dokumentáció a reprodukálható mérési protokollal bővült.


## 0.35.0

### Changed
- A CI az architektúra rétegeihez igazított külön quality, unit, integration, UI, package és Docker jobokra váltott.
- A benchmarkok, FAISS indexek, letöltött raw corpus és processed outputok már nem verziózott release-adatok; mindenki saját környezetben generálja őket.
- A README rövidebb, mérnöki fókuszú projektbemutató lett: megvalósított komponensek, tesztstratégia, CI és CPU/CUDA mérés.

### Added
- Repository contract checker a struktúra és separation-of-concerns regressziók ellen.
- Manuális self-hosted CUDA/FAISS GPU GitHub Actions workflow.
- Dependabot konfiguráció pip és GitHub Actions függőségekhez.
- `docs/ci.md`, `data/README.md` és `artifacts/README.md`.

## 0.34.0

### Architecture
- megszűnt a `src/rag_lab` csomagszint;
- új közvetlen `rag_engine` backend package;
- 21 apró backend terület helyett 6 nagy domain: ingestion, indexing, retrieval, generation, evaluation, platform;
- query transformation modulok összevonva;
- RAG strategy modulok összevonva;
- embedding és vector store közös indexing domainbe került;
- runtime/infrastructure/observability közös platform domainbe került;
- UI feature almappák `components/` alá lettek lapítva.

### Repository
- `data/` és `artifacts/` felelősség szétválasztva;
- új `config/` layout;
- `pyproject.toml` lett a Python konfiguráció központja;
- új `.env.example`, `.gitignore`, `.dockerignore`;
- új Dockerfile és compose;
- Windows/Linux SETUP/RUN launcherek;
- verziózott hotfix dokumentumok helyett tartós témadokumentáció.

### Existing functionality preserved
- 8 chunking stratégia;
- 12 RAG stratégia;
- hybrid/BM25/dense retrieval;
- Cross-Encoder reranking;
- grounded generation + fallback;
- evaluation/performance lab;
- FAISS CPU/GPU code path.
