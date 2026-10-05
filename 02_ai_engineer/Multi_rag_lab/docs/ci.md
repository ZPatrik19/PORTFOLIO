# CI és quality gates

A CI felépítése követi a repository architektúráját: a kódminőség, komponenslogika, integráció, UI, packaging és container build külön hibaterületként jelenik meg.

## Default CI

`.github/workflows/ci.yml`

| Job | Ellenőrzés | Külső szolgáltatás/GPU |
|---|---|---|
| `quality` | repository contract, compileall, Ruff | nem |
| `unit-tests` | unit suite Python 3.14 alatt + coverage | nem |
| `integration-tests` | RAG integration + performance smoke | nem |
| `ui-smoke` | Streamlit AppTest + UI contractok | nem |
| `package-build` | wheel/sdist build + Twine check | nem |
| `docker-build` | CPU Docker image build | nem |
| `ci-success` | az összes kötelező job eredményének gate-je | nem |

A default CI szándékosan nem indít Ollama modellt és nem hív fizetős API-t. Ez gyorsabb, reprodukálhatóbb és biztonságosabb PR gate-et ad.

## Repository contract

`scripts/check_repository.py` megakadályozza, hogy a refaktor után visszakerüljenek tipikus repository hibák:

- régi `src/` / `rag_lab` struktúra;
- `__pycache__`, `.egg-info`, `.venv` vagy build cache;
- verziózott FAISS index, benchmark output vagy letöltött corpus;
- hard-coded gépspecifikus Windows útvonal;
- Streamlit/UI függőség a `rag_engine` core package-ben.

Ez architektúra-regresszió elleni automatizált guardrail.

## GPU workflow

`.github/workflows/gpu-self-hosted.yml`

A workflow csak `workflow_dispatch` eseményre indul, és `[self-hosted, linux, x64, gpu]` címkéjű runnerre vár. Célja:

1. `nvidia-smi` runtime ellenőrzés;
2. PyTorch CUDA ellenőrzés;
3. FAISS GPU API ellenőrzés;
4. `@pytest.mark.gpu` tesztek;
5. CPU/CUDA embedding benchmark.

A GPU workflow azért különül el a default CI-től, mert a CUDA sebesség és a FAISS GPU elérhetősége hardver- és driverfüggő.

## Dependency maintenance

`.github/dependabot.yml` hetente ellenőrzi:

- a Python/pip függőségeket;
- a GitHub Actions verziókat.

## Branch protection javaslat

GitHubon a `CI gate` jobot érdemes required status checkként beállítani a `main` branchre. Így merge csak akkor történhet, ha a teljes CPU CI sikeres.


## Measurement suite smoke

A CI külön, hálózat- és GPUfüggetlen `measurement-smoke` jobot futtat. A verziózott demo dokumentumot hashing embeddinggel és CPU/NumPy vector search fallbackkal méri, majd JSON riportot tölt fel artifactként. Ezzel a bootstrap CI, load benchmark és query-robosztussági mérési útvonal minden PR-ban futásképes marad.
