# Prompt Engineering Benchmark Lab – magyar összefoglaló

Ez a projekt egy reprodukálható Prompt / AI Engineering benchmark rendszer. Nem promptgyűjtemény: kontrollált P0–P16 kísérleteket, saját promptokat, A/B Playgroundot, valós és mock LLM providereket, output-validációt, token/latency/költség mérést, statisztikai bizonytalanságot, futási historyt és interaktív Plotly dashboardot tartalmaz.

## Gyors indítás

Windows:

```bat
run_project.bat
```

Linux/macOS:

```bash
./run_project.sh
```

Tesztek:

```bash
pytest -v
pytest --cov=prompt_benchmark --cov-report=term-missing
```

## Ajánlott workflow

1. Provider / API beállítás és connection test.
2. Dataset ellenőrzés.
3. Playground: P0–P16 vagy saját prompt kipróbálása, prompt mentése.
4. Benchmark: stratifikált smoke/pilot/standard/strong/full futás.
5. Output Validation: invalid output, confusion matrix, class-wise metrikák.
6. Dashboard: quality, confidence interval, token, latency és cost trade-off.
7. History: régi futások visszatöltése és összehasonlítása.

## Dataset

Az offline challenge generator 10 800 egyedi ticketet állít elő. A final holdout 6 000 mintás, 6 osztállyal és 18 scenario family-vel. A development és few-shot adat külön marad a final holdouttól.

## Projektstruktúra

```text
00_setup/       környezet és Windows bootstrap
01_data/        nyers/mock/processed/fine-tuning adatok
02_notebooks/   10 oktató notebook
03_src/         telepíthető prompt_benchmark Python package
04_tests/       pytest tesztek
05_scripts/     pipeline, benchmark, UI és utility entrypointok
06_deployment/  Kubernetes deployment
07_outputs/     eredmények, reportok, logok
configs/        benchmark/provider/pricing/prompt konfigurációk
docs/           architektúra, tesztelés, deployment, döntések
```

## Docker

```bash
docker build -t prompt-engineering-benchmark-lab .
docker run --rm -p 8501:8501 prompt-engineering-benchmark-lab
```

## Fontos

A mock eredmények szimulációk; valós portfolio benchmarkhoz ugyanazt a protokollt futtasd Ollama/Gemini/Groq/OpenRouter/OpenAI providerrel. Kis pilot 1.0 Accuracy/Macro F1 értéke önmagában nem bizonyít tökéletes modellt; a UI mintaszámot és confidence intervalt is mutat.

Részletes dokumentáció: `docs/`.


## Dokumentáció

A teljes dokumentáció párhuzamosan elérhető magyarul és angolul:

- 🇭🇺 [Magyar dokumentációs index](docs/hu/00_DOCUMENTATION_INDEX.md)
- 🇬🇧 [English documentation index](docs/en/00_DOCUMENTATION_INDEX.md)
- [Teljes magyar tesztkatalógus](docs/hu/13_TEST_CATALOG.md)
- [Complete English test catalog](docs/en/13_TEST_CATALOG.md)

A két dokumentációs fa ugyanazokat a témákat fedi le: architektúra, adatpipeline, prompt engineering, evaluation, minden teszt célja, deployment, troubleshooting, design decisionök, API providerek, UI workflow, security/reproducibility, project story, refactor report, validációs eredmények, teljes projektstruktúra és notebook guide.
