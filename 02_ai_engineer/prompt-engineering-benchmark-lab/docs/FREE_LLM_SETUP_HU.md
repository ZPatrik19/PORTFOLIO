# Ingyenes LLM futtatás — lépésről lépésre

Ez a dokumentum azt mutatja meg, hogyan lehet a Prompt Engineering Benchmark Lab projektet úgy lefuttatni, hogy ne legyen kötelező fizetős LLM API-t használni.

A projekt négy ingyenes útvonalat különböztet meg:

```text
1. Mock       → teljesen offline, de NEM valódi LLM
2. Ollama     → valódi LLM lokálisan, API-kulcs nélkül
3. Groq       → valódi cloud LLM, free-tier API-kulccsal
4. Gemini     → valódi cloud LLM, free-tier API-kulccsal
```

A fizetős OpenAI útvonal csak opcionális összehasonlítás.

---

## 1. Melyiket válasszam?

### Ha csak azt akarom ellenőrizni, hogy működik-e a projekt

Használd a `mock` providert.

```bash
python 05_scripts/09_run_smoke_test.py
```

Ez nem LLM. Egy determinisztikus keyword classifier helyettesíti a modellt.

Miért jó?

- nincs internet;
- nincs API-kulcs;
- nincs költség;
- a teljes pipeline ellenőrizhető;
- létrejönnek a CSV-k, metrikák és ábrák.

Miért nem jó végső benchmarknak?

Mert a mock classifier szabályalapú. Az eredménye semmit nem bizonyít egy valódi LLM prompt-engineering képességeiről.

---

## 2. Ollama — a ténylegesen korlátlan, ingyenes megoldás

Az Ollama a modellt a saját gépeden futtatja.

A folyamat:

```text
Python benchmark
      ↓
localhost:11434
      ↓
Ollama
      ↓
local LLM
      ↓
prediction
```

Nincs:

```text
API token számla
cloud quota
API key
```

Van viszont:

```text
lokális számítási idő
RAM/VRAM igény
áramfogyasztás
```

### Telepítés után

Példa kis modell:

```bash
ollama pull llama3.2:3b
```

`.env`:

```text
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TEMPERATURE=0
```

Teszt:

```bash
python 05_scripts/03_check_provider.py --provider ollama
```

Structured Output teszt:

```bash
python 05_scripts/03_check_provider.py --provider ollama --structured
```

Kis benchmark:

```bash
python 05_scripts/04_run_benchmark.py --provider ollama --strategy all --limit 20
```

Teljes benchmark:

```bash
python 05_scripts/04_run_benchmark.py --provider ollama --strategy all
```

Report:

```bash
python 05_scripts/07_generate_report.py --provider ollama
```

### Miért jó portfólióhoz?

Mert valódi LLM inference történik, mégsem kell API-költséget fizetni.

### Mire figyeljek?

A lokális 3B modell gyengébb lehet, mint egy nagy cloud modell. Ez azonban akár érdekes benchmark-következtetés is lehet: megvizsgálható, hogy egy kisebb modell mennyit profitál a precízebb promptingból.

---

## 3. Groq Free Plan

A Groq esetén szükséges API-kulcs, de free-tier quota használható.

`.env`:

```text
LLM_PROVIDER=groq
GROQ_API_KEY=IDE_JÖN_A_KULCS
GROQ_MODEL=qwen/qwen3.8-27b
GROQ_TEMPERATURE=0
```

Egyetlen kéréses ellenőrzés:

```bash
python 05_scripts/03_check_provider.py --provider groq
```

P7 Structured Output ellenőrzés:

```bash
python 05_scripts/03_check_provider.py --provider groq --structured
```

Ezután:

```bash
python 05_scripts/10_run_free_demo.py --provider groq --limit 20
```

### Miért csak 20 minta az alapértelmezett free demo?

Mert:

```text
20 ticket × 8 prompt = 160 LLM request
```

Ha még az ablation is fut:

```text
20 × 5 = 100 további request
```

Összesen:

```text
260 request
```

Ez sokkal biztonságosabb kezdés egy free-tier kvótán belül, mint rögtön 2400+ hívást indítani.

### Hogyan növeljem fokozatosan?

Először:

```bash
--limit 20
```

majd:

```bash
--limit 50
```

majd:

```bash
--limit 100
```

A már elkészült sorokat a runner nem hívja újra.

Ezért a `--limit` nem azt jelenti, hogy minden alkalommal nulláról indulunk.

---

## 4. Gemini API Free Tier

A Gemini esetén Google AI Studio API-kulcs használható.

`.env`:

```text
LLM_PROVIDER=gemini
GEMINI_API_KEY=IDE_JÖN_A_KULCS
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_TEMPERATURE=1.0
GEMINI_THINKING_LEVEL=minimal
```

Teszt:

```bash
python 05_scripts/03_check_provider.py --provider gemini
```

Structured Output:

```bash
python 05_scripts/03_check_provider.py --provider gemini --structured
```

Benchmark:

```bash
python 05_scripts/10_run_free_demo.py --provider gemini --limit 20
```

### Miért `temperature=1.0`?

A Gemini 3 modellekhez a Google jelenlegi dokumentációja az alapértelmezett 1.0 értéket ajánlja. Emiatt a projekt provider-specifikus konfigurációt használ ahelyett, hogy mechanikusan minden modellre ugyanazt a temperature-t kényszerítené.

Ez fontos mérnöki gondolat:

```text
fair comparison within one provider
!=
forcing incompatible settings across different providers
```

A prompt-stratégiák összehasonlításakor ugyanazon Gemini modellen a tisztán prompt-design stratégiák ugyanazt a decoding beállítást kapják. A reasoning/branching kísérletek külön benchmark-családként jelennek meg.

---

## 5. Mock adat vs mock LLM — nem ugyanaz

A projektben két külön fogalom van.

### Mock data

```text
01_data/mock/mock_support_tickets.csv
```

Ez 10 800 szintetikus, címkézett support ticket.

Használható valódi LLM-mel is:

```bash
python 05_scripts/02_prepare_data.py --source mock
python 05_scripts/04_run_benchmark.py --provider groq --strategy all --limit 20
```

Itt:

```text
DATA = mock
MODEL = real Groq LLM
```

### Mock LLM

```bash
--provider mock
```

Itt szabályalapú keyword classifier fut.

Lehet valós Hugging Face adaton is mock providert használni, de az továbbra sem valódi LLM benchmark.

A két tengely tehát független:

```text
               MODEL
          mock       real
DATA
mock      ✓ demo     ✓ LLM demo
real      ✓ pipeline ✓ FINAL BENCHMARK
```

A legjobb GitHub-verzió:

```text
real public dataset
+
real LLM
```

például:

```bash
python 05_scripts/02_prepare_data.py --source huggingface
python 05_scripts/04_run_benchmark.py --provider groq --strategy all
```

vagy teljesen lokálisan:

```bash
python 05_scripts/02_prepare_data.py --source huggingface
python 05_scripts/04_run_benchmark.py --provider ollama --strategy all
```

---

## 6. API kulcs kezelése

Soha ne írd bele a kulcsot Python-fájlba.

Helyes:

```text
.env
```

Példa:

```text
GROQ_API_KEY=gsk_...
```

A `.gitignore` tartalmazza:

```text
.env
```

Ezért a kulcs nem kerül GitHubra.

Commitolható:

```text
.env.example
```

Ez csak azt mutatja meg, milyen változókat kell beállítani.

---

## 7. Mi történik, ha elfogy a free quota?

Minden request után CSV checkpoint készül.

Például:

```text
07_outputs/results/raw/groq/p3_few_shot.csv
```

Ha 87 kérés elkészült és a 88. quota hibát kap, a korábbi 87 eredmény megmarad.

Később ugyanazt a parancsot indítod:

```bash
python 05_scripts/04_run_benchmark.py --provider groq --strategy p3_few_shot
```

A runner az elkészült `sample_id` értékeket kihagyja.

Ez közvetlenül csökkenti a felesleges requesteket és költségkockázatot.

---

## 8. Melyik verziót tenném GitHubra?

Minimum:

```text
mock mode working
+
Ollama support
+
Groq support
+
Gemini support
```

Majd legalább egy valódi lefuttatott benchmark:

```text
Provider: Groq vagy Gemini vagy Ollama
Dataset: public Hugging Face dataset
P0-P16 results
Macro F1
per-class F1
confusion matrices
invalid JSON
latency
tokens
ablation
error analysis
```

Így a reviewer látja, hogy:

1. a pipeline reprodukálható;
2. nincs egyetlen providerhez kötve;
3. tudsz API-kulcsot biztonságosan kezelni;
4. ismered a local inference lehetőségét;
5. tudsz free-tier quota mellett resumable benchmarkot tervezni;
6. a prompt-engineering eredményt valódi metrikákkal bizonyítod.
