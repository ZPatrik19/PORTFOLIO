# API integráció – Mock / Ollama / Groq / Gemini / OpenRouter / OpenAI

## Egységes interface

Minden provider ugyanazt a `BaseLLMClient.classify(PromptPayload)` interface-t használja.

Ez azért fontos, mert így ugyanazt a:

- datasetet;
- P0–P16 promptot;
- parser/validator logikát;
- metric pipeline-t;
- checkpoint/cache rendszert

lehet használni provider-váltáskor is.

## Provider módok

### Mock

API key nélkül működik. Csak szimuláció.

### Ollama

Lokális valódi LLM. API key nem kell.

Példa:

```text
ollama pull llama3.2:3b
python 05_scripts/13_api_integration_example.py --provider ollama
```

### Groq

`.env`:

```text
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
```

A projekt kezeli:

- `temperature`;
- `top_p`;
- reasoning effort;
- JSON Schema Structured Output;
- token usage;
- request latency.

### Gemini

`.env`:

```text
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_THINKING_LEVEL=minimal
```

A projekt Interactions API adaptert használ.

Kezelt beállítások:

- temperature;
- top_p;
- top_k;
- thinking level / reasoning mode;
- Structured Output;
- token usage;
- latency.


### OpenRouter

`.env`:

```text
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openrouter/free
```

Az `openrouter/free` router jelenleg ingyenes modellek közül választ. Gyors explorációhoz jó, de szigorú modell-vs-modell benchmarkhoz célszerű fix OpenRouter modellazonosítót választani, hogy a háttérmodell ne változzon.

A projekt az OpenRouter OpenAI-kompatibilis `chat/completions` API-ját használja.

### OpenAI

`.env`:

```text
OPENAI_API_KEY=...
OPENAI_MODEL=<aktuálisan elérhető Responses API model ID>
```

Az OpenAI modellazonosító nincs hardcode-olva, mert az aktuális API modellek idővel változhatnak.

## Live API teszt a UI-ban

A `🔌 API Integration` tabon:

1. válassz providert a sidebarban;
2. add meg az API keyt, ha szükséges;
3. válaszd ki a prompttechnikát;
4. adj meg egy ticketet;
5. `Élő request küldése`.

A UI megmutatja:

- prediction;
- raw output;
- input token;
- output token;
- total token;
- latency;
- estimated cost;
- token source;
- latency source;
- output-validity;
- JSON-validity;
- temperature/top_p/top_k;
- branch count.

## CLI single-request példa

```text
python 05_scripts/13_api_integration_example.py \
  --provider groq \
  --strategy p16_full_advanced_template \
  --ticket "Please cancel before the next renewal"
```

## Full benchmark

```text
python 05_scripts/04_run_benchmark.py --provider groq --strategy all
python 05_scripts/07_generate_report.py --provider groq
```

A Dashboard ezután ugyanazokat az ábrákat mutatja a valódi modellel, mint a mock szimulációnál.

## Biztonságos Gemini-konfiguráció az egységes UI-ban

A Google AI Studio kulcs session-szintű beállításához:

```text
RUN_UI.bat → sidebar: Gemini → GEMINI_API_KEY → Kapcsolat tesztelése
```

A UI jelszómezőjében megadott kulcs csak az aktuális Streamlit process memóriájába kerül; a projekt, a logok és a benchmark CSV-k nem tárolják. Ha később kifejezetten tartós helyi konfigurációt szeretnél, a Git által ignorált `.env` továbbra is támogatott.

Kapcsolat-ellenőrzés SDK-tól független REST health checkkel:

```bash
python 05_scripts/15_test_gemini_connection.py
```

Structured-output health check:

```bash
python 05_scripts/15_test_gemini_connection.py --structured
```
