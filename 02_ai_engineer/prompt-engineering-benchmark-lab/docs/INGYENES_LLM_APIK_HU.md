# Ingyenes / ingyenesen kipróbálható LLM API-k

Utolsó ellenőrzés: 2026-09-13.

> A free tier limitek és modellek változhatnak. Benchmark előtt mindig ellenőrizd a szolgáltató aktuális dokumentációját.

## 1. Ollama — teljesen lokális, API-költség nélkül

- API key: nem kell.
- Költség: nincs token alapú API-díj.
- Előny: reprodukálható, privát, offline is fut.
- Hátrány: a saját CPU/GPU erőforrásodat használja.
- Projektben: teljesen integrálva.

Ajánlott első próbához: `llama3.2:3b` vagy más, a gépeden kényelmesen futó instruct modell.

Forrás: https://ollama.com/

## 2. Groq Free Plan

- API key: kell.
- Költség: a Free Plan kvótáin belül 0 USD.
- 2026-09-13 körüli dokumentált példa `openai/gpt-oss-20b` modellre: 30 RPM, 1000 RPD, 8K TPM, 200K TPD.
- Előny: nagyon gyors inference.
- Projektben: teljesen integrálva.

Forrás: https://console.groq.com/docs/rate-limits

## 3. OpenRouter Free Models Router

- API key: kell.
- Alapértelmezett modell/router: `openrouter/free`.
- Tokenár: a Free Models Router oldalán 0 USD input/output.
- Előny: több ingyenes modell közül automatikusan választ; feature igények alapján is tud route-olni.
- Hátrány: a háttérben kiválasztott modell változhat, ezért szigorú tudományos összehasonlításhoz inkább fix modellt válassz.
- Projektben: integrálva.

Források:
- https://openrouter.ai/openrouter/free/
- https://openrouter.ai/collections/free-models

## 4. Google Gemini Free Tier

Portfólió-benchmarkhoz a projekt alapértelmezett ingyenes Gemini modellje:

`gemini-3.5-flash-lite`

A Google aktuális pricing oldala szerint Standard Free Tierben az input és output is díjmentes a free-tier limiteken belül.

- API key: kell, Google AI Studio-ból.
- Előny: strukturált output, multimodális és modern Gemini API.
- Projektben: teljesen integrálva.

Forrás: https://ai.google.dev/gemini-api/docs/pricing

## 5. Hugging Face Inference Providers

- Ingyenes felhasználóknak jelenleg havi kis összegű free credit jár (a dokumentáció 0.10 USD-t ír, változhat).
- Sok provider/model egységes API-n keresztül elérhető.
- A projektben jelenleg ajánlásként szerepel, nincs külön adaptere.

Forrás: https://huggingface.co/docs/inference-providers/en/pricing

## Mit ajánlok ehhez a projekthez?

1. **Mock** — csak UI/pipeline demonstrációhoz.
2. **Ollama** — valódi modell, teljesen ingyen, ha a géped bírja.
3. **Groq Free Plan** — a legegyszerűbb gyors cloud benchmark.
4. **Gemini 3.5 Flash-Lite Free Tier** — jó második cloud összehasonlítás.
5. **OpenRouter Free** — sok modell gyors kipróbálásához.

A GitHub README végső eredményeihez legalább egy valódi modell/provider futást használj, ne csak Mock eredményt.
