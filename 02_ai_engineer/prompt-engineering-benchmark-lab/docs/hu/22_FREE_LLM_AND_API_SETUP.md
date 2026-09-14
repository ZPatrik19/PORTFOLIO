# Ingyenes / alacsony költségű LLM beállítás

## Offline szoftvervalidáció
Használd a `mock` providert a teljes pipeline ellenőrzésére credential és modellköltség nélkül. Ez szimulátor, nem LLM benchmark.

## Valódi lokális modell
Ollama-val valódi lokális LLM futtatható requestenkénti cloud számla nélkül. A költség lokális CPU/GPU/RAM/idő formájában jelentkezik.

## Cloud free tier
Gemini, Groq és OpenRouter aktuális account/region/quota/provider policy alapján adhat free-tier/free-model opciót. Ezek idővel változnak; a repository nem ígér állandó ingyenes quotát.

## Ajánlott workflow
1. Mock smoke test.
2. Ollama vagy cloud free-tier pilot 12–36 reprezentatív mintán.
3. Standard 120-as futás.
4. Nagyobb benchmark csak akkor, ha connection/parsing/quota/cost viselkedés már ismert.

API key soha ne kerüljön source-ba vagy GitHubra szánt screenshotba.
