# Tesztelési stratégia

## Cél
A tesztrendszer nem csak a Python-szintaxist védi, hanem az experiment helyességét. Ebben a projektben a legveszélyesebb hibák a csendes data leakage, stale cache újrahasználat, rossz provider payload, malformed output parsing, félrevezető statisztika, secret/config hiba és olyan UI-regresszió, amely félreértelmezhető benchmarkhoz vezet.

## Rétegek
- `unit`: izolált, determinisztikus logika.
- `integration`: több projektkomponens együtt, továbbra is fizetős/élő külső hívás nélkül.
- `smoke`: a legkisebb teljes offline benchmark útvonal.
- `external`: fenntartott marker opt-in élő provider tesztekhez, amelyek credentialt/hálózatot igényelnek.

## Parancsok
```bash
pytest
pytest -m unit
pytest -m integration
pytest -m smoke
pytest --cov=prompt_benchmark --cov-report=term-missing
```

Windows:
```text
run_tests.bat
```
Linux/macOS:
```bash
./run_tests.sh
```

## Jelenlegi igazolt eredmény
2026-09-14-i validáció:
- 76 teszt sikeres, 0 hibás.
- konfigurált core-package coverage: 77%.
- a modulonkénti pontos coverage a `16_VALIDATION_RESULTS.md` dokumentumban található.

## Mit mockolunk?
A provider SDK/network boundaryk unit/integration tesztekben mockoltak. Így quota fogyasztása nélkül ellenőrizhető a request construction, normalized response handling, credential guard, retry-classification és schema contract.

## Mit nem állítunk hamisan?
A default determinisztikus suite nem számít sikeresnek egy élő Gemini/Groq/OpenRouter/OpenAI hívást, mert az felhasználói credentialtől, quotától, modell-elérhetőségtől és hálózattól függ.

## Teljes katalógus
Minden tesztfájl és minden tesztfüggvény engineering célja: [13_TEST_CATALOG.md](13_TEST_CATALOG.md).
