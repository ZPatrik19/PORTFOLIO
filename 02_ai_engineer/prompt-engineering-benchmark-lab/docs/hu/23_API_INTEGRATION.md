# API integráció

## Egységes boundary
Minden provider ugyanazt a kliens contractot implementálja és normalizált telemetriát ad vissza. A benchmarknak nem kell ismernie a provider-specifikus SDK response formátumot.

## Request lifecycle
```text
PromptPayload -> provider adapter -> timeout/retry -> provider response -> normalized LLMResponse -> parser/validáció -> benchmark sor
```

## Hibaosztályok
Authentication/configuration error fail-fast. Timeout, bizonyos rate-limit és transient server hiba korlátozott backoff mellett retry-olható. Malformed model output benchmark eredmény/hibaállapotként kerül rögzítésre, nem tűnik el.

## Token és latency forrás
Valódi provider esetén ahol lehet provider-reported tokenhasználat és wall-clock request latency szerepel. Mock módban a becsült/szimulált forrás explicit címkét kap.

## Connection test
Az UI és provider-check scriptek kis requestet futtatnak benchmark előtt. Itt célszerű model-name, credential, quota és endpoint hibát észlelni.
