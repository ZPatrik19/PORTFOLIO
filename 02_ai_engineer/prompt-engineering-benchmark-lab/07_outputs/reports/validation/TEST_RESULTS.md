# Verified Test Results / Ellenőrzött teszteredmények

Validation date / Validáció dátuma: **2026-09-14**

## Full suite / Teljes suite

```text
76 passed, 0 failed
```

## Marker subsets / Marker részhalmazok

```text
unit:        53 passed
integration: 22 passed
smoke:        1 passed
```

## Coverage

```text
Configured core-package coverage: 77%
```

Commands actually executed / Ténylegesen futtatott parancsok:

```bash
python -m pytest -q
python -m pytest -q -m unit
python -m pytest -q -m integration
python -m pytest -q -m smoke
python -m pytest --cov=prompt_benchmark --cov-report=term-missing -q
```

Live cloud-provider calls are not part of the deterministic default test suite because they depend on credentials, quota, model availability, and network access.

Az élő cloud-provider hívások nem részei a determinisztikus alap tesztrendszernek, mert credentialtől, quotától, modell-elérhetőségtől és hálózattól függenek.
