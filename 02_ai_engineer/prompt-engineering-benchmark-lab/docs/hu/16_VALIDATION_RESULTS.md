# Validációs eredmények

Validáció dátuma: 2026-09-14

## Ebben a környezetben ténylegesen sikeresen futtatva
- `python -m pytest -q` → **76 sikeres, 0 hibás**.
- `python -m pytest --cov=prompt_benchmark --cov-report=term-missing -q` → **77% konfigurált core-package coverage**.
- Marker részhalmazok külön is ellenőrizve: `unit` **53 sikeres**, `integration` **22 sikeres**, `smoke` **1 sikeres**.
- A test marker rendszer támogatja a `unit`, `integration` és `smoke` részhalmazokat.
- A Python package import és típusos benchmark konfiguráció betöltése működik.
- A provider boundaryk offline mockokkal teszteltek, cloud quota fogyasztása nélkül.

| Modulterület | Reprezentatív coverage |
|---|---:|
| promptstratégiák | 94% |
| parsing | 100% |
| mock generátor | 95% |
| path kezelés | 81% |
| provider adapterek | ~77–85% |
| benchmark runner | 74% |
| config | 85% |
| teljes konfigurált core package | **77%** |

## Környezetfüggő ellenőrzések
Az élő cloud-provider request felhasználói credentialt/hálózatot/quotát igényel, ezért nem része a default passing suite-nak. Docker image futtatás Docker daemont, Kubernetes apply kubectl/cluster hozzáférést igényel. Ruff és mypy konfigurálva van, CI/dev környezetben kell futtatni, ahol a binárisok telepítve vannak.

## Integritási nyilatkozat
Mock score nem jelenik meg valódi LLM-minőségként. Élő provider, Docker runtime vagy Kubernetes deployment csak akkor jelölhető sikeresnek, ha ténylegesen végrehajtottuk azt támogató környezetben.
