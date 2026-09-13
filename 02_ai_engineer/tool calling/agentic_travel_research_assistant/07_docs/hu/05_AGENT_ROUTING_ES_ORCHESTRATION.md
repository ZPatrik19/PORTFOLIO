# Agent routing és orchestration

Ez a dokumentum bemutatja, hogyan dönti el a rendszer, mely toolokra van szükség, és hogyan lesz a routing döntésből végrehajtható workflow.

## 1. Miért több routing módszer?

A projekt célja az összehasonlíthatóság. Ugyanazt a tool-réteget több döntési módszer használja, ezért külön mérhető a routing minősége és az execution minősége.

## 2. `rule_based`

Egyszerű lexical/regex baseline. Gyors és determinisztikus, de gyenge generalizációjú. Referenciapontként hasznos.

## 3. `plan_execute`

Előbb explicit tervet készít, majd végrehajtja a lépéseket. A magyar morfológia és argumentum-parsing jelentős része a `heuristics.py`-ban van. Előnye az inspectability; hátránya a kézzel írt szabályok karbantartása.

## 4. `ml_router`

A tool-intent kiválasztást multi-label klasszifikátor végzi; az argumentumok determinisztikus parserből jönnek. High-precision guardrailok felülírhatnak bizonyos classifier false positive-okat, például explicit negáció esetén.

## 5. `openai_direct`

Az LLM kapja meg a tool schema-kat, majd function call objektumokat generál. A Python alkalmazás validálja és végrehajtja a hívást, majd a tool outputot visszaadja a modellnek. A loop addig folytatódik, amíg a modell végső szöveges választ nem ad vagy eléri a max step limitet.

## 6. Kompozit workflow-k

Nem minden capability érdemes tisztán classifier labelként kezelni. A költségterv például lehet determinisztikus dependency workflow: hotel/food/transport eredmények → `calculate`. Ez csökkenti a label-space túlterhelését és explicit dependency-t ad.

## 7. Negáció és guardrail

Az olyan kérések, mint „ne keress hotelt” vagy „a szállás már megvan”, negatív routing signalok. Ezeknél a rendszer explicit guardrailt használ, mert egy classifier valószínűsége önmagában nem elég erős szerződés a side-effect vagy fölösleges tool-call elkerülésére.

## Összegzés

A több routing stratégia ugyanarra a tool-rétegre épül. Ez teszi értelmezhetővé a benchmarkot: a különbség valóban a döntési logikából, nem eltérő tool-implementációból ered.
