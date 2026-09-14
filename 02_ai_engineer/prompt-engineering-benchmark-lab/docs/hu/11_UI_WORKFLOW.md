# UI workflow

## Ajánlott sorrend
1. **Provider / API** — válassz Mock/Ollama/cloud providert, modellt és szükség esetén runtime API kulcsot; teszteld a kapcsolatot.
2. **Dataset** — nézd meg a forrást, méretet, class balance-ot, scenario eloszlást és difficulty profilt.
3. **Playground** — fejlessz promptot klasszifikációval, szabad szöveggenerálással vagy A/B összehasonlítással; ments presetet.
4. **Benchmark** — válassz futási profilt és stratégiákat; figyeld élőben a progresst, tokeneket, latencyt és részleges leaderboardot.
5. **Output Validation** — vizsgáld a confusion matrixot, invalid outputokat, JSON/contract validityt és hibákat a kiválasztott runhoz.
6. **Dashboard** — hasonlítsd össze quality, uncertainty, token, latency, cost, difficulty és scenario heatmap eredményeket.
7. **History / Comparison** — tölts vissza korábbi futásokat és hasonlíts model/provider/prompt kísérleteket.

## Állapotmegőrzés
A futás közbeni UI state Streamlit session state-ben marad; a befejezett benchmarkok lemezre is mentődnek run ID-val, manifesttel, dataset snapshottal, raw predikciókkal és summary metrikákkal.

## Playground A/B
A Classification A/B két valódi promptstratégiát hasonlít ugyanazon a ticketen, opcionális ismert ground truth-tal. A Generative A/B szerkeszthető template-eket hasonlít és releváns esetben output hossz, token, latency, keyword/format adherence és JSON-validity metrikákat mutat.
