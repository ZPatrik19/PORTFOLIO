# Ábraértelmezési útmutató – magyar változat

## Macro F1 bar chart
**Kérdés:** melyik prompt adja a legjobb összesített, osztálykiegyensúlyozott minőséget?
**Ne következtesd:** hogy a legmagasabb F1 automatikusan production winner.

## Bootstrap confidence interval
**Kérdés:** mennyire stabil a mért F1 a benchmark mintavételére?
**Ne következtesd:** hogy nem átfedő CI automatikusan minden statisztikai tesztet helyettesít.

## Quality vs Cost
**Kérdés:** mennyi minőséget kapunk adott API-költség mellett?
**Döntés:** kereshető Pareto-hatékony prompt.

## Quality vs P95 Latency
**Kérdés:** mennyi minőséget kapunk a tail latency árán?
**Döntés:** SLA-kompatibilis stratégia választása.

## Token Usage
**Kérdés:** melyik prompt kontextusigénye nagy?
**Döntés:** few-shot/advanced template overhead indokolt-e?

## Per-class F1 heatmap
**Kérdés:** melyik prompt melyik intenten javít vagy romlik?

## Case-type robustness heatmap
**Kérdés:** melyik technika milyen nehézségen segít (ambiguity, injection, long-context, multi-intent)?

## Confusion matrix
**Kérdés:** mely kategóriapárok keverednek?
Az `__invalid__` oszlop a parse-olhatatlan / szerződésszegő kimeneteket mutatja.

## Parameter sweep chart
**Kérdés:** hogyan változik a quality/reliability/latency egy decoding paraméter függvényében?
**Fontos:** itt a promptnak fixnek kell maradnia.

## Ablation plot
**Kérdés:** mely promptkomponensnek van valódi hozzáadott értéke?
