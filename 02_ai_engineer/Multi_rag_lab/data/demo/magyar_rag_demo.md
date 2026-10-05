# Magyar RAG Engineering demó

Ez a rövid, helyben csomagolt dokumentum csak arra szolgál, hogy a felület internetkapcsolat és külső dokumentumletöltés nélkül is azonnal kipróbálható legyen. A valódi magyar mintakorpusz a **Dokumentumok** oldalon egy kattintással tölthető le hivatalos forrásokból.

## Mesterséges intelligencia és digitalizáció

A mesterséges intelligenciára épülő rendszerek minőségét nem csak a generatív modell határozza meg. Egy RAG rendszerben a dokumentumok letöltése, parsingja, tisztítása, chunkolása, embeddingje, visszakeresése, újrarangsorolása és a kontextus összeállítása együttesen határozza meg, milyen evidence jut el a nyelvi modellhez.

## Chunking

A fix méretű chunking egyszerű baseline. A rekurzív chunking természetes szeparátorokat részesít előnyben. A mondat- és bekezdésalapú stratégiák nyelvi határokat tartanak meg. A szemantikus chunking embedding-hasonlóság segítségével próbál témaváltást felismerni. Parent–Child chunking esetén a keresés kis child egységeken történik, miközben a generálás nagyobb parent kontextust kaphat.

## Dense, sparse és hybrid retrieval

Dense retrieval esetén a query és a dokumentumchunkok embeddingjeit hasonlítjuk össze. Normalizált vektorok és inner product használatakor a pontszám cosine similarityként értelmezhető. BM25 lexikális, sparse módszer, amely a szavak előfordulására és ritkaságára támaszkodik. Hybrid retrieval a dense és sparse rangsorokat például Reciprocal Rank Fusion segítségével egyesítheti. Az RRF score rangfúziós pontszám, nem valószínűség.

## Grounded generation

Grounded generálásnál az LLM utasítást kap arra, hogy csak a retrieved evidence alapján válaszoljon, és forrásjelöléseket használjon. Ha az evidence nem elegendő, a rendszernek ezt jeleznie kell ahelyett, hogy nem alátámasztott tényt találna ki.

## Evaluation és performance

Retrieval minőséghez használható Recall@K, Precision@K, Hit Rate, MRR és nDCG. Generálásnál vizsgálható a teljesség, faithfulness, citation accuracy és context utilization. A latency, throughput, CPU/RAM és GPU/VRAM metrikákat külön kell kezelni a minőségi metrikáktól: a gyorsabb pipeline nem automatikusan jobb minőségű.
