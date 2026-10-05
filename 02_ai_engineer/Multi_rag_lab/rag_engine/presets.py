from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPreset:
    label: str
    query: str
    topic: str


HUNGARIAN_QUERY_PRESETS = [
    QueryPreset(
        "Magas vérnyomás",
        "Melyek a magasvérnyomás-betegség fő kockázatai, és mikor szükséges orvosi kivizsgálás?",
        "Kardiológia",
    ),
    QueryPreset(
        "2-es típusú cukorbetegség",
        "Melyek a 2-es típusú cukorbetegség leggyakoribb tünetei és főbb kockázatai?",
        "Anyagcsere",
    ),
    QueryPreset(
        "Asztma",
        "Melyek az asztma jellemző tünetei, és milyen helyzetben szükséges sürgős segítséget kérni?",
        "Pulmonológia",
    ),
    QueryPreset(
        "Anafilaxia",
        "Milyen tünetek utalhatnak anafilaxiára, és mi a sürgős teendő ilyen esetben?",
        "Allergológia",
    ),
    QueryPreset(
        "Agyrázkódás",
        "Milyen tünetek jelentkezhetnek agyrázkódás után, és mikor indokolt sürgős orvosi ellátás?",
        "Neurológia",
    ),
    QueryPreset(
        "Depresszió",
        "Milyen jellemző tünetei lehetnek a depressziónak, és mikor érdemes szakemberhez fordulni?",
        "Mentális egészség",
    ),
    QueryPreset(
        "Csontritkulás",
        "Melyek a csontritkulás fő kockázati tényezői, és milyen megelőzési lehetőségek vannak?",
        "Mozgásszervi egészség",
    ),
    QueryPreset(
        "Zöldhályog",
        "Mi a zöldhályog, milyen tünetei lehetnek, és miért fontos a korai felismerése?",
        "Szemészet",
    ),
]



CHUNKING_STRATEGIES: dict[str, dict[str, str]] = {
    "fixed": {
        "name": "Fix karakterhossz",
        "summary": "A szöveget közel azonos karakterszámú blokkokra vágja.",
        "how": "A splitter balról jobbra halad, a chunk_size határnál levág, majd a chunk_overlap mennyiséget visszalépi.",
        "best_for": "Gyors baseline, homogén szöveg és reprodukálható benchmark.",
        "tradeoff": "Egyszerű és gyors, de mondatot vagy fogalmi egységet is kettévághat.",
    },
    "fixed-token": {
        "name": "Fix tokenszám",
        "summary": "A dokumentumot közel azonos számú tokenből álló egységekre bontja.",
        "how": "A tokenhatár jobban közelíti az LLM kontextus-költségét, mint a puszta karakterszám.",
        "best_for": "Kontextusablak- és költségtervezés, tokenbudget érzékeny pipeline.",
        "tradeoff": "A szemantikai határokat ugyanúgy figyelmen kívül hagyhatja.",
    },
    "recursive": {
        "name": "Rekurzív",
        "summary": "Természetes határokat keres: szakasz → bekezdés → mondat → szó.",
        "how": "Először nagyobb szeparátorok mentén próbál vágni, és csak szükség esetén lép kisebb egységre.",
        "best_for": "Általános RAG baseline dokumentációkhoz és hosszabb PDF-szövegekhez.",
        "tradeoff": "Jobb szövegkohézió, de a dokumentum struktúráját nem érti teljesen.",
    },
    "sentence": {
        "name": "Mondatalapú",
        "summary": "Teljes mondatokból épít chunkokat egy célméret eléréséig.",
        "how": "Mondathatárokat detektál, majd egymást követő mondatokat csoportosít.",
        "best_for": "Narratív, magyarázó és szabályozási szövegek.",
        "tradeoff": "Megőrzi a mondatokat, de hosszú mondatok vagy táblázatos szöveg problémás lehet.",
    },
    "paragraph": {
        "name": "Bekezdésalapú",
        "summary": "A bekezdéseket tekinti elsődleges természetes egységnek.",
        "how": "A bekezdéshatárokat megtartja, és szükség esetén több rövid bekezdést egyesít.",
        "best_for": "Jól szerkesztett jelentések, kézikönyvek, stratégiák.",
        "tradeoff": "A PDF parser rossz sortörései ronthatják a bekezdéshatárokat.",
    },
    "semantic": {
        "name": "Szemantikus",
        "summary": "Embedding-hasonlóság alapján keresi a témaváltásokat.",
        "how": "Mondatembeddingeket számol, majd a szomszédos mondatok közötti cosine similarity eséseinél határt képez.",
        "best_for": "Hosszú, több témát keverő dokumentumok és témaváltások megtartása.",
        "tradeoff": "Általában drágább és lassabb; az embedding modell és a threshold erősen befolyásolja az eredményt.",
    },
    "structure-aware": {
        "name": "Struktúraérzékeny",
        "summary": "A Markdown/HTML címsorokat és dokumentumszerkezetet is figyelembe veszi.",
        "how": "H1/H2/H3 szekciókat természetes kontextushatárként kezel, és a section metadata-t továbbviszi.",
        "best_for": "Dokumentáció, HTML, Markdown és jól felismerhető fejezetstruktúra.",
        "tradeoff": "PDF esetén csak akkor erős, ha a parsing képes visszaállítani a strukturális jeleket.",
    },
    "parent-child": {
        "name": "Parent–Child",
        "summary": "Kis child chunkokra keres, de nagyobb parent egységet ad az LLM kontextusához.",
        "how": "A child embedding pontos retrievalt ad; találat után parent_id alapján visszakeresi a bővebb kontextust.",
        "best_for": "Amikor a pontos találat és a generáláshoz szükséges tágabb összefüggés egyszerre fontos.",
        "tradeoff": "Több metadata és mapping szükséges, a kontextus nagyobb lehet.",
    },
}


RAG_STRATEGIES: dict[str, dict[str, str]] = {
    "baseline": {
        "name": "Baseline RAG",
        "flow": "Lekérdezés → beágyazás → FAISS → Top-K → kontextus → LLM",
        "summary": "A legegyszerűbb dense RAG. Jó kontrollcsoport a többi stratégia méréséhez.",
        "when": "Gyors baseline, szemantikus keresésre alkalmas korpusz.",
    },
    "hybrid": {
        "name": "Hybrid RAG",
        "flow": "Lekérdezés → Dense + BM25 → rangfúzió → Top-K → kontextus → LLM",
        "summary": "A szemantikus és lexikális találatokat kombinálja.",
        "when": "Tulajdonnevek, jogi kifejezések, rövidítések és szemantikus kérdések együtt.",
    },
    "lexical": {
        "name": "Lexical / BM25 RAG",
        "flow": "Lekérdezés → BM25 → Top-K → kontextus → LLM",
        "summary": "Tiszta lexikális kontrollstratégia, embedding retrieval nélkül.",
        "when": "Pontos terminusok, ritka kifejezések, rövidítések és baseline összehasonlítás.",
    },
    "reranked": {
        "name": "Reranked RAG",
        "flow": "Hibrid visszakeresés → több jelölt → újrarangsorolás → Top-K → kontextus → LLM",
        "summary": "Nagyobb candidate setből egy második modell rendezi újra a találatokat.",
        "when": "Ha retrieval recall fontos, de csak kevés, erős evidence fér a kontextusba.",
    },
    "dense-reranked": {
        "name": "Dense + Reranked RAG",
        "flow": "Dense retrieval → candidate pool → reranking → Top-K → kontextus → LLM",
        "summary": "A szemantikus retrievert külön vizsgálja második körös újrarangsorolással.",
        "when": "Ha a dense retrieval recall jó, de a rangsor finomítása fontos.",
    },
    "hyde": {
        "name": "HyDE RAG",
        "flow": "Kérdés → hipotetikus passage → dense retrieval → evidence → LLM",
        "summary": "Az LLM egy hipotetikus releváns szöveget generál, és annak embeddingjével keres.",
        "when": "Rövid vagy absztrakt kérdések, ahol a query és a dokumentum nyelvezete eltérhet.",
    },
    "multi-query": {
        "name": "Multi-Query RAG",
        "flow": "Kérdés → több lekérdezés → visszakeresés → deduplikáció/rangfúzió → kontextus → LLM",
        "summary": "Több megfogalmazással növeli annak esélyét, hogy eltérő releváns részleteket találjon.",
        "when": "Összetett vagy többféleképpen megfogalmazható kérdések.",
    },
    "query-rewrite": {
        "name": "Query-Rewrite RAG",
        "flow": "Eredeti kérdés → átírt keresési lekérdezés → visszakeresés → kontextus → LLM",
        "summary": "A felhasználói kérdést keresésbarát lekérdezéssé alakítja.",
        "when": "Hosszú, zajos, beszélt nyelvi vagy kontextusfüggő kérdések.",
    },
    "multi-hop": {
        "name": "Multi-Hop RAG",
        "flow": "Első retrieval → evidence → második lekérdezés → második retrieval → RRF → LLM",
        "summary": "Két korlátozott retrieval hopot használ; a második keresés az első evidence alapján készül.",
        "when": "Többlépéses kérdések, ahol egyetlen retrieval kör nem ad minden szükséges részletet.",
    },
    "parent-document": {
        "name": "Parent-Document RAG",
        "flow": "Gyermek-szövegrész visszakeresése → szülő visszakeresése → tágabb kontextus → LLM",
        "summary": "Kis egységen keres, nagyobb egységet generáltat.",
        "when": "A releváns mondat körüli teljes szakasz is kell a pontos válaszhoz.",
    },
    "compression": {
        "name": "Contextual Compression RAG",
        "flow": "Visszakeresés → releváns rész kivonása → tömör kontextus → LLM",
        "summary": "A retrieved chunkokból csak a kérdéshez közel álló részeket tartja meg.",
        "when": "Szűk kontextusablak vagy sok zajos retrieved szöveg.",
    },
    "corrective": {
        "name": "Corrective RAG",
        "flow": "Visszakeresés → relevanciaellenőrzés → szükség esetén lekérdezés-átírás + újrakeresés → LLM",
        "summary": "Gyenge evidence esetén kontrolláltan megpróbálja javítani a keresést.",
        "when": "Heterogén korpusz és változó retrieval-minőség.",
    },
}


CONTEXT_PROFILES: dict[str, dict[str, str | int]] = {
    "balanced": {
        "name": "Kiegyensúlyozott",
        "summary": "Általános célú profil: a legjobb találatokból épít stabil, kiegyensúlyozott kontextust.",
        "budget": 1800,
        "guidance": "Tartsd meg a legerősebb és legváltozatosabb bizonyítékokat, kerüld a felesleges ismétlést.",
    },
    "precise": {
        "name": "Precíz / szűk",
        "summary": "Kevesebb, de magasabban rangsorolt evidence a tömörebb, célzott válaszhoz.",
        "budget": 1200,
        "guidance": "Elsősorban a legmagasabb rangú, közvetlenül releváns evidence-ek maradjanak bent.",
    },
    "broad": {
        "name": "Széles / összehasonlító",
        "summary": "Nagyobb forrásdiverzitás és tágabb kontextus, ha több aspektust akarsz összevetni.",
        "budget": 2600,
        "guidance": "Engedj be több forrásból több részletet, és emeld ki az eltérő nézőpontokat.",
    },
    "compact": {
        "name": "Kompakt",
        "summary": "Szűk kontextusablak és gyorsabb futás rövid válaszokhoz vagy gyengébb hardverhez.",
        "budget": 900,
        "guidance": "Csak a legfontosabb, legkevésbé redundáns evidence-ek maradjanak bent.",
    },
}


PROMPT_PROFILES: dict[str, dict[str, str]] = {
    "professional": {
        "name": "Professzionális összefoglaló",
        "summary": "Részletes, jól strukturált szakmai összefoglaló rövid, de tartalmas bekezdésekkel.",
        "instruction": "Fogalmazz szakmai, letisztult és magyarázó stílusban; több részkérdésnél szakaszonként 3–5 teljes mondatot adj, ha az evidence ezt lehetővé teszi, és minden lényegi állítást forrással támassz alá.",
    },
    "teaching": {
        "name": "Oktató / magyarázó",
        "summary": "Didaktikusabb, magyarázó válasz definíciókkal és rövid értelmezéssel.",
        "instruction": "Magyarázd el egyszerűen, de szakmailag pontosan a kulcsfogalmakat, és röviden értelmezd is a bizonyítékokat.",
    },
    "audit": {
        "name": "Audit / ellenőrző",
        "summary": "Állítás → bizonyíték fókusz, a forráskövetés jól látszik.",
        "instruction": "A válasz legyen állítás-orientált: ahol lehet, sorold fel külön a fő állításokat és a hozzájuk tartozó bizonyítékokat.",
    },
    "concise": {
        "name": "Nagyon tömör",
        "summary": "Rövid, bulletpontos, gyorsan áttekinthető kimenet.",
        "instruction": "Válaszolj kifejezetten tömören, lehetőleg rövid felsorolásban, a legfontosabb 3–5 ponttal.",
    },
}


PAGE_METRICS: dict[str, list[tuple[str, str]]] = {
    "overview": [
        ("Folyamatszintű áttekintés", "Az irányítópult a korpusz, a runtime, az aktív retrieval/RAG konfiguráció és a benchmark állapot legfontosabb adatait foglalja össze."),
        ("Aktív konfiguráció", "A kiválasztott darabolási, retrieval, újrarangsorolási, kontextus- és LLM-beállítások kártyákon jelennek meg, külön architektúra-diagram nélkül."),
    ],
    "documents": [
        ("Dokumentumok / oldalak", "A parser által létrehozott dokumentumegységek száma; PDF-nél jellemzően oldalanként egy egység."),
        ("Karakterek tisztítás előtt/után", "Megmutatja, mennyi zajt távolított el a cleaning pipeline."),
        ("Duplikátumok", "Azonos normalizált tartalmú dokumentumrészek eltávolításának száma."),
    ],
    "chunking": [
        ("Szövegrészek száma", "A létrehozott visszakeresési egységek száma."),
        ("Átlag / medián / P95 hossz", "A chunkméret-eloszlás stabilitását és szélsőértékeit mutatja."),
        ("Darabolási késleltetés", "A feldarabolás teljes futási ideje; semantic chunkingnál embedding időt is tartalmazhat."),
        ("Szövegrész / dokumentum", "Mennyire darabolja fel a forrásokat a stratégia."),
    ],
    "embedding": [
        ("Dimenzió", "Az embedding vektor komponenseinek száma."),
        ("Vektor/s", "Embedding throughput: másodpercenként hány chunk vektorizálható."),
        ("ms/szövegrész", "Egy chunk átlagos embedding-költsége."),
        ("Vektornorma", "Normalizált embeddingnél az L2 norma közel 1; cosine keresésnél ez fontos."),
    ],
    "vector": [
        ("Visszakeresési késleltetés", "A query feldolgozása és a Top-K találat előállítása közötti idő."),
        ("Top-K", "Ennyi eredmény kerül a végső találati listába."),
        ("Pontszám", "Dense keresésnél normalizált inner product ≈ cosine similarity; BM25/RRF esetén más skálájú pontszám."),
        ("Forrásdiverzitás", "Hány különböző dokumentumból érkeznek a Top-K találatok."),
    ],
    "rag": [
        ("Retrieval latency", "A releváns evidence-ek visszakeresésének ideje."),
        ("Reranking latency", "A jelöltek második körös újrarendezésének ideje."),
        ("TTFT", "Time to First Token: mennyi idő után érkezik az első generált token streaming Ollama esetén."),
        ("Token/s", "Az LLM dekódolási throughputja: másodpercenként hány output token készül."),
        ("Context tokens", "A végső LLM-kontextus becsült tokenmennyisége."),
        ("Citation accuracy", "A válaszban lévő [Sx] hivatkozások érvényes retrieved evidence-re mutatnak-e."),
        ("Citation coverage", "A válasz állításainak mekkora része kap explicit hivatkozást; transzparens proxy."),
        ("Context utilization", "A válasz lexikális tartalmának mekkora része támaszkodik a rendelkezésre adott evidence-re."),
    ],
    "evaluation": [
        ("Recall@K", "A forrásolt evidence-hez illeszkedő releváns chunkok mekkora részét találtuk meg a Top-K között."),
        ("Precision@K", "A Top-K találatok mekkora része evidence-szempontból releváns."),
        ("F1@K", "A Recall@K és Precision@K harmonikus átlaga."),
        ("Hit Rate@K", "Volt-e legalább egy releváns találat a Top-K listában."),
        ("MRR", "Az első releváns találat reciprok rangjának átlaga."),
        ("MAP@K", "Az Average Precision query-szintű átlaga, amely több releváns találat sorrendjét is figyelembe veszi."),
        ("nDCG@K", "A releváns elemek rangpozícióját is figyelembe vevő normalizált ranking metrika."),
        ("First relevant rank", "Az első evidence-hez illeszkedő találat átlagos rangpozíciója; kisebb érték kedvezőbb."),
        ("Source diversity", "A Top-K találatok hány különböző forrásdokumentumból származnak."),
        ("Duplicate ratio", "A Top-K listán belüli ismétlődő chunkok aránya; kisebb érték kedvezőbb."),
        ("P50/P95/P99 latency", "A tipikus és tail-latency külön mérése, nem csak egyetlen átlag."),
        ("QPS", "Retrieval throughput: másodpercenként hány query dolgozható fel."),
        ("Key-fact coverage", "Az expected key factek mekkora része jelenik meg lexikálisan a válaszban; proxy, nem klinikai correctness judge."),
        ("Citation accuracy", "A válasz [Sx] hivatkozásai érvényes retrieved evidence-ekre mutatnak-e."),
        ("Citation coverage", "Mekkora a hivatkozással ellátott válaszmondatok aránya."),
        ("Citation source coverage", "A visszakeresett források mekkora része jelenik meg ténylegesen hivatkozásként."),
        ("Context utilization", "Mennyire hasznosul a retrievalből felépített kontextus a végső válaszban."),
        ("Answer redundancy", "Ismétlődő válaszmondatok lexikális proxyja; kisebb érték kedvezőbb."),
        ("Label coverage", "A dataset itemek mekkora részéhez sikerült az adott chunking mellett evidence-alapú releváns chunkot feloldani."),
    ],
}
