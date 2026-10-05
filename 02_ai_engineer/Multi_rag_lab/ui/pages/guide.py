from __future__ import annotations

import streamlit as st

from ui.components.education import info_cards, kpi_cards, note_box, page_intro, section_intro
from ui.components.runtime_status import render_runtime_badges, runtime_snapshot



def render() -> None:
    snapshot = runtime_snapshot()
    page_intro(
        "Használati útmutató",
        "Ez az oldal a teljes rendszert tanulási sorrendben magyarázza el. A cél az, hogy gyorsan átlásd az adatfeldolgozás, a retrieval, a kontextusépítés, a generálás és a kiértékelés kapcsolatát.",
        eyebrow="NAVIGÁTOR / NULLADIK LÉPÉS",
    )

    kpi_cards(
        [
            ("Aktív chunking", str(snapshot["chunking"]), f"Chunk size: {snapshot['chunk_size']} · overlap: {snapshot['overlap']}"),
            ("Visszakeresési mód", str(snapshot["retrieval"]), f"Top-K: {snapshot['top_k']} · rangfúzió: {snapshot['fusion']}"),
            ("Kontextus + prompt", str(snapshot["context_profile"]), str(snapshot["prompt_profile"])),
            ("Futtatási mód", f"LLM: {snapshot['llm']}", f"Embedding: {snapshot['embedding_device']} · FAISS: {snapshot['vector_device']}"),
        ],
        columns=4,
    )

    col1, col2 = st.columns([1.15, 0.85])
    with col1:
        section_intro("Mit csinál egy-egy réteg?", "A rendszer négy nagy zónára bontható: adat, indexelés, retrieval, generálás és mérés.")
        info_cards(
            [
                ("1–3. Adatfeldolgozás", "Letöltés vagy feltöltés után parsing, Unicode- és whitespace-normalizálás, header/footer kezelés, tisztítás, deduplikáció és provenance építés történik."),
                ("4. Darabolás (chunking)", "A hosszú dokumentum kereshető egységekre bomlik. Fix, tokenes, rekurzív, mondat-, bekezdés-, szemantikus, struktúraérzékeny és Parent–Child módszert is össze tudsz vetni."),
                ("5–6. Beágyazás (embedding) + FAISS", "A szövegrészekből numerikus vektorok lesznek. Ezeket indexeli a FAISS vagy fallback exact-search backend, hogy a dense retrieval gyors és reprodukálható legyen."),
                ("7. Visszakeresés (retrieval)", "Dense módban szemantikus, BM25-ben lexikális, hybridben kombinált keresés fut. Az RRF score rangfúziós érték, nem relevanciavalószínűség."),
                ("8–9. Rerank + Kontextus", "A jelöltlista újrarendeződik, majd a context builder deduplikál, forrásdiverzitást kezel és betartja a tokenkeretet."),
                ("10–11. LLM + mérés", "A grounded LLM kizárólag a kapott bizonyíték-ekből válaszol [S1], [S2] jelölésekkel. A quality és performance mérések külön rétegben történnek."),
            ],
            columns=2,
        )
    with col2:
        section_intro("CPU, CUDA, FAISS és LLM kapcsolata", "A kért és a tényleges runtime-réteget külön kezeld: ettől lesz diagnosztizálható a rendszer.")
        render_runtime_badges()
        note_box(
            "Fontos megjegyzés",
            "Ha CUDA-t kérsz, de a környezetben a PyTorch vagy az embedding provider nem lát támogatott GPU-t, a rendszer kontrolláltan CPU fallbackre vált. Ez védőmechanizmus, nem UI-hiba.",
        )

    section_intro("Javasolt munkamenet", "Gyakorlati sorrend, amellyel a labor végigjárható és a változtatások hatása mérhető.")
    for number, text in enumerate(
        [
            "Töltsd le a magyar mintakorpuszt a Dokumentumok oldalon, és ellenőrizd a parsing/cleaning statisztikákat.",
            "A Darabolási laborban hasonlítsd össze ugyanazon dokumentumon a chunking stratégiákat, majd válassz retrieval-szempontból életszerű alapbeállítást.",
            "A Vektorkeresési laborban nézd meg, hogy a kiválasztott chunking + retrieval együtt valóban releváns bizonyíték-et hoz-e fel magyar lekérdezésekre.",
            "A RAG játszótérben teszteld a végső válasz minőségét, a trace-et, a context-utilizációt és a hivatkozási viselkedést.",
            "A Kiértékelés oldalon címkézéssel mérd a retrieval qualityt, majd futtass teljes rendszer proxy-evaluációt több stratégiára.",
        ],
        start=1,
    ):
        st.markdown(f'<div class="rag-step"><b>{number}.</b> {text}</div>', unsafe_allow_html=True)
