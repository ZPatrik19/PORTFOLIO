from __future__ import annotations

import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from rag_engine.ingestion.chunking.factory import create_chunker
from rag_engine.ingestion.cleaning import clean_documents
from rag_engine.ingestion.parser import parse_file
from rag_engine.presets import CHUNKING_STRATEGIES
from rag_engine.models import ChunkingConfig
from ui.charts.chunking import chunk_document_strip_chart
from ui.components.charts import (
    chunk_boundary_chart,
    chunk_length_chart,
    chunk_sequence_chart,
    strategy_comparison_chart,
)
from ui.components.common import active_paths, cached_embedding_provider, selected_embedding_model
from ui.components.education import info_cards, metrics_reference, page_intro
from ui.components.tables import safe_dataframe


@st.cache_data(show_spinner="Dokumentum parsing + cleaning...")
def _load_document(path_text: str, mtime_ns: int):
    del mtime_ns
    docs = parse_file(Path(path_text))
    return clean_documents(docs)


def _chunk_stats(chunks, elapsed_ms: float) -> dict[str, float | int]:
    lengths = [len(chunk.text) for chunk in chunks]
    return {
        "count": len(chunks),
        "mean": statistics.fmean(lengths) if lengths else 0.0,
        "median": statistics.median(lengths) if lengths else 0.0,
        "p95": float(np.percentile(lengths, 95)) if lengths else 0.0,
        "min": min(lengths, default=0),
        "max": max(lengths, default=0),
        "tokens": sum(max(1, length // 4) for length in lengths),
        "latency_ms": elapsed_ms,
    }


def _make_chunker(strategy: str, docs, size: int, overlap: int, threshold: float, semantic_device: str):
    embedder = None
    if strategy == "semantic":
        model_name, fallback_embedding = selected_embedding_model()
        embedder = cached_embedding_provider(model_name, semantic_device, fallback_embedding)
    config = ChunkingConfig(
        strategy=strategy,
        szövegrész_méret=size,
        chunk_átfedés=overlap,
        szemantikus_küszöb=threshold,
    )
    return create_chunker(config, embedder=embedder), embedder


def render() -> None:
    strategy = st.session_state.get("chunking_strategy", "recursive")
    details = CHUNKING_STRATEGIES[strategy]
    page_intro(
        "Darabolási labor",
        "Ugyanazon magyar dokumentumon vizsgálhatod, hogyan változtatja meg a darabolási stratégia a visszakeresési egységeket. A cél nem a legtöbb vagy legkevesebb chunk, hanem olyan egységek létrehozása, amelyek jól kereshetők, és elegendő kontextust tartanak meg.",
        eyebrow="DOKUMENTUM → SZÖVEGRÉSZ-HATÁROK → VISSZAKERESHETŐ EGYSÉGEK",
    )
    metrics_reference("chunking")

    info_cards(
        [
            (details["name"], details["summary"]),
            ("Hogyan működik?", details["how"]),
            ("Mikor jó?", details["best_for"]),
            ("Kompromisszum", details["tradeoff"]),
        ],
        columns=4,
    )

    paths = active_paths()
    if not paths:
        st.warning("Nincs aktív dokumentum. Előbb a Dokumentumok oldalon válassz korpuszt.")
        return

    selected_path = st.selectbox(
        "Vizsgált dokumentum",
        paths,
        format_func=lambda path: path.name,
    )
    docs, cleaning = _load_document(str(selected_path), selected_path.stat().st_mtime_ns)
    if not docs:
        st.warning("A dokumentumból a cleaning után nem maradt feldolgozható tartalom.")
        return

    max_parts = min(len(docs), 60)
    if max_parts == 1:
        part_count = 1
        st.caption("A kiválasztott fájl egyetlen feldolgozható dokumentumrészt tartalmaz.")
    else:
        part_count = st.slider(
            "Elemzett dokumentumrészek / PDF-oldalak",
            1,
            max_parts,
            min(max_parts, 12),
            help="Nagy PDF-nél a darabolási labor szándékosan mintán dolgozik, hogy a stratégia interaktívan összehasonlítható maradjon.",
        )
    sample_docs = docs[:part_count]
    size = int(st.session_state.get("chunk_size", 700))
    overlap = int(st.session_state.get("chunk_overlap", 100))
    threshold = float(st.session_state.get("semantic_threshold", 0.72))
    semantic_device = st.session_state.get("embedding_device", "auto")

    st.markdown(
        f"**Aktív paraméterek:** `{details['name']}` · szövegrész_méret=`{size}` · átfedés=`{overlap}`"
        + (f" · szemantikus_küszöb=`{threshold:.2f}` · eszköz=`{semantic_device}`" if strategy == "semantic" else "")
    )

    try:
        chunker, embedder = _make_chunker(strategy, sample_docs, size, overlap, threshold, semantic_device)
        started = time.perf_counter()
        chunks = chunker.chunk(sample_docs)
        elapsed = (time.perf_counter() - started) * 1000
    except Exception as exc:
        st.error(f"A chunking futás sikertelen: {exc}")
        return

    stats = _chunk_stats(chunks, elapsed)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Szövegrészek", stats["count"])
    c2.metric("Átlag", f"{stats['mean']:.0f} karakter")
    c3.metric("Medián", f"{stats['median']:.0f}")
    c4.metric("P95", f"{stats['p95']:.0f}")
    c5.metric("Becsült token", f"{int(stats['tokens']):,}")
    c6.metric("Futási idő", f"{stats['latency_ms']:.1f} ms")
    if embedder is not None:
        st.caption(
            f"Szemantikus beágyazási modell: {getattr(embedder, 'model_name', 'ismeretlen')} · tényleges eszköz: {getattr(embedder, 'device', semantic_device)}"
        )

    tabs = st.tabs(["Dokumentumszalag", "Vizualizáció", "Eloszlások", "Kiválasztott szövegrész", "A/B összehasonlítás"])

    with tabs[0]:
        chunk_document_strip_chart(
            chunks,
            key=f"chunk_strip_{strategy}_{size}_{overlap}_{part_count}",
            title=f"Dokumentumszalag · {details['name']}",
        )
        st.caption(
            "A szalag a chunkok relatív sorrendjét és méretét szemlélteti. Nem állít pontos karakter-offsetet olyan stratégiáknál, amelyek nem tárolnak explicit forráspozíciót."
        )

    with tabs[1]:
        selected_idx = 0
        if chunks:
            selected_idx = (
                st.select_slider(
                    "Kiemelt szövegrész",
                    options=list(range(1, len(chunks) + 1)),
                    value=1,
                    help="A diagramon a kiválasztott szövegrész kiemelve jelenik meg.",
                )
                - 1
            )
        chunk_boundary_chart(
            chunks,
            selected_index=selected_idx,
            key=f"chunk_boundary_{strategy}_{size}_{overlap}_{part_count}_{selected_idx}",
        )
        st.caption(
            "A kiemelt oszlop a kiválasztott szövegrész. A diagram egyszerre mutatja a szövegrészek sorrendjét és méretét, így könnyebb észrevenni a túl kicsi, túl nagy vagy szabálytalan határokat."
        )

        preview_count = min(len(chunks), 9)
        for start in range(0, preview_count, 3):
            cols = st.columns(3)
            for display_idx, (col, chunk) in enumerate(
                zip(cols, chunks[start : start + 3], strict=False), start=start + 1
            ):
                meta = chunk.metadata
                text = chunk.text.replace("<", "&lt;").replace(">", "&gt;")
                short = text[:420] + ("…" if len(text) > 420 else "")
                highlight = (
                    "border:2px solid #3979d3; background: rgba(57,121,211,.06);"
                    if display_idx - 1 == selected_idx
                    else ""
                )
                col.markdown(
                    f"""
<div class="rag-card" style="min-height:260px; {highlight}">
<h4>Szövegrész {display_idx}</h4>
<div class="rag-small">{len(chunk.text)} karakter · oldal: {meta.get("page", "—")} · szekció: {meta.get("section", "—")}</div>
<hr/>
<p>{short}</p>
</div>
""",
                    unsafe_allow_html=True,
                )

    with tabs[2]:
        chart1, chart2 = st.columns(2)
        with chart1:
            chunk_length_chart(chunks, key=f"chunk_hist_{strategy}_{size}_{overlap}_{part_count}")
        with chart2:
            chunk_sequence_chart(chunks, key=f"chunk_seq_{strategy}_{size}_{overlap}_{part_count}")

    with tabs[3]:
        if chunks:
            selected = chunks[selected_idx]
            info_cards(
                [
                    ("Kiválasztott szövegrész", f"#{selected_idx + 1}"),
                    ("Karakterek", str(len(selected.text))),
                    ("Oldal", str(selected.metadata.get("page", "—"))),
                    ("Szekció", str(selected.metadata.get("section", "—"))),
                ],
                columns=4,
            )
            st.code(selected.text, language=None)
            st.json(selected.metadata)

    with tabs[4]:
        compare = st.multiselect(
            "Összehasonlított stratégiák",
            list(CHUNKING_STRATEGIES),
            default=["fixed", "recursive", "sentence", "paragraph"],
            format_func=lambda key: CHUNKING_STRATEGIES[key]["name"],
        )
        if st.button("Darabolási benchmark futtatása", disabled=not compare):
            records = []
            for key in compare:
                try:
                    current, _ = _make_chunker(key, sample_docs, size, overlap, threshold, semantic_device)
                    t0 = time.perf_counter()
                    current_chunks = current.chunk(sample_docs)
                    ms = (time.perf_counter() - t0) * 1000
                    current_stats = _chunk_stats(current_chunks, ms)
                    records.append(
                        {
                            "Stratégia": CHUNKING_STRATEGIES[key]["name"],
                            "Szövegrészek": current_stats["count"],
                            "Átlagos hossz": round(float(current_stats["mean"]), 1),
                            "Medián": round(float(current_stats["median"]), 1),
                            "P95": round(float(current_stats["p95"]), 1),
                            "Idő (ms)": round(float(current_stats["latency_ms"]), 2),
                        }
                    )
                except Exception as exc:
                    records.append({"Stratégia": CHUNKING_STRATEGIES[key]["name"], "Hiba": str(exc)})
            safe_dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
            valid_records = [record for record in records if "Hiba" not in record]
            strategy_comparison_chart(valid_records, key=f"chunk_compare_{part_count}_{size}_{overlap}")
            st.info(
                "A kisebb késleltetés nem jelent automatikusan jobb visszakeresési minőséget. A darabolást ugyanazon lekérdezés- és címkekészlettel érdemes Recall@K/MRR alapján is összevetni."
            )

    with st.expander("Tisztítási háttérinformáció"):
        st.json(cleaning.__dict__)
