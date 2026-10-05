from __future__ import annotations

import statistics
import time

import pandas as pd
import streamlit as st

from rag_engine.presets import HUNGARIAN_QUERY_PRESETS
from ui.components.charts import retrieval_rank_flow_chart, retrieval_scores_chart
from ui.components.common import get_lab
from ui.components.education import empty_state, evidence_cards, info_cards, metrics_reference, note_box, page_intro
from ui.components.reranker import build_selected_reranker, reranker_status_text
from ui.components.tables import safe_dataframe



def _retrieve(lab, mode: str, query: str, top_k: int, candidate_count: int):
    if mode == "dense":
        return lab.dense.retrieve(query, top_k)
    if mode == "bm25":
        return lab.sparse.retrieve(query, top_k)
    return lab.hybrid.retrieve(query, top_k=top_k, candidate_count=candidate_count)



def render() -> None:
    page_intro(
        "Visszakeresési és vektorkeresési labor",
        "Itt még nincs LLM-generálás: kizárólag azt vizsgáljuk, hogy egy magyar lekérdezésre mely szövegrészeket találja meg a visszakeresés. Ez a RAG egyik legfontosabb diagnosztikai pontja, mert rossz bizonyítékból a generatív modell sem tud megbízható választ készíteni.",
        eyebrow="LEKÉRDEZÉS → DENSE / BM25 / HYBRID → TOP-K BIZONYÍTÉK",
    )
    metrics_reference("vector")

    mode = st.session_state.get("retrieval_mode", "hybrid")
    mode_cards = {
        "dense": ("Dense visszakeresés", "A query embeddinget hasonlítja a chunk embeddingekhez. Jó szemantikus egyezésre, de pontos ritka kulcsszavakat néha elvéthet."),
        "bm25": ("BM25", "Lexikális term-frequency alapú rangsorolás. Erős konkrét kifejezéseknél, neveknél, rövidítéseknél és jogi terminusoknál."),
        "hybrid": ("Hibrid visszakeresés", "Dense + BM25 eredményeket egyesít. RRF esetén a végső score rangfúziós érték; nem probability és nem cosine similarity."),
    }
    info_cards(
        [
            mode_cards[mode],
            ("Top-K", "A végső találati lista mérete. Túl kicsi K ronthatja a recallt, túl nagy K zajt és nagyobb kontextust okozhat."),
            ("Jelöltek száma", "Hibrid/újrarangsorolt folyamatban ennyi jelöltet kérünk az első retrieval körből, mielőtt szűkítünk."),
            ("Mit nézz?", "Ne csak a pontszámot: olvasd el a szövegrészt, ellenőrizd a forrást/oldalt, a rangsort és azt, hogy a releváns bizonyíték bekerült-e."),
        ],
        columns=4,
    )

    preset = st.selectbox(
        "Magyar lekérdezésminta",
        HUNGARIAN_QUERY_PRESETS,
        format_func=lambda item: f"{item.label} · {item.topic}",
    )
    query = st.text_area(
        "Lekérdezés / keresési kérdés",
        value=preset.query,
        height=90,
        key=f"vector_query_{preset.label}",
    )
    top_k = int(st.session_state.get("top_k", 5))
    candidate_count = int(st.session_state.get("candidate_count", max(20, top_k)))
    _, reranker_preview = build_selected_reranker("reranked", allow_inactive=True)
    st.caption(
        f"Aktív visszakereső: **{mode}** · Top-K: **{top_k}** · jelöltek: **{candidate_count}**"
        + (f" · rangfúzió: **{st.session_state.get('fusion_mode', 'rrf')}**" if mode == "hybrid" else "")
        + f" · újrarangsorolás: **{reranker_status_text(reranker_preview)}**"
    )

    if st.button("Keresés futtatása", type="primary"):
        try:
            lab = get_lab()
            started = time.perf_counter()
            results = _retrieve(lab, mode, query, top_k, candidate_count)
            latency_ms = (time.perf_counter() - started) * 1000
            reranker, reranker_status = build_selected_reranker("reranked", allow_inactive=True)
            reranked_results = []
            reranking_ms = 0.0
            if reranker is not None:
                candidate_pool = _retrieve(lab, mode, query, candidate_count, candidate_count)
                r0 = time.perf_counter()
                reranked_results = reranker.rerank(query, candidate_pool, top_k=top_k)
                reranking_ms = (time.perf_counter() - r0) * 1000
            st.session_state["last_vector_results"] = results
            st.session_state["last_vector_reranked"] = reranked_results
            st.session_state["last_vector_reranking_ms"] = reranking_ms
            st.session_state["last_vector_reranker_status"] = reranker_status_text(reranker_status)
            st.session_state["last_vector_latency"] = latency_ms
            st.session_state["last_vector_mode"] = mode
            st.session_state["last_vector_query"] = query
            st.session_state["last_vector_runtime"] = {
                "vector_backend": getattr(lab.vector_store, "backend_name", "ismeretlen"),
                "vector_device": getattr(lab.vector_store, "device", "ismeretlen"),
                "embedding_model": getattr(lab.embedder, "model_name", "ismeretlen"),
                "embedding_device": getattr(lab.embedder, "device", "ismeretlen"),
                "fusion": st.session_state.get("fusion_mode", "rrf"),
            }

            if mode == "hybrid":
                st.session_state["last_dense_results"] = lab.dense.retrieve(query, min(candidate_count, 10))
                st.session_state["last_bm25_results"] = lab.sparse.retrieve(query, min(candidate_count, 10))
        except Exception as exc:
            st.error(f"A retrieval sikertelen: {exc}")

    results = st.session_state.get("last_vector_results", [])
    if not results:
        empty_state(
            "Még nincs retrieval futás",
            "Adj meg egy kérdést, majd futtasd a keresést. Ez az oldal az LLM előtt mutatja meg a candidate pool, a rangsor és a források minőségét.",
            hint="Első próbához a Hybrid / RRF + Top-K=5 beállítás jó baseline.",
        )
        return

    latency_ms = float(st.session_state.get("last_vector_latency", 0.0))
    display_mode = st.session_state.get("last_vector_mode", mode)
    runtime = st.session_state.get("last_vector_runtime", {})
    scores = [float(item.score) for item in results]
    sources = {str(item.metadata.get("source") or item.source) for item in results}
    score_gap = scores[0] - scores[1] if len(scores) > 1 else 0.0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Visszakeresési késleltetés", f"{latency_ms:.1f} ms")
    c2.metric("Találatok", len(results))
    c3.metric("Külön forrás", len(sources))
    c4.metric("Átlagpontszám", f"{statistics.fmean(scores):.4f}")
    c5.metric("Top1–Top2 gap", f"{score_gap:.4f}")

    tabs = st.tabs(["Áttekintés", "Retrieval Inspector", "Újrarangsorolás", "Találati lista", "Bizonyítékok", "Diagnosztika"])

    with tabs[0]:
        retrieval_scores_chart(results, key=f"retrieval_scores_{display_mode}_{len(results)}")
        if display_mode == "dense":
            note_box("Pontszám értelmezése", "Dense módban a projekt normalizált vektorokat és skalárszorzat keresést használ; a pontszám koszinusz-hasonlóságként értelmezhető.")
        elif display_mode == "bm25":
            note_box("Pontszám értelmezése", "A BM25 score lexikális rangsorolási érték. Nem probability és nem cosine similarity.")
        else:
            fusion = runtime.get("fusion", "rrf")
            note_box(
                "Pontszám értelmezése",
                "Hybrid módban a végső score rangfúzió eredménye. "
                + ("RRF esetén rangpozíciókból képzett érték, ezért ne relevanciavalószínűségként értelmezd." if fusion == "rrf" else "Weighted fusion esetén normalizált dense/BM25 komponensek kombinációja."),
            )

    with tabs[1]:
        dense = st.session_state.get("last_dense_results", [])
        bm25 = st.session_state.get("last_bm25_results", [])
        if display_mode == "hybrid" and (dense or bm25):
            retrieval_rank_flow_chart(
                dense,
                bm25,
                results,
                key=f"retrieval_rank_flow_{len(results)}_{len(dense)}_{len(bm25)}",
                title="Rangmozgás · candidate pool → final ranking",
            )
            dense_rank = {str(x.chunk_id): int(x.rank) for x in dense}
            bm25_rank = {str(x.chunk_id): int(x.rank) for x in bm25}
            inspector = pd.DataFrame(
                [
                    {
                        "Final rank": int(item.rank),
                        "Chunk": str(item.chunk_id),
                        "Dense rank": dense_rank.get(str(item.chunk_id)),
                        "BM25 rank": bm25_rank.get(str(item.chunk_id)),
                        "Final score": round(float(item.score), 5),
                        "Forrás": item.metadata.get("title") or item.source,
                    }
                    for item in results
                ]
            )
            # Keep rank columns numeric and nullable. Mixing integers with the em dash
            # string creates an ``object`` dtype that PyArrow cannot serialize reliably.
            inspector["Dense rank"] = pd.array(inspector["Dense rank"], dtype="Int64")
            inspector["BM25 rank"] = pd.array(inspector["BM25 rank"], dtype="Int64")
            safe_dataframe(
                inspector,
                width="stretch",
                hide_index=True,
                column_config={
                    "Final rank": st.column_config.NumberColumn(format="%d"),
                    "Dense rank": st.column_config.NumberColumn(format="%d"),
                    "BM25 rank": st.column_config.NumberColumn(format="%d"),
                    "Final score": st.column_config.NumberColumn(format="%.5f"),
                },
            )
            note_box(
                "Mit mutat ez?",
                "A chart azt teszi láthatóvá, hogy ugyanaz a chunk milyen helyen állt a dense és BM25 candidate listában, majd hova került a final hybrid rangsorban. Ez retrieval-debuggingnál sokkal informatívabb, mint egyetlen végső score.",
            )
        else:
            empty_state(
                "A rangmozgás Hybrid módban a leghasznosabb",
                "Válts Hybrid retrievalre, majd futtasd újra a keresést. A rendszer ekkor külön eltárolja a Dense és BM25 candidate listát is.",
            )

    with tabs[2]:
        reranked = st.session_state.get("last_vector_reranked", [])
        rerank_status = st.session_state.get("last_vector_reranker_status", "Nincs újrarangsorolás")
        rerank_ms = float(st.session_state.get("last_vector_reranking_ms", 0.0))
        st.caption(f"Aktív újrarangsoroló: **{rerank_status}** · futási idő: **{rerank_ms:.1f} ms**")
        if reranked:
            retrieval_scores_chart(reranked, key=f"reranked_scores_{display_mode}_{len(reranked)}", title="Újrarangsorolt Top-K pontszámok")
            evidence_cards(reranked, columns=2)
        else:
            empty_state("Nincs újrarangsorolt lista", "A globális újrarangsoroló ki van kapcsolva vagy a jelenlegi futás nem készített reranked eredményt.")

    with tabs[3]:
        table = pd.DataFrame(
            [
                {
                    "Rang": item.rank,
                    "Pontszám": round(float(item.score), 5),
                    "Forrás": item.metadata.get("title") or item.source,
                    "Oldal": item.metadata.get("page"),
                    "Szövegrész": item.chunk_id,
                }
                for item in results
            ]
        )
        safe_dataframe(table, width="stretch", hide_index=True)

    with tabs[4]:
        evidence_cards(results, columns=2)
        with st.expander("Nyers bizonyítékok + metadata", expanded=False):
            for item in results:
                title = item.metadata.get("title") or item.source or "ismeretlen forrás"
                page = item.metadata.get("page")
                st.markdown(f"**#{item.rank} · {title} · oldal {page or '—'} · score {item.score:.5f}**")
                st.write(item.text)
                st.json(item.metadata)

    with tabs[5]:
        st.caption(
            f"Tényleges vektorbackend: **{runtime.get('vector_backend', 'ismeretlen')} / {runtime.get('vector_device', 'ismeretlen')}** · "
            f"beágyazás: **{runtime.get('embedding_model', 'ismeretlen')} / {runtime.get('embedding_device', 'ismeretlen')}**"
        )
        info_cards(
            [
                ("Aktív lekérdezésminta témája", preset.topic),
                ("Ajánlott ellenőrzés", "Nézd meg, hogy a releváns bizonyíték már a Top-K-ban megjelenik-e, mielőtt az LLM-hez mennél."),
                ("Forrásdiverzitás", f"{len(sources)} külön forrás a Top-{len(results)} listában"),
                ("Pontszámrés", f"Top1–Top2 gap = {score_gap:.4f}; alacsony gap közel azonos relevanciát jelezhet."),
            ],
            columns=2,
        )
    st.caption("Recall@K, Precision@K, MRR és nDCG csak relevancia-címkével értelmezhető. Ezeket a Kiértékelés oldalon tudod korrektül mérni; pusztán a retrieval score-ból nem következnek.")
