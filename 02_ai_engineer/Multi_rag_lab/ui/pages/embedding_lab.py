from __future__ import annotations

import time

import numpy as np
import streamlit as st

from rag_engine.platform.device import cuda_available
from ui.charts.embedding import embedding_projection_chart
from ui.components.common import get_lab
from ui.components.education import info_cards, metrics_reference, page_intro
from ui.components.tables import safe_dataframe


def render() -> None:
    page_intro(
        "Beágyazási labor",
        "A beágyazási modell a szövegrészeket sűrű numerikus vektorokká alakítja. Dense visszakeresésnél a lekérdezés is ugyanabba a vektortérbe kerül, majd a vektortár a legközelebbi reprezentációkat keresi.",
        eyebrow="SZÖVEGRÉSZEK → BEÁGYAZÁSOK → VEKTORTÉR",
    )
    metrics_reference("embedding")
    info_cards(
        [
            ("Mi a beágyazás (embedding)?", "Egy szöveghez fix dimenziós vektort rendel. A hasonló jelentésű szövegek ideális esetben közel kerülnek egymáshoz a vektortérben."),
            ("Koszinusz-hasonlóság", "A vektorok irányának hasonlóságát méri. Normalizált vektoroknál az inner product megegyezik a cosine similarityvel."),
            ("CPU és CUDA", "A modell matematikája ugyanaz; CUDA nagyobb batchnél jelentősen gyorsíthatja az embedding számítást, de a retrieval minősége ettől nem lesz automatikusan jobb."),
            ("Többnyelvű modell", "Magyar korpusznál a többnyelvű embedding modell a célszerű alapbeállítás, mert magyar szemantikai reprezentációra is tanították."),
        ],
        columns=4,
    )

    requested = st.radio(
        "Ehhez a méréshez használt beágyazási eszköz",
        ["cpu", "cuda", "auto"],
        horizontal=True,
        index=["cpu", "cuda", "auto"].index(st.session_state.get("embedding_device", "auto")),
    )
    if requested == "cuda" and not cuda_available():
        st.warning("A PyTorch ebben a környezetben nem lát CUDA-t. A provider a konfigurált fallback szabály szerint CPU-ra válthat.")

    max_chunks = st.slider("Benchmarkba bevont szövegrészek maximális száma", 10, 1000, 200, 10)
    if st.button("Beágyazási mérés futtatása", type="primary"):
        try:
            lab = get_lab(embedding_device=requested)
            texts = [chunk.text for chunk in lab.ingestion.chunks if chunk.metadata.get("role") != "parent"][:max_chunks]
            if not texts:
                st.warning("Nincs beágyazható szövegrész.")
                return
            started = time.perf_counter()
            vectors = lab.embedder.embed_documents(texts)
            elapsed_ms = (time.perf_counter() - started) * 1000
            norms = np.linalg.norm(vectors, axis=1) if len(vectors) else np.array([])
            throughput = len(texts) / max(elapsed_ms / 1000, 1e-9)
            ms_per_chunk = elapsed_ms / max(len(texts), 1)
            st.session_state["embedding_projection_vectors"] = np.asarray(vectors)
            st.session_state["embedding_projection_chunks"] = [c for c in lab.ingestion.chunks if c.metadata.get("role") != "parent"][:max_chunks]
            st.session_state["embedding_projection_model"] = getattr(lab.embedder, "model_name", "ismeretlen")
            st.session_state["embedding_projection_device"] = getattr(lab.embedder, "device", "ismeretlen")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Modell", getattr(lab.embedder, "model_name", "ismeretlen"))
            c2.metric("Tényleges eszköz", getattr(lab.embedder, "device", "ismeretlen"))
            c3.metric("Dimenzió", vectors.shape[1] if vectors.ndim == 2 else 0)
            c4.metric("Vektor/s", f"{throughput:.1f}")
            c5.metric("ms/szövegrész", f"{ms_per_chunk:.2f}")
            c6.metric("Átlagos L2 norma", f"{norms.mean():.4f}" if len(norms) else "—")

            st.caption(
                "Ha a szolgáltató normalizálja a beágyazásokat, az L2 norma közel 1,0. A projekt FAISS IndexFlatIP keresése így koszinusz-hasonlóság jellegű pontszámot ad."
            )
            safe_dataframe(
                {
                    "Mutató": ["Szövegrészek", "Összes beágyazási idő (ms)", "Minimum norma", "Maximum norma"],
                    "Érték": [len(texts), round(elapsed_ms, 2), round(float(norms.min()), 5), round(float(norms.max()), 5)],
                },
                width="stretch",
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"A beágyazási mérés sikertelen: {exc}")


    vectors = st.session_state.get("embedding_projection_vectors")
    chunks = st.session_state.get("embedding_projection_chunks", [])
    if vectors is not None and len(chunks) >= 2:
        st.divider()
        st.subheader("Beágyazási tér 2D vetítése")
        st.caption("A PCA-vetítés diagnosztikai vizualizáció: a magas dimenziós beágyazási tér szerkezetének kétdimenziós közelítését mutatja, nem helyettesíti a visszakeresési pontszámot.")
        query = st.text_input("Opcionális lekérdezés kiemelése a vetítésen", value="Melyek a magas vérnyomás fő tünetei?", key="embedding_projection_query")
        query_vector = None
        if query:
            try:
                lab = get_lab(embedding_device=requested)
                query_vector = lab.embedder.embed_query(query)
            except Exception as exc:
                st.warning(f"A lekérdezés beágyazása nem készült el: {exc}")
        embedding_projection_chart(
            np.asarray(vectors),
            chunks,
            query_vector=np.asarray(query_vector) if query_vector is not None else None,
            query_label="Lekérdezés",
            key=f"embedding_pca_{len(chunks)}_{st.session_state.get('embedding_projection_device','cpu')}",
        )
