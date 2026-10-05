from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rag_engine.ingestion.cleaning import clean_documents
from rag_engine.ingestion.medical import download_medical_corpus, load_medical_corpus_config
from rag_engine.ingestion.parser import parse_file
from ui.components.common import ROOT, active_paths, cached_lab, get_lab
from ui.components.education import info_cards, kpi_cards, metrics_reference, note_box, page_intro, section_intro
from ui.components.tables import safe_dataframe


MEDICAL_CONFIG_PATH = ROOT / "config" / "medical_corpus.yaml"


@st.cache_data(show_spinner="Dokumentumok parsingja és tisztítása...")
def _analyze_file(path_text: str, mtime_ns: int) -> dict[str, object]:
    del mtime_ns
    path = Path(path_text)
    docs = parse_file(path)
    cleaned, stats = clean_documents(docs)
    return {
        "fájl": path.name,
        "parser egységek": len(docs),
        "tisztított egységek": len(cleaned),
        "karakter előtte": stats.characters_before,
        "karakter utána": stats.characters_after,
        "duplikátum": stats.duplicates_removed,
        "üres/rövid eltávolítva": stats.empty_sections_removed,
        "header/footer sor eltávolítva": stats.repeated_edge_lines_removed,
    }


def render() -> None:
    page_intro(
        "Dokumentumok és magyar orvosi mintakorpusz",
        "A RAG minősége már az ingestionnél eldől. Az elsődleges készlet 100 magyar Egészségvonal orvosi cikkből épül, automatikusan letölthető és indexelhető; saját dokumentumokat is hozzáadhatsz.",
        eyebrow="DATA → DOWNLOAD → PARSE → CLEAN → METADATA",
    )
    metrics_reference("documents")

    info_cards(
        [
            ("PDF", "Oldalanként külön Document objektum készül; az oldalszám megmarad a provenance metadata-ban."),
            (
                "HTML",
                "A script/style zaj eltűnik, a H1/H2/H3 headingek Markdown-jelöléssé alakulnak a structure-aware chunking számára.",
            ),
            ("TXT / Markdown / DOCX", "UTF-8 szöveg és bekezdések kerülnek egységes Document modellbe."),
            (
                "Tisztítás",
                "Unicode NFKC, whitespace-normalizálás, ismétlődő PDF header/footer, rövid blokkok és exact duplikátumok szűrése.",
            ),
            (
                "Provenance",
                "A forrás URL, cím, szervezet, kategória, oldal és SHA-256 metadata továbbmegy a chunkokhoz és citationökhöz.",
            ),
            ("Idempotens letöltés", "Azonos SHA-256 esetén a downloader nem írja újra ugyanazt a fájlt."),
        ]
    )

    section_intro(
        "Magyar orvosi mintakorpusz – 100 cikk",
        "Az elsődleges RAG-korpusz az Egészségvonal Egészség A–Z nyilvánosan elérhető magyar egészségügyi cikkeiből épül. A rendszer automatikusan fedezi fel, tölti le, aktiválja és igény szerint indexeli a dokumentumokat.",
    )
    medical_cfg = load_medical_corpus_config(MEDICAL_CONFIG_PATH)
    medical_dir = ROOT / medical_cfg.output_dir
    existing_medical = sorted(medical_dir.glob("*.html")) if medical_dir.exists() else []
    manifest_path = medical_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    kpi_cards(
        [
            ("Célméret", f"{medical_cfg.target_documents} cikk", "Magyar Egészségvonal Egészség A–Z"),
            (
                "Lokálisan elérhető",
                f"{len(existing_medical)} cikk",
                "Újrafuttatáskor a meglévő fájlok újrahasznosulnak",
            ),
            ("Forrás", "Egészségvonal / NNGYK", "Hivatalos magyar egészségügyi tájékoztatás"),
            ("Index", "FAISS / fallback", str(ROOT / medical_cfg.vectorstore_dir)),
        ],
        columns=4,
    )
    note_box("Orvosi biztonság", medical_cfg.disclaimer)

    target_count = st.slider(
        "Orvosi dokumentumok száma", 20, 200, medical_cfg.target_documents, step=10, key="medical_target_count"
    )
    m1, m2, m3 = st.columns(3)
    if m1.button(f"{target_count} cikk letöltése + aktiválása + indexelése", type="primary", width="stretch"):
        progress = st.progress(0.0, text="Egészségvonal cikkek felfedezése...")
        status = st.empty()

        def _progress(current: int, total: int, title: str) -> None:
            status.write(f"**{current}/{total}** – {title}")
            progress.progress(current / max(total, 1), text=f"{current}/{total} cikk feldolgozva")

        try:
            result = download_medical_corpus(medical_cfg, medical_dir, target=target_count, progress_callback=_progress)
            paths = sorted(str(path) for path in medical_dir.glob("*.html"))[:target_count]
            st.session_state["document_paths"] = paths
            cached_lab.clear()
            _analyze_file.clear()
            status.write("**Indexelés:** parsing → cleaning → chunking → embedding → FAISS")
            with st.spinner(
                f"A {target_count} cikk indexelése folyamatban. Első futáskor a multilingual embedding modell letöltése miatt ez hosszabb lehet..."
            ):
                lab = get_lab()
                output_dir = ROOT / medical_cfg.vectorstore_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                lab.vector_store.save(output_dir)
            st.success(
                f"{len(paths)} magyar orvosi cikk kész és aktív · {len(lab.ingestion.chunks)} chunk · "
                f"backend: {getattr(lab.vector_store, 'backend_name', 'ismeretlen')} · "
                f"device: {getattr(lab.vector_store, 'device', 'ismeretlen')}"
            )
            if result.get("failures"):
                st.warning(f"{len(result['failures'])} forrás letöltése sikertelen volt; a manifestben részletezve.")
        except Exception as exc:
            st.error(f"Az orvosi korpusz előkészítése sikertelen: {exc}")

    if m2.button("Meglévő orvosi korpusz aktiválása", width="stretch", disabled=not existing_medical):
        st.session_state["document_paths"] = [str(path) for path in existing_medical[:target_count]]
        cached_lab.clear()
        st.success(f"{min(target_count, len(existing_medical))} orvosi cikk aktiválva.")

    if m3.button("Aktív orvosi korpusz indexelése", width="stretch", disabled=not existing_medical):
        try:
            st.session_state["document_paths"] = [str(path) for path in existing_medical[:target_count]]
            cached_lab.clear()
            with st.spinner(
                "Parsing → cleaning → chunking → embedding → FAISS index... Ez első futáskor több perc lehet."
            ):
                lab = get_lab()
                output_dir = ROOT / medical_cfg.vectorstore_dir
                output_dir.mkdir(parents=True, exist_ok=True)
                lab.vector_store.save(output_dir)
            st.success(
                f"Index kész: {len(lab.ingestion.chunks)} chunk · backend: "
                f"{getattr(lab.vector_store, 'backend_name', 'ismeretlen')} · "
                f"device: {getattr(lab.vector_store, 'device', 'ismeretlen')}"
            )
        except Exception as exc:
            st.error(f"Az indexépítés sikertelen: {exc}")

    if manifest:
        with st.expander("Orvosi korpusz manifest és forráslista", expanded=False):
            docs = manifest.get("documents", [])
            st.caption(str(manifest.get("license_note") or medical_cfg.license_note))
            if docs:
                safe_dataframe(
                    pd.DataFrame(docs)[["title", "group", "url", "filename"]],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "url": st.column_config.LinkColumn("Eredeti Egészségvonal cikk", display_text="megnyitás")
                    },
                )

    st.divider()
    st.subheader("Saját dokumentumok")
    uploaded = st.file_uploader(
        "PDF, TXT, Markdown, HTML vagy DOCX feltöltése",
        type=["pdf", "txt", "md", "markdown", "html", "htm", "docx"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Feltöltött dokumentumok mentése és aktiválása"):
        raw = ROOT / "data" / "raw" / "uploads"
        raw.mkdir(parents=True, exist_ok=True)
        paths = []
        for item in uploaded:
            target = raw / Path(item.name).name
            target.write_bytes(item.getvalue())
            paths.append(str(target))
        st.session_state["document_paths"] = paths
        cached_lab.clear()
        _analyze_file.clear()
        st.success(f"{len(paths)} dokumentum mentve és aktiválva.")

    st.divider()
    st.subheader("Aktív korpusz és tisztítási statisztikák")
    paths = active_paths()
    if not paths:
        st.warning("Nincs aktív dokumentum.")
        return
    st.caption(
        f"Aktív fájlok: {len(paths)}. A parsing eredmény cache-elt, így a Streamlit rerun nem olvassa újra feleslegesen ugyanazt a fájlt."
    )
    if st.button("Aktív korpusz elemzése"):
        analyzed = []
        for path in paths:
            try:
                analyzed.append(_analyze_file(str(path), path.stat().st_mtime_ns))
            except Exception as exc:
                analyzed.append({"fájl": path.name, "hiba": str(exc)})
        st.session_state["document_analysis"] = analyzed

    analyzed = st.session_state.get("document_analysis", [])
    if analyzed:
        safe_dataframe(pd.DataFrame(analyzed), width="stretch", hide_index=True)
        valid = [row for row in analyzed if "hiba" not in row]
        if valid:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Parser egységek", sum(int(row["parser egységek"]) for row in valid))
            c2.metric("Tisztított egységek", sum(int(row["tisztított egységek"]) for row in valid))
            c3.metric("Duplikátumok eltávolítva", sum(int(row["duplikátum"]) for row in valid))
            before = sum(int(row["karakter előtte"]) for row in valid)
            after = sum(int(row["karakter utána"]) for row in valid)
            c4.metric("Tisztítási csökkenés", f"{(1 - after / before) * 100:.1f}%" if before else "0%")
