from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from tkip.config import resolve_path
from tkip.ingestion import SUPPORTED_EXTENSIONS
from tkip.monitoring import drift_report
from tkip.multi_index import INDEX_STRATEGIES, MultiIndexManager
from tkip.prompt_engineering import PROFILE_KEYS
from tkip.presets import CHUNK_PRESETS
from tkip.workflow_graph import workflow_rows
from ui_workflow import render_orchestration_graph
from i18n import localize_columns, localize_value, pick, tr


def _safe_upload_name(name: str) -> str:
    """Return a filesystem-safe basename for an uploaded knowledge document."""

    basename = Path(name).name.strip()
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", basename)
    return stem[:180] or "document"


def _save_uploaded_documents(uploaded_files, target_dir: Path, *, overwrite: bool) -> tuple[list[str], list[str]]:
    saved: list[str] = []
    skipped: list[str] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        filename = _safe_upload_name(uploaded.name)
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            skipped.append(f"{filename}: unsupported file type")
            continue
        payload = uploaded.getvalue()
        if len(payload) > 100 * 1024 * 1024:
            skipped.append(f"{filename}: larger than 100 MB")
            continue
        destination = target_dir / filename
        if destination.exists() and not overwrite:
            skipped.append(f"{filename}: already exists")
            continue
        destination.write_bytes(payload)
        saved.append(filename)
    return saved, skipped


def _source_display(ui_lang: str, value: str) -> str:
    if value == "private":
        return pick(ui_lang, "Saját könyvtár", "User library")
    if value == "public":
        return pick(ui_lang, "Referenciaanyag", "Reference docs")
    return localize_value(ui_lang, value)


def _render_library_ingestion(kp, ui_lang: str) -> None:
    st.markdown(pick(ui_lang, "### Dokumentum hozzáadása", "### Add documents"))
    st.caption(
        pick(
            ui_lang,
            "A fájlokat feltöltheted itt, vagy kézzel bemásolhatod a megfelelő 01_data mappába. Az index újraépítése után az új tartalom azonnal kereshető.",
            "Upload files here or copy them manually into the matching 01_data folder. After rebuilding the index, the new content becomes searchable immediately.",
        )
    )

    user_dir = resolve_path(kp.cfg["paths"]["user_library"])
    reference_dir = resolve_path(kp.cfg["paths"]["reference_docs"])
    collection = st.radio(
        pick(ui_lang, "Hova kerüljön?", "Destination"),
        ["user_library", "reference_docs"],
        horizontal=True,
        format_func=lambda value: (
            pick(ui_lang, "Saját könyvtár", "User library")
            if value == "user_library"
            else pick(ui_lang, "Referenciaanyagok", "Reference docs")
        ),
        key="library_upload_collection",
    )
    if collection == "user_library":
        target_dir = user_dir
        st.info(
            pick(
                ui_lang,
                "Saját könyvtár: a saját könyveid, jegyzeteid és belső dokumentumaid. Ez a mappa Gitből és a kiadási ZIP-ekből alapértelmezetten ki van zárva.",
                "User library: your own books, notes and internal documents. This folder is excluded from Git and release ZIPs by default.",
            )
        )
    else:
        target_dir = reference_dir
        st.info(
            pick(
                ui_lang,
                "Referenciaanyagok: publikus dokumentációk, saját demo-összefoglalók vagy más újraépíthető referenciaforrások. Csak olyan tartalmat verziókezelj, amelynek licencelése ezt engedi.",
                "Reference docs: public documentation, your own demo summaries or other rebuildable reference sources. Only version content whose license permits redistribution.",
            )
        )

    supported = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
    uploaded = st.file_uploader(
        pick(ui_lang, "Fájlok kiválasztása", "Choose files"),
        type=supported,
        accept_multiple_files=True,
        help=pick(
            ui_lang,
            "Támogatott: PDF, DOCX, TXT, Markdown, HTML és EPUB. Legfeljebb 100 MB fájlonként.",
            "Supported: PDF, DOCX, TXT, Markdown, HTML and EPUB. Maximum 100 MB per file.",
        ),
        key="library_uploader",
    )
    overwrite = st.checkbox(
        pick(ui_lang, "Azonos nevű fájl felülírása", "Overwrite files with the same name"),
        value=False,
        key="library_upload_overwrite",
    )

    add_col, rebuild_col = st.columns(2)
    add_clicked = add_col.button(
        pick(ui_lang, "Feltöltés + index újraépítése", "Upload + rebuild index"),
        type="primary",
        use_container_width=True,
        disabled=not uploaded,
    )
    rebuild_clicked = rebuild_col.button(
        pick(ui_lang, "Csak index újraépítése", "Rebuild index only"),
        use_container_width=True,
        help=pick(
            ui_lang,
            "Akkor használd, ha kézzel másoltál fájlokat a könyvtár mappáiba.",
            "Use this after manually copying files into the library folders.",
        ),
    )

    if add_clicked:
        saved, skipped = _save_uploaded_documents(uploaded, target_dir, overwrite=overwrite)
        if skipped:
            st.warning("\n".join(f"• {item}" for item in skipped))
        if saved:
            with st.status(
                pick(ui_lang, "Dokumentumok indexelése…", "Indexing documents…"),
                expanded=True,
            ) as status:
                report = kp.ingest_and_index()
                st.json(report)
                status.update(
                    label=pick(ui_lang, "Index frissítve", "Index updated"),
                    state="complete",
                    expanded=False,
                )
            st.success(
                pick(
                    ui_lang,
                    f"{len(saved)} fájl hozzáadva: {', '.join(saved)}",
                    f"Added {len(saved)} file(s): {', '.join(saved)}",
                )
            )
            st.rerun()
        elif not skipped:
            st.info(pick(ui_lang, "Nem történt módosítás.", "No changes were made."))

    if rebuild_clicked:
        with st.status(
            pick(ui_lang, "Index újraépítése…", "Rebuilding index…"), expanded=True
        ) as status:
            report = kp.ingest_and_index()
            st.json(report)
            status.update(
                label=pick(ui_lang, "Index frissítve", "Index updated"),
                state="complete",
                expanded=False,
            )
        st.rerun()

    with st.expander(
        pick(ui_lang, "Kézi hozzáadás és mappalogika", "Manual add and folder logic"),
        expanded=False,
    ):
        st.code(
            f"User library   -> {user_dir.relative_to(Path(kp.cfg['_project_root']))}\n"
            f"Reference docs -> {reference_dir.relative_to(Path(kp.cfg['_project_root']))}",
            language="text",
        )
        st.markdown(
            pick(
                ui_lang,
                "1. Másold a fájlt a megfelelő mappába. 2. Kattints a **Csak index újraépítése** gombra. 3. A pipeline discovery → parsing → chunking → embedding → indexelés lépéseken át frissíti a tudásbázist. A forrásfájl nem kerül át a másik gyűjteménybe.",
                "1. Copy the file into the appropriate folder. 2. Click **Rebuild index only**. 3. The pipeline refreshes the knowledge base through discovery → parsing → chunking → embedding → indexing. The source file is never moved to the other collection.",
            )
        )


def render_library(kp, ui_lang: str) -> None:
    documents: dict[str, dict] = {}
    for chunk in kp.chunks:
        row = documents.setdefault(
            chunk.document_id,
            {
                "document_id": chunk.document_id,
                "title": chunk.title,
                "source_type": chunk.source_type,
                "language": chunk.language,
                "chunks": 0,
                "pages": set(),
                "frameworks": set(),
            },
        )
        row["chunks"] += 1
        if chunk.page_start:
            row["pages"].add(chunk.page_start)
        if chunk.framework:
            row["frameworks"].add(chunk.framework)

    rows = [
        {
            **document,
            "pages": len(document["pages"]),
            "frameworks": ", ".join(sorted(document["frameworks"])),
        }
        for document in documents.values()
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["source_type", "title"])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric(tr(ui_lang, "library_documents"), len(frame))
    col_b.metric(tr(ui_lang, "indexed_chunks"), f"{len(kp.chunks):,}")
    private_count = int((frame["source_type"] == "private").sum()) if not frame.empty else 0
    col_c.metric(tr(ui_lang, "private_documents"), private_count)
    display_source = frame.copy()
    if "source_type" in display_source.columns:
        display_source["source_type"] = display_source["source_type"].map(
            lambda value: _source_display(ui_lang, value)
        )
    display_frame = display_source.rename(
        columns=localize_columns(ui_lang, list(display_source.columns))
    )
    st.dataframe(display_frame, use_container_width=True, hide_index=True, height=520)

    _render_library_ingestion(kp, ui_lang)

    st.markdown(f"### {tr(ui_lang, 'chunking_indexes')}")
    manager = MultiIndexManager(kp.cfg)
    index_frame = pd.DataFrame(manager.available())
    display_indexes = index_frame.rename(columns=localize_columns(ui_lang, list(index_frame.columns)))
    st.dataframe(display_indexes, use_container_width=True, hide_index=True)

    with st.expander(tr(ui_lang, "build_indexes"), expanded=False):
        selected = st.multiselect(
            tr(ui_lang, "strategies"),
            list(INDEX_STRATEGIES),
            default=["fixed", "recursive", "semantic"],
        )
        size = st.slider(
            tr(ui_lang, "persistent_chunk_size"),
            400,
            1800,
            int(kp.cfg["chunking"].get("chunk_size", 900)),
            100,
        )
        overlap = st.slider(
            tr(ui_lang, "persistent_overlap"),
            0,
            min(500, size - 1),
            min(int(kp.cfg["chunking"].get("overlap", 120)), size - 1),
            20,
        )
        if st.button(tr(ui_lang, "build_selected_indexes"), type="secondary", disabled=not selected):
            with st.status(tr(ui_lang, "building_indexes"), expanded=True) as status:
                result = manager.build(selected, chunk_size=size, overlap=overlap, force=False)
                st.json(result)
                status.update(label=tr(ui_lang, "indexes_ready"), state="complete")


def render_monitoring(kp, ui_lang: str) -> None:
    monitoring_tab, evaluation_tab = st.tabs(
        [tr(ui_lang, "monitoring_tab"), tr(ui_lang, "retrieval_benchmark_tab")]
    )

    with monitoring_tab:
        rows = kp.telemetry.recent(1000)
        summary = kp.telemetry.summary()
        col_1, col_2, col_3, col_4 = st.columns(4)
        col_1.metric(tr(ui_lang, "requests"), summary.get("requests", summary.get("count", 0)))
        col_2.metric(tr(ui_lang, "p50_latency"), f"{summary.get('p50_latency_ms', 0) or 0:.0f} ms")
        col_3.metric(tr(ui_lang, "p95_latency"), f"{summary.get('p95_latency_ms', 0) or 0:.0f} ms")
        col_4.metric(tr(ui_lang, "success"), f"{100 * float(summary.get('success_rate', 0) or 0):.1f}%")

        if rows:
            frame = pd.DataFrame(rows)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            numeric = [
                column
                for column in [
                    "total_latency_ms",
                    "retrieval_latency_ms",
                    "reranking_latency_ms",
                    "gemini_latency_ms",
                ]
                if column in frame.columns
            ]
            if numeric and frame["timestamp"].notna().any():
                chart = frame.dropna(subset=["timestamp"]).set_index("timestamp")[numeric].copy()
                chart = chart.rename(columns=localize_columns(ui_lang, list(chart.columns)))
                st.line_chart(chart, height=330)
            left, right = st.columns(2)
            with left:
                if "query_type" in frame:
                    query_types = frame["query_type"].map(
                        lambda value: localize_value(ui_lang, value)
                    )
                    st.bar_chart(query_types.value_counts(), height=280)
            with right:
                st.json(drift_report(rows, kp.cfg["monitoring"].get("drift_window", 50)))
        else:
            st.info(tr(ui_lang, "no_telemetry"))

    with evaluation_tab:
        sample_count = st.slider(tr(ui_lang, "evaluation_questions"), 50, 500, 300, step=50)
        if st.button(tr(ui_lang, "run_retrieval_benchmark"), type="primary"):
            from tkip.evaluation import generate_eval_dataset, run_retrieval_benchmark

            with st.spinner(tr(ui_lang, "running_retrieval_benchmark")):
                samples = generate_eval_dataset(kp.chunks, sample_count)
                frame = run_retrieval_benchmark(kp.retriever, samples)
            display = frame.rename(columns=localize_columns(ui_lang, list(frame.columns)))
            st.dataframe(display, use_container_width=True)
            st.bar_chart(frame.select_dtypes("number").mean().to_frame("mean"), height=320)


def _render_architecture_contracts(ui_lang: str) -> None:
    cards = [
        (
            pick(ui_lang, "1. Tudásréteg felépítése", "1. Build the knowledge layer"),
            pick(
                ui_lang,
                "A dokumentumok parsing, minőségellenőrzés, chunking és embedding után külön BM25- és vektoros indexekbe kerülnek. A több indexváltozat miatt a retrieval-kísérletek reprodukálhatók.",
                "Documents pass through parsing, quality checks, chunking and embeddings before entering BM25 and vector indexes. Multiple index variants keep retrieval experiments reproducible.",
            ),
            "#2dd4bf",
        ),
        (
            pick(ui_lang, "2. Kérdésből keresési terv", "2. Turn a question into a search plan"),
            pick(
                ui_lang,
                "A guardrails és prompttervezés után a router dönti el, hogyan fusson a keresés. A BM25 és dense retrieval párhuzamosan dolgozik, majd RRF és reranker állítja elő a végső evidencialistát.",
                "After guardrails and prompt engineering, the router selects the search path. BM25 and dense retrieval run in parallel, then RRF and reranking produce the final evidence list.",
            ),
            "#60a5fa",
        ),
        (
            pick(ui_lang, "3. Kontrollált modellvégrehajtás", "3. Controlled model execution"),
            pick(
                ui_lang,
                "A Context Builder a legjobb bizonyítékokból tokenkeretes kontextust épít. Az engedélyezett Tool Node csak szükség esetén fut, a Gemini válasza pedig Pydantic séma és hivatkozás-validáció után kerül ki.",
                "The Context Builder creates a budgeted context from the strongest evidence. Allowlisted tools run only when needed, and Gemini output passes Pydantic schema and citation validation before delivery.",
            ),
            "#f59e0b",
        ),
        (
            pick(ui_lang, "4. Mérés → review → fejlesztés", "4. Measure → review → improve"),
            pick(
                ui_lang,
                "Minden futásból latency-, token-, költség- és minőségmetrikák készülnek. A gyenge vagy elutasított válasz regressziós mintává válhat, így a következő prompt-, retrieval- vagy indexverzió ugyanazon teszteken újramérhető.",
                "Each run produces latency, token, cost and quality metrics. Weak or rejected answers can become regression samples, so the next prompt, retrieval or index version can be measured against the same tests.",
            ),
            "#4ade80",
        ),
    ]
    cols = st.columns(4)
    for col, (title, body, accent) in zip(cols, cards):
        col.markdown(
            f'<div class="evidence-card" style="min-height:184px;border-top:3px solid {accent};background:rgba(15,23,42,.08)"><div class="evidence-title">{title}</div><div class="evidence-text">{body}</div></div>',
            unsafe_allow_html=True,
        )


def render_workflow_page(kp, ui_lang: str) -> None:
    st.markdown(
        f'<div style="margin-bottom:.65rem"><div class="section-kicker">{pick(ui_lang,"RENDSZERTERVEZÉS","SYSTEM DESIGN")}</div><div class="section-title" style="font-size:1.55rem">{pick(ui_lang,"Munkafolyamat és architektúra","Workflow & architecture")}</div><div class="small-muted">{pick(ui_lang,"Interaktív orchestration graph: indexelés, routing, párhuzamos retrieval, generálás, review és visszacsatolás egyetlen nézetben.","Interactive orchestration graph: indexing, routing, parallel retrieval, generation, review and feedback in one view.")}</div></div>',
        unsafe_allow_html=True,
    )

    render_orchestration_graph(kp, ui_lang)
    _render_architecture_contracts(ui_lang)

    left, right = st.columns([1.15, .85])
    with left:
        with st.expander(pick(ui_lang, "Csomópontok részletes útvonala", "Detailed node route"), expanded=False):
            rows = pd.DataFrame(workflow_rows("full", language=ui_lang))
            rows = rows.rename(columns={
                "number": "#",
                "label": pick(ui_lang, "Lépés", "Step"),
                "detail": pick(ui_lang, "Felelősség", "Responsibility"),
                "group": pick(ui_lang, "Fázis", "Phase"),
            })
            st.dataframe(rows, use_container_width=True, hide_index=True, height=520)
    with right:
        with st.expander(pick(ui_lang, "Runtime konfiguráció · JSON", "Runtime configuration · JSON"), expanded=True):
            st.code(
                """{
  \"orchestration\": \"native_tkip_core\",
  \"visual_model\": \"plotly_workflow_graph\",
  \"routing\": {\"type\": \"branch\", \"parallel_retrieval\": true},
  \"retrieval\": {
    \"parallel\": [\"bm25\", \"dense\"],
    \"fusion\": \"rrf\",
    \"reranker\": true
  },
  \"context\": {\"deduplicate\": true, \"diversity\": true, \"budgeted\": true},
  \"tools\": {\"allowlist\": true, \"max_steps\": 2},
  \"generation\": {\"provider\": \"gemini\", \"structured_output\": \"pydantic\"},
  \"review\": {\"human_in_the_loop\": true, \"feedback_to_regression\": true}
}""",
                language="json",
            )
