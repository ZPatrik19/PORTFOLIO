from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WorkflowKind = Literal["full", "query", "indexing"]


@dataclass(frozen=True)
class WorkflowNode:
    number: int
    key: str
    label: str
    detail: str
    group: str


FULL_WORKFLOW = [
    WorkflowNode(1, "sources", "Data sources", "Private books + public technical docs", "Indexing"),
    WorkflowNode(2, "discovery", "File discovery & versioning", "Stable document_id + checksum/version detection", "Indexing"),
    WorkflowNode(3, "parse", "Document parsing", "PDF/DOCX/MD/HTML structure extraction", "Indexing"),
    WorkflowNode(4, "quality", "Data quality validation", "Failures, empty pages, duplicates, size distributions", "Indexing"),
    WorkflowNode(5, "chunk", "Chunking strategy", "fixed / recursive / structure-aware / semantic", "Indexing"),
    WorkflowNode(6, "embed", "Embedding generation", "Provider abstraction + cache + batching", "Indexing"),
    WorkflowNode(7, "multi_index", "Chunk-specific indexes", "Independent vector/BM25 index for each experiment", "Indexing"),
    WorkflowNode(8, "ui", "Streamlit research workspace", "Question, Gemini key, presets and A/B playground", "Query"),
    WorkflowNode(9, "fastapi", "FastAPI service boundary", "Typed request/response models and external API", "Query"),
    WorkflowNode(10, "trace", "Request + trace middleware", "request_id, trace_id and telemetry", "Query"),
    WorkflowNode(11, "guard", "Input guardrails", "Injection/exfiltration and request validation", "Query"),
    WorkflowNode(12, "promptclean", "Language / clarity check", "Grammar cleanup without changing technical intent", "Query"),
    WorkflowNode(13, "promptopt", "Advanced prompt optimizer", "16 selectable prompt-engineering profiles", "Query"),
    WorkflowNode(14, "understand", "Query understanding", "Intent, language, topic/context needs", "Query"),
    WorkflowNode(15, "strategy", "Chunk-index selection", "Compact / balanced / semantic persistent index", "Query"),
    WorkflowNode(16, "retrieve", "BM25 + Dense retrieval", "Lexical and semantic candidate generation", "Query"),
    WorkflowNode(17, "fusion", "Hybrid fusion", "Reciprocal Rank Fusion", "Query"),
    WorkflowNode(18, "rerank", "Reranking", "Re-score strongest hybrid candidates", "Query"),
    WorkflowNode(19, "context", "Context engineering", "Dedup, diversity, re-chunking and token budget", "Query"),
    WorkflowNode(20, "tools", "Tool calling", "Gemini selects allowlisted backend functions", "Query"),
    WorkflowNode(21, "gemini", "Gemini grounded synthesis", "Coherent structured answer + diagram", "Query"),
    WorkflowNode(22, "citation", "Citation & output validation", "Resolve citations and attach trusted source visual", "Query"),
    WorkflowNode(23, "response", "Answer + inspectors", "Sources, images, tools, ranking, prompt and QA", "Query"),
    WorkflowNode(24, "telemetry", "Tracing & AI monitoring", "Latency, token, cost, quality and retrieval metrics", "Quality"),
    WorkflowNode(25, "feedback", "Feedback & failure analysis", "Negative feedback → regression sample", "Quality"),
    WorkflowNode(26, "eval", "Offline evaluation", "Retrieval, generation, tool and no-answer benchmarks", "Quality"),
    WorkflowNode(27, "improve", "Improve & re-evaluate", "Építés → mérés → elemzés → fejlesztés → újramérés", "Quality"),
]


HU_WORKFLOW_COPY = {
    "sources": ("Adatforrások", "Privát könyvek + nyilvános technikai dokumentumok", "Indexelés"),
    "discovery": ("Fájlfelderítés és verziózás", "Stabil dokumentumazonosító + ellenőrzőösszeg/verziókövetés", "Indexelés"),
    "parse": ("Dokumentumfeldolgozás", "PDF/DOCX/MD/HTML struktúra kinyerése", "Indexelés"),
    "quality": ("Adatminőség-ellenőrzés", "Hibák, üres oldalak, duplikátumok és méreteloszlások", "Indexelés"),
    "chunk": ("Darabolási stratégia", "fix / rekurzív / szerkezetérzékeny / szemantikus", "Indexelés"),
    "embed": ("Beágyazások létrehozása", "Szolgáltatófüggetlen réteg + gyorsítótár + kötegelt feldolgozás", "Indexelés"),
    "multi_index": ("Darabolásspecifikus indexek", "Külön vektoros/BM25 index minden kísérlethez", "Indexelés"),
    "ui": ("Streamlit kutatási felület", "Kérdés, Gemini-kulcs, profilok és A/B teszt", "Lekérdezés"),
    "fastapi": ("FastAPI szolgáltatási határ", "Típusos kérés-/válaszmodellek és külső API", "Lekérdezés"),
    "trace": ("Kérés- és nyomkövetési köztesréteg", "kérésazonosító, nyomkövetési azonosító és telemetria", "Lekérdezés"),
    "guard": ("Bemeneti védelmi szabályok", "Promptinjekció, adatkiszivárogtatás és bemeneti validáció", "Lekérdezés"),
    "promptclean": ("Nyelvi és érthetőségi ellenőrzés", "Nyelvi javítás a technikai szándék megőrzésével", "Lekérdezés"),
    "promptopt": ("Haladó promptoptimalizálás", "16 választható prompttervezési profil", "Lekérdezés"),
    "understand": ("Kérdésértelmezés", "Szándék, nyelv, témák és kontextusigény", "Lekérdezés"),
    "strategy": ("Darabolási index kiválasztása", "Kompakt / kiegyensúlyozott / szemantikus index", "Lekérdezés"),
    "retrieve": ("BM25 + szemantikus visszakeresés", "Lexikális és szemantikus jelöltek", "Lekérdezés"),
    "fusion": ("Hibrid rangsoregyesítés", "Reciprocal Rank Fusion (RRF)", "Lekérdezés"),
    "rerank": ("Újrarangsorolás", "A legerősebb hibrid jelöltek újrapontozása", "Lekérdezés"),
    "context": ("Kontextustervezés", "Deduplikáció, diverzitás, újradarabolás és tokenkeret", "Lekérdezés"),
    "tools": ("Eszközhívás", "A Gemini által választott engedélyezett backend-funkciók", "Lekérdezés"),
    "gemini": ("Gemini forrásalapú szintézis", "Koherens, strukturált válasz + diagram", "Lekérdezés"),
    "citation": ("Hivatkozás- és kimenetvalidáció", "Hivatkozások feloldása és validált forrásvizuál", "Lekérdezés"),
    "response": ("Válasz és diagnosztika", "Források, képek, eszközök, rangsor, prompt és minőségellenőrzés", "Lekérdezés"),
    "telemetry": ("Nyomkövetés és AI-megfigyelés", "Késleltetés, token, költség, minőség és visszakeresési metrikák", "Minőség"),
    "feedback": ("Visszajelzés és hibaanalízis", "Negatív visszajelzés → regressziós minta", "Minőség"),
    "eval": ("Offline kiértékelés", "Visszakeresési, generálási, eszközhívási és nincs-válasz benchmarkok", "Minőség"),
    "improve": ("Fejlesztés és újramérés", "Építés → mérés → elemzés → fejlesztés → újramérés", "Minőség"),
}


def _localized_node(node: WorkflowNode, language: str) -> WorkflowNode:
    if language != "hu":
        return node
    label, detail, _ = HU_WORKFLOW_COPY.get(node.key, (node.label, node.detail, node.group))
    # Keep the internal group identifier stable for graph palettes and filtering.
    return WorkflowNode(node.number, node.key, label, detail, node.group)


def _subset(kind: WorkflowKind) -> list[WorkflowNode]:
    if kind == "indexing":
        return [n for n in FULL_WORKFLOW if n.group == "Indexing"]
    if kind == "query":
        return [n for n in FULL_WORKFLOW if n.group == "Query"]
    return FULL_WORKFLOW


def workflow_rows(kind: WorkflowKind = "full", language: str = "en") -> list[dict]:
    rows = []
    for node in _subset(kind):
        localized = _localized_node(node, language).__dict__.copy()
        if language == "hu":
            localized["group"] = {"Indexing": "Indexelés", "Query": "Lekérdezés", "Quality": "Minőség"}.get(node.group, node.group)
        rows.append(localized)
    return rows


def _node_line(
    key: str,
    number: int,
    label: str,
    detail: str,
    fill: str,
    border: str,
    status: str | None = None,
    language: str = "en",
) -> str:
    suffix = ""
    if status:
        status_label = status.upper()
        if language == "hu":
            status_label = {
                "success": "SIKERES",
                "warning": "FIGYELMEZTETÉS",
                "blocked": "BLOKKOLVA",
                "fallback": "TARTALÉK MÓD",
                "skipped": "KIHAGYVA",
            }.get(status, status_label)
        suffix = f"\\n[{status_label}]"
    text = f"{number:02d}  {label}{suffix}".replace('"', "'")
    pen = "2.4" if status in {"success", "warning", "blocked", "fallback"} else "1.25"
    status_border = {
        "success": "#2D9D68",
        "warning": "#D79A24",
        "blocked": "#D84F57",
        "fallback": "#7A5DE8",
        "skipped": "#9A9A9A",
    }.get(status or "", border)
    return f'{key} [label="{text}", fillcolor="{fill}", color="{status_border}", penwidth={pen}];'


def workflow_dot(kind: WorkflowKind = "full", pipeline_steps: list[dict] | None = None, language: str = "en") -> str:
    """KNIME-style workflow graph with parallel branches and readable step numbers."""
    nodes = {n.key: _localized_node(n, language) for n in _subset(kind)}
    if not nodes:
        return "digraph G {}"

    status_by_label = {str(x.get("stage")): str(x.get("status", "success")) for x in (pipeline_steps or [])}
    stage_key = {
        "Input guardrails": "guard", "Knowledge index": "strategy", "Index strategy": "strategy",
        "Prompt language check": "promptclean", "Advanced prompt engineering": "promptopt",
        "Query understanding": "understand", "Topic & context analysis": "understand",
        "Hybrid retrieval": "retrieve", "Metadata filters": "retrieve", "Reranker": "rerank",
        "Document diversity": "context", "Dynamic chunking": "context", "Context engineering": "context",
        "Tool calling": "tools", "Context refresh": "context", "Gemini generation": "gemini",
        "Citation validator": "citation", "Source visual": "citation", "Prompt & output QA": "citation",
        "Response ready": "response",
    }
    status_by_key: dict[str, str] = {}
    for label, status in status_by_label.items():
        key = stage_key.get(label)
        if key:
            # Keep the most severe/meaningful final state.
            status_by_key[key] = status

    palettes = {
        "Indexing": ("#FFF4BF", "#B77900"),
        "Query": ("#E2F0FF", "#2466C2"),
        "Quality": ("#E3F8EC", "#1F8E57"),
    }
    lines = [
        "digraph TKI {",
        'rankdir=TB;',
        'graph [bgcolor="#FCFCFE", pad="0.24", nodesep="0.34", ranksep="0.64", splines="spline", compound=true, newrank=true];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontcolor="#111827", fontsize=10.0, margin="0.16,0.09", width=1.85, height=.62];',
        'edge [color="#667085", penwidth=1.35, arrowsize=.72];',
    ]

    # Full/query views intentionally look like a visual workflow canvas rather
    # than one long vertical diagram. Indexing and retrieval have branches.
    for n in nodes.values():
        if n.key == "retrieve" and kind in {"full", "query"}:
            # The visual canvas expands this logical step into parallel BM25 and
            # Dense nodes, so the aggregate node itself is not drawn.
            continue
        fill, border = palettes[n.group]
        lines.append(_node_line(n.key, n.number, n.label, n.detail, fill, border, status_by_key.get(n.key), language))

    compact_label = "5A  Kompakt / rekurzív\n~700 karakter · gyors" if language == "hu" else "5A  Compact / Recursive\n~700 chars · fast"
    balanced_label = "5B  Kiegyensúlyozott / szerkezetérzékeny\n~1000 karakter · ajánlott" if language == "hu" else "5B  Balanced / Structure-aware\n~1000 chars · recommended"
    semantic_label = "5C  Szemantikus / Mély\n~1500 karakter · koncepcionális" if language == "hu" else "5C  Semantic / Deep\n~1500 chars · conceptual"
    lexical_label = "16A  BM25\nlexikális" if language == "hu" else "16A  BM25\nlexical"
    semantic_retrieval_label = "16B  Szemantikus\nbeágyazásos" if language == "hu" else "16B  Dense\nsemantic"
    tool_search_label = "20A Könyvtárkeresés" if language == "hu" else "20A Search library"
    tool_code_label = "20B Kódpéldák" if language == "hu" else "20B Code examples"
    tool_meta_label = "20C Metaadat / oldal" if language == "hu" else "20C Metadata / page"
    tool_compare_label = "20D Összehasonlítás / tanulás" if language == "hu" else "20D Compare / learning"
    compact_dot = compact_label.replace("\n", "\\n")
    balanced_dot = balanced_label.replace("\n", "\\n")
    semantic_dot = semantic_label.replace("\n", "\\n")
    lexical_dot = lexical_label.replace("\n", "\\n")
    dense_dot = semantic_retrieval_label.replace("\n", "\\n")

    if kind in {"full", "indexing"}:
        for a, b in [("sources","discovery"),("discovery","parse"),("parse","quality"),("quality","chunk")]:
            if a in nodes and b in nodes: lines.append(f"{a} -> {b};")
        if "chunk" in nodes:
            # Visual fan-out for the three user-facing chunking presets.
            lines += [
                f'chunk_compact [label="{compact_dot}", fillcolor="#FFE9A3", color="#A66A00", shape=box, style="rounded,filled,dashed"];',
                f'chunk_balanced [label="{balanced_dot}", fillcolor="#FFE9A3", color="#A66A00", shape=box, style="rounded,filled,dashed"];',
                f'chunk_semantic [label="{semantic_dot}", fillcolor="#FFE9A3", color="#A66A00", shape=box, style="rounded,filled,dashed"];',
                'chunk -> chunk_compact; chunk -> chunk_balanced; chunk -> chunk_semantic;',
            ]
            if "embed" in nodes:
                lines += ['chunk_compact -> embed;', 'chunk_balanced -> embed;', 'chunk_semantic -> embed;']
        if "embed" in nodes and "multi_index" in nodes: lines.append("embed -> multi_index;")

    if kind in {"full", "query"}:
        chain = [("fastapi","trace"),("trace","guard"),("guard","promptclean"),("promptclean","promptopt"),("promptopt","understand"),("understand","strategy")]
        for a,b in chain:
            if a in nodes and b in nodes:
                lines.append(f"{a} -> {b};")
        if "ui" in nodes and "guard" in nodes:
            lines.append("ui -> guard;")
        if "strategy" in nodes and "retrieve" in nodes:
            lines += [
                f'bm25 [label="{lexical_dot}", fillcolor="#D8E9FF", color="#2259A8", shape=box, style="rounded,filled"];',
                f'dense [label="{dense_dot}", fillcolor="#D8E9FF", color="#2259A8", shape=box, style="rounded,filled"];',
                'strategy -> bm25; strategy -> dense;',
            ]
            if "fusion" in nodes:
                lines += ['bm25 -> fusion;', 'dense -> fusion;']
        for a,b in [("fusion","rerank"),("rerank","context"),("context","tools")]:
            if a in nodes and b in nodes: lines.append(f"{a} -> {b};")
        if "tools" in nodes:
            lines += [
                f'tool_library [label="{tool_search_label}", fillcolor="#EEE7FF", color="#6E51D8", fontsize=8.5];',
                f'tool_code [label="{tool_code_label}", fillcolor="#EEE7FF", color="#6E51D8", fontsize=8.5];',
                f'tool_meta [label="{tool_meta_label}", fillcolor="#EEE7FF", color="#6E51D8", fontsize=8.5];',
                f'tool_compare [label="{tool_compare_label}", fillcolor="#EEE7FF", color="#6E51D8", fontsize=8.5];',
                'tools -> tool_library [style=dashed]; tools -> tool_code [style=dashed]; tools -> tool_meta [style=dashed]; tools -> tool_compare [style=dashed];',
            ]
            if "gemini" in nodes:
                lines += ['tool_library -> gemini [style=dashed];','tool_code -> gemini [style=dashed];','tool_meta -> gemini [style=dashed];','tool_compare -> gemini [style=dashed];','tools -> gemini;']
        for a,b in [("gemini","citation"),("citation","response")]:
            if a in nodes and b in nodes: lines.append(f"{a} -> {b};")

    if kind == "full":
        if "multi_index" in nodes and "ui" in nodes:
            corpus_label = "kész korpusz" if language == "hu" else "ready corpus"
            lines.append(f'multi_index -> ui [color="#A3A3A3", label="{corpus_label}", fontsize=8];')
        for a,b in [("response","telemetry"),("telemetry","feedback"),("feedback","eval"),("eval","improve")]:
            if a in nodes and b in nodes: lines.append(f"{a} -> {b};")
        if "improve" in nodes and "chunk" in nodes:
            version_label = "új verzió" if language == "hu" else "new version"
            lines.append(f'improve -> chunk [style=dashed, color="#35A46F", label="{version_label}", fontsize=8];')

    # Arrange the graph like a workflow canvas with multiple horizontal rows
    # instead of one ultra-wide chain.
    row_specs = []
    if kind == "full":
        row_specs = [
            ["sources", "discovery", "parse", "quality", "chunk"],
            ["chunk_compact", "chunk_balanced", "chunk_semantic", "embed", "multi_index"],
            ["ui", "fastapi", "trace", "guard", "promptclean", "promptopt", "understand", "strategy"],
            ["bm25", "dense", "fusion", "rerank", "context", "tools", "gemini"],
            ["tool_library", "tool_code", "tool_meta", "tool_compare"],
            ["citation", "response", "telemetry", "feedback", "eval", "improve"],
        ]
    elif kind == "query":
        row_specs = [
            ["ui", "fastapi", "trace", "guard", "promptclean", "promptopt", "understand", "strategy"],
            ["bm25", "dense", "fusion", "rerank", "context", "tools", "gemini"],
            ["tool_library", "tool_code", "tool_meta", "tool_compare"],
            ["citation", "response"],
        ]
    else:
        row_specs = [
            ["sources", "discovery", "parse", "quality", "chunk"],
            ["chunk_compact", "chunk_balanced", "chunk_semantic", "embed", "multi_index"],
        ]
    for row in row_specs:
        lines.append("{ rank=same; " + "; ".join(row) + "; }")

    lines.append("}")
    return "\n".join(lines)
