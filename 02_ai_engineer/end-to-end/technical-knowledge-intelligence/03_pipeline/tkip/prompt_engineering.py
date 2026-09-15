from __future__ import annotations

from typing import Any

# Advanced prompt profiles are deliberately explicit templates. They transform the
# user's task into a better specified instruction without adding factual content.
# `none` is a control condition; the 16 named profiles below are the experiment set.
PROFILES: dict[str, dict[str, str]] = {
    "none": {
        "label": "No optimization",
        "category": "Control",
        "purpose": "Use the corrected user question as-is.",
        "template": "{question}",
    },
    "rag_grounded": {
        "label": "01 · Grounded RAG — Recommended",
        "category": "Forrásalapúság",
        "purpose": "Maximize evidence quality, citation discipline and coherent synthesis.",
        "template": (
            "GOAL\nAnswer the question from the indexed technical library as a coherent expert explanation.\n\n"
            "QUESTION\n{question}\n\n"
            "GROUNDING RULES\nUse retrieved evidence for document-specific claims. Synthesize; do not dump raw chunks, tables of contents, "
            "or broken sentence fragments. Prefer complete explanations with a clear beginning, middle and conclusion. "
            "Cite material claims and explicitly abstain when evidence is insufficient.\n\n"
            "OUTPUT\nDirect answer → core mechanism → evidence-backed details → practical takeaway → sources/caveats."
        ),
    },
    "concise_expert": {
        "label": "02 · Concise Expert",
        "category": "Efficiency",
        "purpose": "Produce a compact senior-level answer with minimal token waste.",
        "template": (
            "Act as a concise senior technical expert. Question: {question}\n"
            "Answer directly in 4–7 focused paragraphs or a compact structured list. Preserve technical precision, cite grounded claims, "
            "avoid background that is not needed, and finish with one practical takeaway."
        ),
    },
    "technical_deep_dive": {
        "label": "03 · Technical Deep Dive",
        "category": "Depth",
        "purpose": "Turn a short question into a rigorous engineering investigation.",
        "template": (
            "Investigate this technical question deeply: {question}\n"
            "Cover where relevant: intuition; formal mechanism/math; architecture; implementation/API pattern; failure modes; trade-offs; "
            "production implications; and meaningful differences between retrieved sources. Ground book-specific claims and mark inference explicitly."
        ),
    },
    "structured_tutor": {
        "label": "04 · Structured Tutor",
        "category": "Learning",
        "purpose": "Teach progressively from intuition to implementation.",
        "template": (
            "Teach me this topic from my technical library: {question}\n"
            "Sequence: prerequisites → intuition → core concepts → mathematics → minimal example → implementation → common mistakes → "
            "self-check questions → recommended reading from retrieved sources. Do not invent book references."
        ),
    },
    "socratic_tutor": {
        "label": "05 · Socratic Tutor",
        "category": "Learning",
        "purpose": "Explain while surfacing assumptions and diagnostic questions.",
        "template": (
            "Use a Socratic teaching style for: {question}\n"
            "First give the minimum explanation needed to orient me, then identify 3–5 questions an engineer should be able to answer, "
            "walk through those questions using retrieved evidence, and end with a short knowledge check."
        ),
    },
    "comparison_matrix": {
        "label": "06 · Source Comparison Matrix",
        "category": "Comparison",
        "purpose": "Compare several sources without collapsing their differences.",
        "template": (
            "Compare the retrieved sources for this question: {question}\n"
            "Identify shared claims, differences in terminology/emphasis, implementation differences, trade-offs, and disagreements. "
            "Use a compact comparison table when useful, then provide a justified synthesis."
        ),
    },
    "code_first": {
        "label": "07 · Code-First Engineer",
        "category": "Implementation",
        "purpose": "Prioritize implementation patterns and code evidence.",
        "template": (
            "Solve this as an implementation-focused engineer: {question}\n"
            "Prioritize relevant APIs, code patterns, data flow, edge cases, and minimal runnable examples when supported by evidence. "
            "Explain why the pattern works and cite the exact source context for important implementation claims."
        ),
    },
    "debug_root_cause": {
        "label": "08 · Debug / Root-Cause Analysis",
        "category": "Engineering",
        "purpose": "Structure troubleshooting around hypotheses, evidence and verification.",
        "template": (
            "Treat this as a root-cause investigation: {question}\n"
            "Structure the answer as symptoms → likely hypotheses → evidence from the library → diagnostic checks → probable root cause(s) → "
            "fixes → prevention. Distinguish confirmed evidence from hypotheses."
        ),
    },
    "system_architecture": {
        "label": "09 · System Architecture Review",
        "category": "Architecture",
        "purpose": "Explain components, interfaces, data flow and trade-offs.",
        "template": (
            "Analyze this as a system architecture question: {question}\n"
            "Explain components, responsibilities, interfaces, data/control flow, state, failure boundaries, scalability concerns and design trade-offs. "
            "Include a grounded architecture diagram when the evidence supports it."
        ),
    },
    "production_readiness": {
        "label": "10 · Production Readiness / SRE",
        "category": "Éles üzem",
        "purpose": "Focus on deployability, reliability, observability and operations.",
        "template": (
            "Evaluate this from a production-readiness perspective: {question}\n"
            "Cover reliability, latency, cost, observability, testing, failure handling, security/guardrails, rollback/recovery, and operational trade-offs. "
            "Ground recommendations in retrieved material and label engineering inference."
        ),
    },
    "mathematical_derivation": {
        "label": "11 · Mathematical Derivation",
        "category": "Mathematics",
        "purpose": "Make assumptions, notation and derivation explicit.",
        "template": (
            "Explain and derive the mathematics behind: {question}\n"
            "Define notation and assumptions first, derive the key relationships step by step at an appropriate level, connect equations to intuition, "
            "then show a small numerical or implementation example when supported."
        ),
    },
    "research_synthesis": {
        "label": "12 · Multi-Source Research Synthesis",
        "category": "Research",
        "purpose": "Build a rigorous synthesis from multiple documents.",
        "template": (
            "Create a multi-source technical synthesis for: {question}\n"
            "Organize evidence by claim rather than by document. For each major claim, triangulate across sources where possible, preserve disagreements, "
            "identify evidence gaps, and finish with a concise synthesis and further reading."
        ),
    },
    "decision_tradeoff": {
        "label": "13 · Decision / Trade-off Analysis",
        "category": "Decision",
        "purpose": "Support engineering decisions with explicit criteria and trade-offs.",
        "template": (
            "Support an engineering decision about: {question}\n"
            "Identify decision criteria, viable options, advantages, disadvantages, risks, constraints and when each option is appropriate. "
            "Use an evidence-backed decision matrix if useful, then give a conditional recommendation rather than an absolute one."
        ),
    },
    "costar": {
        "label": "14 · CO-STAR",
        "category": "Keretrendszer",
        "purpose": "Make context, objective, style, tone, audience and response format explicit.",
        "template": (
            "Context: I am querying a private technical knowledge base containing programming, ML, AI and data-engineering books.\n"
            "Objective: {question}\n"
            "Style: precise, technically rigorous, explanatory.\n"
            "Tone: professional; do not oversimplify technical terms.\n"
            "Audience: an engineer developing toward AI/ML engineering.\n"
            "Response: coherent grounded synthesis with citations, examples when supported, and explicit uncertainty."
        ),
    },
    "crispe": {
        "label": "15 · CRISPE",
        "category": "Keretrendszer",
        "purpose": "Assign role, context, task, constraints and expected output.",
        "template": (
            "Capacity/Role: Act as a senior AI/ML technical tutor working over my indexed technical library.\n"
            "Insight/Context: Retrieved evidence is authoritative for book-specific claims.\n"
            "Statement/Task: {question}\n"
            "Personality/Style: rigorous, engineering-oriented, clear.\n"
            "Experiment/Constraints: distinguish evidence from inference; cite exact sources; do not fabricate missing details.\n"
            "Expected output: direct answer, explanation, practical example, caveats, and recommended source sections."
        ),
    },
    "evidence_verification": {
        "label": "16 · Evidence Verification",
        "category": "Quality",
        "purpose": "Favor claim-evidence alignment and explicit verification over breadth.",
        "template": (
            "Answer this question with verification discipline: {question}\n"
            "For every important claim, ensure the retrieved evidence actually supports it. Prefer fewer well-supported claims over broad speculation. "
            "Flag conflicting evidence, weak support, or missing evidence. Finish with a short 'What is verified vs inferred' summary."
        ),
    },
}


PROFILE_HU: dict[str, dict[str, str]] = {
    "none": {
        "label": "Nincs optimalizálás",
        "category": "Kontroll",
        "purpose": "A javított kérdés változtatás nélküli használata.",
        "template": "{question}",
    },
    "rag_grounded": {
        "label": "01 · Forrásalapú RAG — ajánlott",
        "category": "Forrásalapúság",
        "purpose": "Bizonyítékalapú, jól hivatkozott, koherens válasz.",
        "template": "CÉL\nVálaszold meg a kérdést az indexelt technikai könyvtár alapján, koherens szakértői magyarázatként.\n\nKÉRDÉS\n{question}\n\nGROUNDING SZABÁLYOK\nA dokumentumspecifikus állításokat a visszakeresett bizonyítékokra alapozd. Ne másolj nyers darabokat vagy töredékes szöveget. A fontos állításokat hivatkozd, elégtelen bizonyíték esetén jelezd a bizonytalanságot.\n\nKIMENET\nKözvetlen válasz → működés/mechanizmus → bizonyítékalapú részletek → gyakorlati következtetés → források és korlátok.",
    },
    "concise_expert": {
        "label": "02 · Tömör szakértő",
        "category": "Hatékonyság",
        "purpose": "Rövid, tapasztalt szakértői szintű, technikailag pontos válasz.",
        "template": "Dolgozz tömör, tapasztalt technikai szakértőként. Kérdés: {question}\nVálaszolj közvetlenül 4–7 fókuszált bekezdésben vagy kompakt strukturált listában. Tartsd meg a technikai pontosságot, hivatkozd a forrásalapú állításokat, és zárd egy gyakorlati következtetéssel.",
    },
    "technical_deep_dive": {
        "label": "03 · Technikai mélyelemzés",
        "category": "Mélység",
        "purpose": "Részletes mérnöki vizsgálat kompromisszumokkal és hibamódokkal.",
        "template": "Vizsgáld meg mélyen ezt a technikai kérdést: {question}\nAhol releváns, térj ki az intuícióra, formális működésre/matematikára, architektúrára, implementációra, hibamódokra, kompromisszumokra és éles üzemi következményekre. A könyvspecifikus állításokat alapozd bizonyítékra, a következtetéseket jelöld külön.",
    },
    "structured_tutor": {
        "label": "04 · Strukturált oktató",
        "category": "Tanulás",
        "purpose": "Fokozatos tanítás intuíciótól implementációig.",
        "template": "Tanítsd meg ezt a témát a technikai könyvtár alapján: {question}\nSorrend: előfeltételek → intuíció → alapfogalmak → matematika → minimális példa → implementáció → gyakori hibák → önellenőrző kérdések → releváns források. Ne találj ki könyvhivatkozást.",
    },
    "socratic_tutor": {
        "label": "05 · Szókratészi oktató",
        "category": "Tanulás",
        "purpose": "Magyarázat diagnosztikus kérdésekkel és önellenőrzéssel.",
        "template": "Használj szókratészi tanítási stílust ehhez: {question}\nElőször adj rövid orientáló magyarázatot, majd fogalmazz meg 3–5 mérnöki ellenőrző kérdést, válaszold meg őket a visszakeresett bizonyítékokkal, és zárd rövid tudásellenőrzéssel.",
    },
    "comparison_matrix": {
        "label": "06 · Forrás-összehasonlítás",
        "category": "Összehasonlítás",
        "purpose": "Források közös pontjainak és eltéréseinek összevetése.",
        "template": "Hasonlítsd össze a visszakeresett forrásokat ehhez a kérdéshez: {question}\nAzonosítsd a közös állításokat, terminológiai és hangsúlybeli eltéréseket, implementációs különbségeket, kompromisszumokat és esetleges ellentmondásokat. Ha hasznos, használj kompakt összehasonlító táblát, majd adj indokolt szintézist.",
    },
    "code_first": {
        "label": "07 · Kódközpontú mérnök",
        "category": "Implementáció",
        "purpose": "API-k, kódminták és implementációs részletek előtérben.",
        "template": "Oldd meg implementációközpontú mérnökként: {question}\nHelyezd előtérbe a releváns API-kat, kódmintákat, adatfolyamot, szélső eseteket és minimális futtatható példákat, ha a források támogatják. Magyarázd el, miért működik a minta, és hivatkozd a fontos implementációs állításokat.",
    },
    "debug_root_cause": {
        "label": "08 · Hibakeresés / gyökérok",
        "category": "Mérnöki",
        "purpose": "Hipotézis → bizonyíték → diagnosztika → javítás struktúra.",
        "template": "Kezeld gyökérok-elemzésként: {question}\nStruktúra: tünetek → valószínű hipotézisek → könyvtári bizonyíték → diagnosztikai ellenőrzések → valószínű gyökérok(ok) → javítás → megelőzés. Válaszd szét a bizonyított tényeket és a hipotéziseket.",
    },
    "system_architecture": {
        "label": "09 · Rendszerarchitektúra",
        "category": "Architektúra",
        "purpose": "Komponensek, interfészek, adatfolyam és kompromisszumok.",
        "template": "Elemezd rendszerarchitektúra-kérdésként: {question}\nMutasd be a komponenseket, felelősségeket, interfészeket, adat- és vezérlési folyamatot, állapotot, hibahatárokat, skálázási szempontokat és tervezési kompromisszumokat. Ha a bizonyíték támogatja, adj forrásalapú architektúradiagramot.",
    },
    "production_readiness": {
        "label": "10 · Éles üzemre készség / SRE",
        "category": "Éles üzem",
        "purpose": "Megbízhatóság, megfigyelhetőség, költség és üzemeltetés.",
        "template": "Értékeld éles üzemre készségi nézőpontból: {question}\nTérj ki a megbízhatóságra, késleltetésre, költségre, megfigyelhetőségre, tesztelésre, hibakezelésre, biztonsági és védelmi szempontokra, visszaállítási és helyreállítási megoldásokra és üzemeltetési kompromisszumokra. A javaslatokat alapozd a forrásokra, a következtetést jelöld.",
    },
    "mathematical_derivation": {
        "label": "11 · Matematikai levezetés",
        "category": "Matematika",
        "purpose": "Feltételezések, jelölések és lépésenkénti levezetés.",
        "template": "Magyarázd el és vezesd le a matematika lényegét ehhez: {question}\nElőször definiáld a jelöléseket és feltételezéseket, majd vezesd le lépésenként a kulcsösszefüggéseket, kapcsold őket intuícióhoz, végül adj kis numerikus vagy implementációs példát, ha támogatott.",
    },
    "research_synthesis": {
        "label": "12 · Többforrású kutatási szintézis",
        "category": "Kutatás",
        "purpose": "Több dokumentumból állításközpontú, rigorózus szintézis.",
        "template": "Készíts többforrású technikai szintézist ehhez: {question}\nAz anyagot állítások szerint szervezd, ne dokumentumonként. A fő állításokat lehetőség szerint több forrásból trianguláld, őrizd meg az eltéréseket, jelöld a bizonyítékhiányokat, és zárd tömör szintézissel és további olvasnivalóval.",
    },
    "decision_tradeoff": {
        "label": "13 · Döntési és kompromisszumelemzés",
        "category": "Döntés",
        "purpose": "Opciók és mérnöki döntési kritériumok összevetése.",
        "template": "Támogass mérnöki döntést ebben: {question}\nAzonosítsd a döntési kritériumokat, opciókat, előnyöket, hátrányokat, kockázatokat, korlátokat és alkalmazási feltételeket. Ha hasznos, készíts bizonyítékalapú döntési mátrixot, majd adj feltételes ajánlást.",
    },
    "costar": {
        "label": "14 · CO-STAR",
        "category": "Keretrendszer",
        "purpose": "Kontextus, cél, stílus, hangnem, célközönség és válaszforma explicit megadása.",
        "template": "Kontextus: privát technikai tudásbázist kérdezek programozás, ML, AI és adatmérnökségi témákban.\nCél: {question}\nStílus: pontos, technikailag rigorózus, magyarázó.\nHangnem: professzionális, a technikai fogalmakat ne egyszerűsítsd túl.\nKözönség: AI/ML-mérnöki irányba fejlődő mérnök.\nVálasz: koherens forrásalapú szintézis hivatkozásokkal, támogatott példákkal és explicit bizonytalansággal.",
    },
    "crispe": {
        "label": "15 · CRISPE",
        "category": "Keretrendszer",
        "purpose": "Szerep, kontextus, feladat, korlátok és elvárt kimenet rögzítése.",
        "template": "Szerep: tapasztalt AI/ML technikai oktató az indexelt könyvtár felett.\nKontextus: a visszakeresett bizonyíték mérvadó a könyvspecifikus állításokhoz.\nFeladat: {question}\nStílus: rigorózus, mérnöki, világos.\nKorlátok: válaszd szét a bizonyítékot és a következtetést, hivatkozz pontosan, ne találj ki hiányzó részleteket.\nElvárt kimenet: közvetlen válasz, magyarázat, gyakorlati példa, korlátok és ajánlott forrásszakaszok.",
    },
    "evidence_verification": {
        "label": "16 · Bizonyítékellenőrzés",
        "category": "Minőség",
        "purpose": "Állítás–bizonyíték megfelelés és explicit bizonytalanság.",
        "template": "Válaszolj bizonyítékellenőrzési fegyelemmel: {question}\nMinden fontos állításnál ellenőrizd, hogy a visszakeresett bizonyíték valóban támogatja-e. Inkább kevesebb jól alátámasztott állítást adj, mint széles spekulációt. Jelöld az ellentmondást, gyenge támogatást és hiányzó bizonyítékot. Zárd rövid 'bizonyított vs. következtetett' összefoglalóval.",
    },
}

# Exactly sixteen named experimental profiles, excluding the no-optimization control.
PROFILE_KEYS = [k for k in PROFILES if k != "none"]


def local_optimize(question: str, profile: str, language: str = "en") -> dict[str, Any]:
    """Apply a deterministic prompt profile in the requested language."""
    source = PROFILE_HU if language == "hu" else PROFILES
    profile_data = source.get(profile, source["none"])
    optimized = profile_data["template"].format(question=question.strip())
    improvements = []
    if profile != "none":
        improvements = (
            [
                "A cél és az elvárt kimenet pontosítása",
                "Bizonyíték- és forrásalapúsági szabályok hozzáadása",
                "Profil-specifikus struktúra és korlátok hozzáadása",
            ]
            if language == "hu"
            else [
                "Clarified the objective and expected output",
                "Added evidence/grounding discipline",
                "Added profile-specific structure and constraints",
            ]
        )
    return {
        "profile": profile,
        "profile_label": profile_data["label"],
        "category": profile_data.get("category", "General"),
        "purpose": profile_data["purpose"],
        "original": question,
        "optimized": optimized,
        "changed": optimized.strip() != question.strip(),
        "improvements": improvements,
        "reviewer": "local-template",
    }
