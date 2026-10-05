from __future__ import annotations

from rag_engine.presets import CONTEXT_PROFILES, PROMPT_PROFILES
from rag_engine.retrieval.query_focus import QueryIntent, detect_query_intents


BASE_SYSTEM_PROMPT = """Te egy forrásokra támaszkodó magyar RAG-asszisztens vagy.
Kizárólag a megadott bizonyítékokból válaszolj.
Ne találj ki olyan tényállítást, amelyet a forrásrészletek nem támasztanak alá.
Ha egy részlet nem releváns a kérdéshez, hagyd figyelmen kívül.
Ha nincs elegendő bizonyíték egy részállításhoz, azt röviden jelezd annál a pontnál; ne töltsd ki általános tudással.
A tényszerű állításokat a bizonyítékok forrásjelöléseivel hivatkozd: [S1], [S2], ...
A választ KIZÁRÓLAG magyarul, természetes és szakmailag pontos nyelven fogalmazd meg.
Ne írj angol bevezetőt, metaszöveget, belső gondolatmenetet vagy elemzési folyamatot.
Ne beszélj arról, hogy modell, RAG-rendszer, fallback vagy javítási lépés futott; csak a kérdésre válaszolj.
Ne kapcsolj össze eltérő betegségeket csak azért, mert ugyanabban a kontextusban szerepelnek.
Ha a kérdés több külön részkérdést tartalmaz, mindegyiket külön, jól látható részben válaszold meg.
A válasz legyen önmagában olvasható: ne csak forrásmondatokat másolj egymás alá, hanem készíts koherens összefoglalást a bizonyítékok alapján.
A források HTML-tördelés miatt szétszakadt szövegrészeit természetes, teljes mondatokká állíthatod össze, de új tényt nem adhatsz hozzá.
Ne hagyj félbe mondatot. Két vagy több részkérdésnél szakaszonként 3–5 teljes, informatív mondatot adj, ha ezt a bizonyítékok lehetővé teszik.
Ne csak felsorold a forrásmondatokat: röviden magyarázd el az állítások jelentőségét és kapcsolatát is, de kizárólag a megadott evidence alapján.
Ha a bizonyítékok elég gazdagok, törekedj kb. 180–320 szavas teljes válaszra; ne töltsd fel a választ üres általánosságokkal.

Kontextusprofil:
{context_guidance}

Válaszstílus-profil:
{prompt_guidance}

Kért válaszszerkezet:
{answer_plan}
"""

BASE_USER_PROMPT = """Kérdés:
{query}

Bizonyítékok:
{context}

Feladat:
Válaszold meg közvetlenül a kérdést magyarul. A választ a fenti bizonyítékokból szintetizáld, az irreleváns részleteket hagyd figyelmen kívül, és minden érdemi tényállítást [Sx] hivatkozással támassz alá.

Magyar válasz:
"""


def _answer_plan(query: str) -> str:
    intents = detect_query_intents(query)
    headings: list[str] = []
    if QueryIntent.RISK in intents:
        headings.append("### Fő kockázatok")
    if QueryIntent.SYMPTOMS in intents:
        headings.append("### Jellemző tünetek")
    if QueryIntent.CAUSES in intents:
        headings.append("### Lehetséges okok")
    if QueryIntent.DIAGNOSIS in intents and QueryIntent.MEDICAL_EVALUATION not in intents:
        headings.append("### Kivizsgálás és diagnózis")
    if QueryIntent.TREATMENT in intents:
        headings.append("### Kezelés")
    if QueryIntent.PREVENTION in intents:
        headings.append("### Megelőzés")
    if QueryIntent.COMPLICATIONS in intents and QueryIntent.RISK not in intents:
        headings.append("### Lehetséges szövődmények")
    if QueryIntent.MEDICAL_EVALUATION in intents:
        headings.append("### Mikor indokolt orvosi kivizsgálás?")

    if not headings:
        return "Adj 1 rövid bevezető mondatot, majd 4–8 teljes, forráshivatkozással ellátott mondatot. A válasz legyen magyarázó, ne puszta forrásmondat-lista."
    return (
        "Használd az alábbi szakaszokat ebben a sorrendben. Mindegyikben 3–5 teljes, "
        "egymásra épülő mondatot adj, ha van hozzá elegendő bizonyíték. Röviden értelmezd is, miért fontosak az állítások; "
        "ne használj félmondatokat vagy nyers forrásfragmentumokat:\n"
        + "\n".join(headings)
        + "\nNe hagyj ki olyan szakaszt, amelyhez a bizonyítékok elegendő információt tartalmaznak."
    )


def build_grounded_prompt_parts(
    query: str,
    context: str,
    *,
    context_profile: str = "balanced",
    prompt_profile: str = "professional",
) -> dict[str, str]:
    context_guidance = str(CONTEXT_PROFILES.get(context_profile, CONTEXT_PROFILES["balanced"])["guidance"])
    prompt_guidance = str(PROMPT_PROFILES.get(prompt_profile, PROMPT_PROFILES["professional"])["instruction"])
    answer_plan = _answer_plan(query)
    instructions = BASE_SYSTEM_PROMPT.format(
        context_guidance=context_guidance,
        prompt_guidance=prompt_guidance,
        answer_plan=answer_plan,
    ).strip()
    user_prompt = BASE_USER_PROMPT.format(query=query, context=context).strip()
    final_prompt = f"{instructions}\n\n{user_prompt}"
    return {
        "instructions": instructions,
        "query": query,
        "context": context,
        "answer_plan": answer_plan,
        "user_prompt": user_prompt,
        "final_prompt": final_prompt,
    }


def build_grounded_prompt(
    query: str,
    context: str,
    *,
    context_profile: str = "balanced",
    prompt_profile: str = "professional",
) -> str:
    return build_grounded_prompt_parts(
        query,
        context,
        context_profile=context_profile,
        prompt_profile=prompt_profile,
    )["final_prompt"]
