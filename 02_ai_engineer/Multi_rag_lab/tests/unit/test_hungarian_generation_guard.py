from __future__ import annotations

from rag_engine.generation.fallbacks import build_evidence_fallback_answer
from rag_engine.generation.grounded import GroundedGenerator
from rag_engine.generation.validators import (
    answer_requires_hungarian_retry,
    answer_requires_quality_retry,
    looks_like_english_meta_answer,
)


class SequenceOllama:
    name = "ollama"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[str, str]] = []
        self.last_metrics = {}

    def generate_chat(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.outputs.pop(0)


def _hypertension_context() -> str:
    return (
        "[S1]\nCím: Magasvérnyomás-betegség\nSzakasz: Szövődmények\nTartalom:\n"
        "A tartósan magas vérnyomás növeli a szív- és érrendszeri szövődmények kockázatát. "
        "A kezeletlen állapot hosszabb távon szervkárosodással járhat.\n\n"
        "[S2]\nCím: Vaginizmus\nSzakasz: Mikor forduljon orvoshoz?\nTartalom:\n"
        "A panaszok esetén nőgyógyászati vizsgálat javasolt.\n\n"
        "[S3]\nCím: Magasvérnyomás-betegség\nSzakasz: Mikor forduljon orvoshoz?\nTartalom:\n"
        "Ismételten 140/90 Hgmm feletti vérnyomásértékek esetén orvosi kivizsgálás szükséges. "
        "Mellkasi fájdalom vagy légszomj esetén mielőbbi orvosi ellátás indokolt."
    )


def test_rejects_english_meta_reasoning() -> None:
    answer = "Okay, let's tackle this problem. The user provided some medical content and wants an answer."
    assert looks_like_english_meta_answer(answer)
    assert answer_requires_hungarian_retry(answer)


def test_accepts_normal_hungarian_medical_answer() -> None:
    answer = (
        "A tartósan magas vérnyomás növeli a szív- és érrendszeri szövődmények kockázatát [S1]. "
        "Orvosi kivizsgálás indokolt, ha az értékek ismételten magasak [S2]."
    )
    assert not answer_requires_hungarian_retry(answer)


def test_quality_guard_rejects_incomplete_two_part_answer() -> None:
    query = "Melyek a magasvérnyomás-betegség fő kockázatai, és mikor szükséges orvosi kivizsgálás?"
    answer = "Orvosi kivizsgálás ismételten magas értékeknél indokolt [S3]."
    assert answer_requires_quality_retry(query, answer, _hypertension_context())


def test_grounded_generator_accepts_substantive_hungarian_repair() -> None:
    repaired = (
        "### Fő kockázatok\n"
        "A tartósan magas vérnyomás növeli a szív- és érrendszeri szövődmények kockázatát, "
        "és kezeletlenül hosszabb távú szervkárosodással járhat [S1].\n\n"
        "### Mikor indokolt orvosi kivizsgálás?\n"
        "Ismételten 140/90 Hgmm feletti értékek esetén orvosi kivizsgálás szükséges [S3]. "
        "Mellkasi fájdalom vagy légszomj esetén mielőbbi orvosi ellátás indokolt [S3]."
    )
    llm = SequenceOllama([
        "Okay, let's tackle this problem. The user asks about hypertension.",
        repaired,
    ])
    generator = GroundedGenerator(llm)
    answer, citations = generator.generate(
        "Melyek a magasvérnyomás-betegség fő kockázatai, és mikor szükséges orvosi kivizsgálás?",
        _hypertension_context(),
    )
    assert answer == repaired
    assert citations == ["[S1]", "[S3]"]
    assert generator.last_generation_mode == "grounded-repair"
    assert len(llm.calls) == 2
    assert "KIZÁRÓLAG magyarul" in llm.calls[0][0]


def test_repeated_bad_output_returns_structured_source_synthesis() -> None:
    llm = SequenceOllama([
        "Okay, let's tackle this problem. The user asks about hypertension.",
        "The provided content says hypertension can cause complications and the user needs evaluation.",
    ])
    generator = GroundedGenerator(llm)
    answer, citations = generator.generate(
        "Melyek a magasvérnyomás-betegség fő kockázatai, és mikor szükséges orvosi kivizsgálás?",
        _hypertension_context(),
    )
    assert "### Fő kockázatok" in answer
    assert "### Mikor indokolt orvosi kivizsgálás?" in answer
    assert "szív- és érrendszeri" in answer
    assert "140/90" in answer
    assert "vaginizmus" not in answer.lower()
    assert "lokális modell" not in answer.lower()
    assert citations == ["[S1]", "[S3]"]
    assert generator.last_generation_mode == "source-synthesis-fallback"


def test_extractive_fallback_is_conservative() -> None:
    answer = build_evidence_fallback_answer(
        "Mikor szükséges orvosi kivizsgálás?",
        "[S1]\nCím: Magasvérnyomás-betegség\nTartalom:\nIsmételten magas értékek esetén orvosi kivizsgálás szükséges.\n\n"
        "[S2]\nCím: Vaginizmus\nTartalom:\nA vaginizmus más egészségügyi állapot.",
    )
    assert "orvosi kivizsgálás" in answer
    assert "vaginizmus" not in answer.lower()


def test_fallback_rejoins_html_linebreak_fragments_into_complete_sentences() -> None:
    context = (
        "[S1]\nCím: Magasvérnyomás-betegség\nSzakasz: Bevezetés\nTartalom:\n"
        "Az\nalacsony vérnyomás\n(hipotónia) az esetek többségében nem jelent egészségügyi kockázatot, "
        "ám a tartósan\nmagas vérnyomás\n(hipertónia) komoly, több szervet érintő kárt is tud okozni.\n\n"
        "[S2]\nCím: Magasvérnyomás-betegség\nSzakasz: Mikor forduljon orvoshoz?\nTartalom:\n"
        "Ha otthonában legalább 3 alkalommal nyugalomban magas vérnyomásértékeket mért különböző "
        "időpontokban, mindenképpen forduljon háziorvosához!"
    )
    answer = build_evidence_fallback_answer(
        "Melyek a magasvérnyomás-betegség fő kockázatai, és mikor szükséges orvosi kivizsgálás?",
        context,
    )

    assert "A tartósan magas vérnyomás" in answer
    assert "alacsony vérnyomás" not in answer
    assert "legalább 3 alkalommal" in answer
    assert "- (hipotónia)" not in answer


class BudgetAwareOllama(SequenceOllama):
    def __init__(self, outputs: list[str], num_predict: int = 256) -> None:
        super().__init__(outputs)
        self.num_predict = num_predict
        self.predict_seen: list[int] = []

    def generate_chat(self, system_prompt: str, user_prompt: str) -> str:
        self.predict_seen.append(self.num_predict)
        return super().generate_chat(system_prompt, user_prompt)


def test_detailed_generation_temporarily_increases_output_budget() -> None:
    answer = (
        "### Fő kockázatok\n"
        "A tartósan magas vérnyomás növeli a szív- és érrendszeri kockázatot [S1]. "
        "Kezeletlenül hosszabb távon szervkárosodással járhat [S1].\n\n"
        "### Mikor indokolt orvosi kivizsgálás?\n"
        "Ismételten magas értékeknél orvosi kivizsgálás szükséges [S3]. "
        "Mellkasi fájdalom vagy légszomj esetén mielőbbi ellátás indokolt [S3]."
    )
    llm = BudgetAwareOllama([answer], num_predict=256)
    generator = GroundedGenerator(llm)
    generator.generate(
        "Melyek a magasvérnyomás-betegség fő kockázatai, és mikor szükséges orvosi kivizsgálás?",
        _hypertension_context(),
    )
    assert llm.predict_seen[0] == 384
    assert llm.num_predict == 256


def test_rich_context_rejects_overly_short_multi_part_answer() -> None:
    query = "Melyek a magasvérnyomás-betegség fő kockázatai, és mikor szükséges orvosi kivizsgálás?"
    rich_context = _hypertension_context() + (" További releváns evidence a hipertónia kockázatairól és kivizsgálásáról." * 40)
    short_answer = (
        "### Fő kockázatok\nA hipertónia szervkárosodást okozhat [S1].\n\n"
        "### Mikor indokolt orvosi kivizsgálás?\nIsmételten magas értékeknél orvoshoz kell fordulni [S3]."
    )
    assert answer_requires_quality_retry(query, short_answer, rich_context)


def test_fallback_uses_multiple_sentences_per_section_when_evidence_is_available() -> None:
    context = (
        "[S1]\nCím: Magasvérnyomás-betegség\nSzakasz: Kockázatok\nTartalom:\n"
        "A tartósan magas vérnyomás több szervet érintő kárt okozhat. "
        "A hosszú ideje fennálló hipertónia károsítja a keringési rendszert. "
        "Az extrém vérnyomás-emelkedés stroke-hoz, szívinfarktushoz vagy veseelégtelenséghez vezethet.\n\n"
        "[S2]\nCím: Magasvérnyomás-betegség\nSzakasz: Mikor forduljon orvoshoz?\nTartalom:\n"
        "Ha legalább három alkalommal magas értéket mér, forduljon háziorvoshoz. "
        "Ha kezelés mellett a vérnyomás meghaladja a 140/90 Hgmm-t, forduljon ismételten orvoshoz. "
        "180/120 Hgmm felett sürgősségi ellátásra lehet szükség. "
        "Hipertenzív krízis jeleinél hívja a 112-t és kérjen mentőt."
    )
    answer = build_evidence_fallback_answer(
        "Melyek a magasvérnyomás-betegség fő kockázatai, és mikor szükséges orvosi kivizsgálás?",
        context,
    )
    assert answer.count("[S1]") >= 2
    assert answer.count("[S2]") >= 3
    assert "- A tartósan" not in answer
    assert len(answer.split()) >= 70
