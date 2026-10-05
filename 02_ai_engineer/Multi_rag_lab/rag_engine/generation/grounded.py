from __future__ import annotations

import re

from rag_engine.generation.fallbacks import build_evidence_fallback_answer
from rag_engine.generation.prompts import build_grounded_prompt_parts
from rag_engine.generation.validators import answer_requires_hungarian_retry, answer_requires_quality_retry


class GroundedGenerator:
    def __init__(self, llm, *, context_profile: str = "balanced", prompt_profile: str = "professional") -> None:
        self.llm = llm
        self.context_profile = context_profile
        self.prompt_profile = prompt_profile
        self.last_prompt = ""
        self.last_prompt_parts: dict[str, str] = {}
        self.last_generation_mode = "standard"

    def _invoke_with_min_budget(self, system_prompt: str, user_prompt: str, *, min_predict: int = 0) -> str:
        """Invoke the provider while temporarily reserving enough output budget for detailed answers.

        Small local Ollama profiles often use ``num_predict=256`` for speed. That is
        sufficient for short answers, but a two-part grounded answer can be truncated
        before both sections are explained. The override is local to this call and is
        restored immediately, so runtime profiles remain the source of truth elsewhere.
        """
        original_predict = getattr(self.llm, "num_predict", None)
        can_override = isinstance(original_predict, int) and min_predict > 0 and original_predict < min_predict
        if can_override:
            self.llm.num_predict = min_predict
        try:
            generate_chat = getattr(self.llm, "generate_chat", None)
            if callable(generate_chat):
                return str(generate_chat(system_prompt, user_prompt))
            return str(self.llm.generate(f"{system_prompt}\n\n{user_prompt}"))
        finally:
            if can_override:
                self.llm.num_predict = original_predict

    def _invoke(self, parts: dict[str, str]) -> str:
        min_predict = 0 if self.prompt_profile == "concise" else 384
        return self._invoke_with_min_budget(
            parts["instructions"],
            parts["user_prompt"],
            min_predict=min_predict,
        )

    def _repair_grounded_answer(self, *, query: str, context: str, answer_plan: str) -> str:
        system_prompt = (
            "Te egy magyar nyelvű, forrásokra támaszkodó RAG-asszisztens vagy. "
            "KIZÁRÓLAG magyarul írj. Ne írj gondolatmenetet, metaszöveget vagy angol bevezetőt. "
            "Ne említsd a modellt, a RAG-rendszert, a javítási lépést vagy fallbacket. "
            "Csak a megadott bizonyítékokkal alátámasztható állításokat használd. "
            "Minden érdemi tényállítást jelölj [S1], [S2] formában. "
            "A kérdés minden külön részkérdését válaszold meg, és az irreleváns betegségeket hagyd figyelmen kívül. "
            "Szakaszonként 3–5 teljes, természetes, egymásra épülő mondatot írj, ha van hozzá elegendő evidence. "
            "A HTML-tördelés miatt szétszakadt forrásmondatokat állítsd helyre, de új tényt ne adj hozzá, és ne hagyj félbe mondatot. "
            "Ha a bizonyítékok elegendők, készíts kb. 180–320 szavas választ; ne rövidítsd egyetlen bulletre az egyik részkérdést."
        )
        user_prompt = (
            f"Kérdés:\n{query}\n\nBizonyítékok:\n{context}\n\n"
            f"Kért válaszszerkezet:\n{answer_plan}\n\n"
            "Készíts új, részletesebb, koherens és közvetlen magyar választ a bizonyítékok alapján. "
            "Ne egyszerűen másold ki a forrásmondatokat: fogalmazd őket természetes, összefüggő, magyarázó válasszá, "
            "miközben minden tény forrása ellenőrizhető marad. Ne kommentáld a korábbi próbálkozást. Magyar válasz:"
        )
        return self._invoke_with_min_budget(system_prompt, user_prompt, min_predict=512)

    @staticmethod
    def _needs_retry(query: str, answer: str, context: str) -> bool:
        return answer_requires_hungarian_retry(answer) or answer_requires_quality_retry(query, answer, context)

    def generate(self, query: str, context: str) -> tuple[str, list[str]]:
        parts = build_grounded_prompt_parts(
            query,
            context,
            context_profile=self.context_profile,
            prompt_profile=self.prompt_profile,
        )
        self.last_prompt = parts["final_prompt"]
        self.last_prompt_parts = parts
        self.last_generation_mode = "standard"

        answer = self._invoke(parts).strip()
        validate_output = str(getattr(self.llm, "name", "")).lower() == "ollama"
        if validate_output and self._needs_retry(query, answer, context):
            repaired = self._repair_grounded_answer(
                query=query,
                context=context,
                answer_plan=parts.get("answer_plan", ""),
            ).strip()
            if repaired and not self._needs_retry(query, repaired, context):
                answer = repaired
                self.last_generation_mode = "grounded-repair"
            else:
                answer = build_evidence_fallback_answer(query, context)
                self.last_generation_mode = "source-synthesis-fallback"

        citations = sorted(set(re.findall(r"\[S\d+\]", answer)))
        return answer, citations
