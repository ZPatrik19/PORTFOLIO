from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Any, Callable

import pandas as pd
import plotly.express as px
import streamlit as st

from prompt_benchmark.evaluation.parsing import parse_prediction
from prompt_benchmark.prompts import get_strategy, list_custom_prompts, load_custom_prompt, save_custom_prompt, CustomPromptStrategy
from prompt_benchmark.prompts.base import PromptPayload
from prompt_benchmark.ui.playground_data import reshape_token_usage
from prompt_benchmark.ui.playground_store import (
    append_playground_history,
    list_generation_presets,
    load_generation_preset,
    load_playground_history,
    save_generation_preset,
)

TEXT = {
    "hu": {
        "title": "Playground — prompt fejlesztés és valódi szöveggenerálás",
        "intro": "Próbálj ki promptokat egyetlen inputon, mentsd el őket presetként, hasonlíts össze variánsokat, majd a stabil promptot vidd tovább benchmarkra.",
        "mode": "Playground mód",
        "classification": "🎯 Klasszifikáció",
        "generation": "✍️ Szabad szöveggenerálás",
        "compare": "⚖️ Prompt A/B összehasonlítás",
        "run": "▶ Futtatás",
        "save": "💾 Prompt mentése",
        "history": "🕘 Playground history",
    },
    "en": {
        "title": "Playground — prompt development and real text generation",
        "intro": "Try prompts on one input, save presets, compare variants, then promote the stable prompt into the benchmark.",
        "mode": "Playground mode",
        "classification": "🎯 Classification",
        "generation": "✍️ Free text generation",
        "compare": "⚖️ Prompt A/B comparison",
        "run": "▶ Run",
        "save": "💾 Save prompt",
        "history": "🕘 Playground history",
    },
}



GENERATION_AB_TEMPLATES = {
    "minimal_zero_shot": {
        "title_hu": "T0 · Minimal / zero-shot",
        "title_en": "T0 · Minimal / zero-shot",
        "system_hu": "",
        "system_en": "",
        "user_hu": "Válaszolj az alábbi kérésre:\n{input}",
        "user_en": "Answer the following request:\n{input}",
    },
    "persona_expert": {
        "title_hu": "T1 · Persona — senior AI Engineer",
        "title_en": "T1 · Persona — senior AI Engineer",
        "system_hu": "Te egy senior AI Engineer vagy. Technikailag pontos, production-szemléletű választ adj.",
        "system_en": "You are a senior AI Engineer. Give a technically precise, production-oriented answer.",
        "user_hu": "Feladat:\n{input}",
        "user_en": "Task:\n{input}",
    },
    "context_constraints": {
        "title_hu": "T2 · Context + explicit constraints",
        "title_en": "T2 · Context + explicit constraints",
        "system_hu": "Te egy precíz szakmai asszisztens vagy. A választ csak a feladat szempontjából releváns információkra építsd.",
        "system_en": "You are a precise technical assistant. Base the answer only on information relevant to the task.",
        "user_hu": "KONTEXTUS: A cél egy szakmai, gyakorlatban használható válasz.\nSZABÁLYOK: ne ismételd a kérdést; ne találj ki adatot; használj konkrét terminológiát.\nINPUT:\n{input}",
        "user_en": "CONTEXT: The goal is a practical professional answer.\nRULES: do not repeat the question; do not invent facts; use concrete terminology.\nINPUT:\n{input}",
    },
    "audience_tone": {
        "title_hu": "T3 · Audience + tone + format",
        "title_en": "T3 · Audience + tone + format",
        "system_hu": "Magyarázz egy szoftverfejlesztő közönségnek, szakmai, tömör hangnemben.",
        "system_en": "Explain to a software-engineer audience in a professional, concise tone.",
        "user_hu": "Adj 5 rövid bullet pointot és egy konkrét példát.\nTÉMA:\n{input}",
        "user_en": "Give 5 concise bullet points and one concrete example.\nTOPIC:\n{input}",
    },
    "few_shot_structure": {
        "title_hu": "T4 · Few-shot strukturált válasz",
        "title_en": "T4 · Few-shot structured answer",
        "system_hu": "Kövesd a megadott válaszstruktúrát.",
        "system_en": "Follow the demonstrated response structure.",
        "user_hu": "PÉLDA FORMÁTUM:\nÖsszefoglaló: ...\nMiért fontos: ...\nGyakorlati példa: ...\nTrade-off: ...\n\nMost válaszolj ugyanebben a formában:\n{input}",
        "user_en": "EXAMPLE FORMAT:\nSummary: ...\nWhy it matters: ...\nPractical example: ...\nTrade-off: ...\n\nNow answer in the same format:\n{input}",
    },
    "critical_analysis": {
        "title_hu": "T5 · Strukturált kritikai elemzés",
        "title_en": "T5 · Structured critical analysis",
        "system_hu": "Elemezd a feladatot mérnöki trade-offok alapján. Ne írj rejtett gondolatmenetet; csak a végső indokolt következtetést add.",
        "system_en": "Analyze the task using engineering trade-offs. Do not expose hidden chain-of-thought; give only the final justified conclusion.",
        "user_hu": "Válaszstruktúra: 1) megoldás, 2) előnyök, 3) kockázatok, 4) mikor választanád, 5) rövid döntés.\n\nINPUT:\n{input}",
        "user_en": "Response structure: 1) solution, 2) benefits, 3) risks, 4) when to choose it, 5) short decision.\n\nINPUT:\n{input}",
    },
    "json_contract": {
        "title_hu": "T6 · JSON output contract",
        "title_en": "T6 · JSON output contract",
        "system_hu": "Csak valid JSON-t adj vissza, markdown nélkül.",
        "system_en": "Return valid JSON only, without markdown.",
        "user_hu": 'Elemezd az inputot és add vissza ezt a sémát: {"summary":"...","key_points":["..."],"risk":"..."}.\nINPUT:\n{input}',
        "user_en": 'Analyze the input and return this schema: {"summary":"...","key_points":["..."],"risk":"..."}.\nINPUT:\n{input}',
    },
}


def _ab_overrides(language: str, side: str) -> dict[str, object]:
    label = "A" if side == "a" else "B"
    with st.expander((f"⚙️ Prompt {label} paraméterek" if language == "hu" else f"⚙️ Prompt {label} parameters"), expanded=False):
        max_tokens = st.slider("max_output_tokens", 32, 2048, 512, 32, key=f"{language}_ab_{side}_max_tokens_v4")
        use_temp = st.checkbox("temperature override", False, key=f"{language}_ab_{side}_use_temp_v4")
        temp = st.slider("temperature", 0.0, 2.0, 0.2, 0.1, disabled=not use_temp, key=f"{language}_ab_{side}_temp_v4")
        use_top_p = st.checkbox("top_p override", False, key=f"{language}_ab_{side}_use_top_p_v4")
        top_p = st.slider("top_p", 0.05, 1.0, 0.95, 0.05, disabled=not use_top_p, key=f"{language}_ab_{side}_top_p_v4")
        use_top_k = st.checkbox("top_k override", False, key=f"{language}_ab_{side}_use_top_k_v4")
        top_k = st.slider("top_k", 1, 100, 40, 1, disabled=not use_top_k, key=f"{language}_ab_{side}_top_k_v4")
    values: dict[str, object] = {"max_output_tokens": max_tokens}
    if use_temp:
        values["temperature"] = temp
    if use_top_p:
        values["top_p"] = top_p
    if use_top_k:
        values["top_k"] = top_k
    return values


def _generation_quality_metrics(text: str, required_keywords: list[str], output_format: str) -> dict[str, object]:
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    bullets = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text))
    coverage = 1.0
    if required_keywords:
        lowered = text.lower()
        coverage = sum(1 for k in required_keywords if k.lower() in lowered) / len(required_keywords)
    json_valid = False
    try:
        json.loads(text)
        json_valid = True
    except (json.JSONDecodeError, TypeError):
        json_valid = False
    if output_format == "JSON":
        format_ok = json_valid
    elif output_format == "Bullets":
        format_ok = bullets >= 3
    else:
        format_ok = True
    return {
        "words": len(words),
        "characters": len(text),
        "bullets": bullets,
        "keyword_coverage": coverage,
        "json_valid": json_valid,
        "format_ok": format_ok,
    }


def _load_generation_template(language: str, template_key: str) -> tuple[str, str]:
    item = GENERATION_AB_TEMPLATES[template_key]
    suffix = "hu" if language == "hu" else "en"
    return str(item[f"system_{suffix}"]), str(item[f"user_{suffix}"])



def render_playground_panel(
    *,
    language: str,
    provider: str,
    model: str,
    strategies: list[str],
    strategy_title: Callable[[str], str],
    build_client: Callable[..., Any],
) -> None:
    t = TEXT[language]
    st.subheader(t["title"])
    st.write(t["intro"])
    mode = st.radio(t["mode"], [t["classification"], t["generation"], t["compare"]], horizontal=True, key=f"{language}_play_mode_v3")

    if mode == t["classification"]:
        source = st.radio(
            "Prompt forrás" if language == "hu" else "Prompt source",
            ["Beépített" if language == "hu" else "Built-in", "Mentett custom" if language == "hu" else "Saved custom"],
            horizontal=True,
            key=f"{language}_play_class_source_v3",
        )
        if source.startswith("Beép") or source.startswith("Built"):
            name = st.selectbox("Stratégia" if language == "hu" else "Strategy", strategies, format_func=strategy_title, key=f"{language}_play_class_strategy_v3")
            strategy = get_strategy(name)
        else:
            presets = {p.stem: p for p in list_custom_prompts()}
            if not presets:
                st.info("Nincs mentett custom prompt." if language == "hu" else "No saved custom prompts.")
                strategy = None
            else:
                preset = st.selectbox("Custom preset", list(presets), key=f"{language}_play_class_custom_v3")
                strategy = load_custom_prompt(presets[preset])
        ticket = st.text_area(
            "Ticket / input",
            "I was charged twice, but my main request is to cancel before the next renewal.",
            height=140,
            key=f"{language}_play_class_ticket_v3",
        )
        expected = st.selectbox("Várt címke" if language == "hu" else "Expected label", ["", "api", "billing", "cancellation", "complaint", "technical", "upgrade"], key=f"{language}_play_class_expected_v3")
        if strategy is not None:
            payload = strategy.build(ticket)
            with st.expander("Renderelt prompt" if language == "hu" else "Rendered prompt"):
                st.code(f"SYSTEM:\n{payload.instructions or '(none)'}\n\nUSER:\n{payload.input_text}", language="text")
            c1, c2 = st.columns([1, 1])
            run_clicked = c1.button(t["run"], type="primary", key=f"{language}_play_class_run_v3")
            save_clicked = c2.button(t["save"], key=f"{language}_play_class_save_v3")
            if save_clicked:
                user_template = payload.input_text.replace(ticket, "{ticket}")
                preset = CustomPromptStrategy(
                    display_name=f"playground_{getattr(strategy, 'name', 'prompt')}",
                    system_prompt=payload.instructions or "",
                    user_template=user_template,
                    output_mode=payload.output_mode,
                    structured_output=payload.structured_output,
                    reasoning_effort=payload.reasoning_effort,
                )
                path = save_custom_prompt(preset)
                st.success(("Mentve: " if language == "hu" else "Saved: ") + str(path))
            if run_clicked:
                client = build_client()
                response = client.classify(payload)
                parsed = parse_prediction(response.raw_output, payload.output_mode)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Prediction", parsed.label or "INVALID")
                m2.metric("Tokens", response.total_tokens)
                m3.metric("Latency", f"{response.latency_seconds:.3f}s")
                m4.metric("Correct", "—" if not expected else ("✅" if parsed.label == expected else "❌"))
                st.code(response.raw_output, language="text")
                append_playground_history({
                    "mode": "classification", "provider": provider, "model": model,
                    "prompt": getattr(strategy, "name", "custom"), "input": ticket,
                    "output": response.raw_output, "prediction": parsed.label,
                    "expected": expected, "tokens": response.total_tokens,
                    "latency_seconds": response.latency_seconds,
                })

    elif mode == t["generation"]:
        presets = list_generation_presets()
        preset_choice = st.selectbox(
            "Mentett generatív preset" if language == "hu" else "Saved generation preset",
            ["(new)", *presets.keys()],
            key=f"{language}_gen_preset_v3",
        )
        default_system = "Te egy precíz AI asszisztens vagy. Adj tömör, hasznos és jól strukturált választ." if language == "hu" else "You are a precise AI assistant. Give a concise, useful and well-structured answer."
        default_user = "Magyarázd el röviden, miért fontos a prompt engineering egy production LLM rendszerben." if language == "hu" else "Briefly explain why prompt engineering matters in a production LLM system."
        if preset_choice != "(new)" and st.button("Preset betöltése" if language == "hu" else "Load preset", key=f"{language}_gen_load_v3"):
            loaded = load_generation_preset(presets[preset_choice])
            st.session_state[f"{language}_gen_name_v3"] = loaded.get("name", preset_choice)
            st.session_state[f"{language}_gen_system_v3"] = loaded.get("system_prompt", "")
            st.session_state[f"{language}_gen_user_v3"] = loaded.get("user_template", "")
            st.rerun()
        st.session_state.setdefault(f"{language}_gen_name_v3", "production_explainer")
        st.session_state.setdefault(f"{language}_gen_system_v3", default_system)
        st.session_state.setdefault(f"{language}_gen_user_v3", default_user)
        name = st.text_input("Prompt neve" if language == "hu" else "Prompt name", key=f"{language}_gen_name_v3")
        system_prompt = st.text_area("System prompt", height=120, key=f"{language}_gen_system_v3")
        user_prompt = st.text_area("User prompt", height=220, key=f"{language}_gen_user_v3")
        max_tokens = st.slider("Max output tokens", 64, 2048, 512, 64, key=f"{language}_gen_tokens_v3")
        c1, c2 = st.columns(2)
        if c1.button(t["save"], key=f"{language}_gen_save_v3"):
            path = save_generation_preset(name, system_prompt, user_prompt, {"max_output_tokens": max_tokens})
            st.success(("Mentve: " if language == "hu" else "Saved: ") + str(path))
        if c2.button(t["run"], type="primary", key=f"{language}_gen_run_v3"):
            client = build_client({"max_output_tokens": max_tokens})
            payload = PromptPayload(
                strategy_name=f"playground_generation_{name}",
                instructions=system_prompt or None,
                input_text=user_prompt,
                output_mode="text",
            )
            response = client.classify(payload)
            st.markdown("### " + ("Generált válasz" if language == "hu" else "Generated response"))
            st.write(response.raw_output)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Input tokens", response.input_tokens)
            m2.metric("Output tokens", response.output_tokens)
            m3.metric("Total tokens", response.total_tokens)
            m4.metric("Latency", f"{response.latency_seconds:.3f}s")
            if provider == "mock":
                st.info("Mock módban ez szimulált szöveg. Valódi generáláshoz válassz Gemini/Groq/OpenRouter/OpenAI/Ollama providert." if language == "hu" else "In mock mode this is simulated text. Select Gemini/Groq/OpenRouter/OpenAI/Ollama for real generation.")
            append_playground_history({
                "mode": "generation", "provider": provider, "model": model, "prompt": name,
                "system_prompt": system_prompt, "input": user_prompt, "output": response.raw_output,
                "tokens": response.total_tokens, "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens, "latency_seconds": response.latency_seconds,
            })

    else:
        st.caption(
            "Válaszd ki egyértelműen az A és B promptstratégiát/template-et, állíts külön sampling paramétereket, majd ugyanazon inputon hasonlítsd össze a minőségi és erőforrás-metrikákat."
            if language == "hu" else
            "Choose the A and B prompt strategy/template explicitly, tune sampling parameters independently, then compare quality and resource metrics on the same input."
        )
        task_type = st.radio(
            "A/B feladat" if language == "hu" else "A/B task",
            ["🎯 Klasszifikáció" if language == "hu" else "🎯 Classification", "✍️ Szöveggenerálás" if language == "hu" else "✍️ Text generation"],
            horizontal=True,
            key=f"{language}_ab_task_v4",
        )

        if task_type.startswith("🎯"):
            input_text = st.text_area(
                "Azonos ticket mindkét stratégiának" if language == "hu" else "Same ticket for both strategies",
                "I was charged twice last month, but that issue is resolved. My current request is to cancel before the next renewal.",
                height=140,
                key=f"{language}_ab_class_input_v4",
            )
            expected_options = ["", "api", "billing", "cancellation", "complaint", "technical", "upgrade"]
            expected = st.selectbox(
                "Várt címke / ground truth" if language == "hu" else "Expected label / ground truth",
                expected_options,
                index=0,
                format_func=(lambda value: ("— nincs megadva —" if language == "hu" else "— not provided —") if value == "" else value),
                key=f"{language}_ab_expected_v5",
                help=(
                    "Ez egy kézzel megadott referencia címke. Ha nem vagy biztos benne, hagyd üresen; ilyenkor a Playground nem jelöli helyesnek/hibásnak a választ."
                    if language == "hu" else
                    "This is a manually supplied reference label. Leave it empty if uncertain; the Playground will then show correctness as N/A."
                ),
            )
            a, b = st.columns(2)
            with a:
                st.markdown("### A")
                strat_a = st.selectbox("Promptstratégia A" if language == "hu" else "Prompt strategy A", strategies, format_func=strategy_title, index=0, key=f"{language}_ab_strat_a_v4")
                strategy_a = get_strategy(strat_a)
                payload_a = strategy_a.build(input_text)
                st.caption(strategy_title(strat_a))
                with st.expander("Prompt A — teljes render" if language == "hu" else "Prompt A — full render"):
                    st.code(f"SYSTEM:\n{payload_a.instructions or '(none)'}\n\nUSER:\n{payload_a.input_text}", language="text")
                overrides_a = _ab_overrides(language, "a")
            with b:
                st.markdown("### B")
                default_b = strategies.index("p16_full_advanced_template") if "p16_full_advanced_template" in strategies else min(1, len(strategies)-1)
                strat_b = st.selectbox("Promptstratégia B" if language == "hu" else "Prompt strategy B", strategies, format_func=strategy_title, index=default_b, key=f"{language}_ab_strat_b_v4")
                strategy_b = get_strategy(strat_b)
                payload_b = strategy_b.build(input_text)
                st.caption(strategy_title(strat_b))
                with st.expander("Prompt B — teljes render" if language == "hu" else "Prompt B — full render"):
                    st.code(f"SYSTEM:\n{payload_b.instructions or '(none)'}\n\nUSER:\n{payload_b.input_text}", language="text")
                overrides_b = _ab_overrides(language, "b")

            if st.button(t["run"], type="primary", key=f"{language}_ab_class_run_v4"):
                rows = []
                for label, strategy, overrides in [("A", strategy_a, overrides_a), ("B", strategy_b, overrides_b)]:
                    client = build_client(overrides)
                    payload = strategy.build(input_text)
                    response = client.classify(payload)
                    parsed = parse_prediction(response.raw_output, payload.output_mode)
                    prompt_chars = len((payload.instructions or "") + payload.input_text)
                    row = {
                        "variant": label,
                        "strategy": strategy_title(strategy.name),
                        "prediction": parsed.label or "INVALID",
                        "correct": None if not expected else parsed.label == expected,
                        "valid_output": parsed.valid_output,
                        "prompt_chars": prompt_chars,
                        "approx_prompt_tokens": round(prompt_chars / 4),
                        "tokens": response.total_tokens,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "latency_seconds": response.latency_seconds,
                        "temperature": overrides.get("temperature"),
                        "top_p": overrides.get("top_p"),
                        "top_k": overrides.get("top_k"),
                        "max_output_tokens": overrides.get("max_output_tokens"),
                        "response": response.raw_output,
                    }
                    rows.append(row)
                    append_playground_history({"mode": "ab_classification", "provider": provider, "model": model, "input": input_text, "expected": expected, **row})
                result_df = pd.DataFrame(rows)
                c1, c2 = st.columns(2)
                for col, row in zip((c1, c2), rows):
                    with col:
                        st.markdown(f"### {row['variant']} · {row['strategy']}")
                        m = st.columns(4)
                        m[0].metric("Prediction", row["prediction"])
                        m[1].metric("Correct", "N/A" if row["correct"] is None else ("✅" if row["correct"] else "❌"))
                        m[2].metric("Tokens", row["tokens"])
                        m[3].metric("Latency", f"{row['latency_seconds']:.3f}s")
                        st.code(row["response"], language="text")
                rc1, rc2 = st.columns(2)
                token_long = reshape_token_usage(result_df, id_vars=["variant", "strategy"])
                rc1.plotly_chart(px.bar(token_long, x="variant", y="token_count", color="token_type", barmode="stack", hover_data=["strategy"], title="Token usage A/B"), use_container_width=True)
                rc2.plotly_chart(px.bar(result_df, x="variant", y="latency_seconds", color="variant", hover_data=["strategy"], title="Latency A/B"), use_container_width=True)
                st.dataframe(result_df.drop(columns=["response"]), use_container_width=True, hide_index=True)

        else:
            input_text = st.text_area(
                "Azonos input mindkét promptnak" if language == "hu" else "Same input for both prompts",
                "Explain RAG to a software engineer in 5 bullet points and mention embeddings, retrieval, context, generation and evaluation.",
                height=150,
                key=f"{language}_ab_gen_input_v4",
            )
            eval_cols = st.columns(2)
            required_raw = eval_cols[0].text_input(
                "Elvárt kulcsszavak (vesszővel)" if language == "hu" else "Expected keywords (comma-separated)",
                "embeddings,retrieval,context,generation,evaluation",
                key=f"{language}_ab_keywords_v4",
            )
            output_format = eval_cols[1].selectbox("Elvárt formátum" if language == "hu" else "Expected format", ["Any", "Bullets", "JSON"], key=f"{language}_ab_format_v4")
            required_keywords = [x.strip() for x in required_raw.split(",") if x.strip()]

            template_keys = list(GENERATION_AB_TEMPLATES)
            display_map = {
                k: GENERATION_AB_TEMPLATES[k]["title_hu" if language == "hu" else "title_en"]
                for k in template_keys
            }
            a, b = st.columns(2)
            variants = {}
            for side, col, default_idx in [("a", a, 0), ("b", b, min(5, len(template_keys)-1))]:
                with col:
                    st.markdown(f"### {'A' if side == 'a' else 'B'}")
                    choice = st.selectbox(
                        "Előre definiált template" if language == "hu" else "Predefined template",
                        template_keys,
                        format_func=lambda k: display_map[k],
                        index=default_idx,
                        key=f"{language}_ab_gen_template_{side}_v4",
                    )
                    sys_key = f"{language}_ab_gen_sys_{side}_v4"
                    usr_key = f"{language}_ab_gen_usr_{side}_v4"
                    last_key = f"{language}_ab_gen_last_template_{side}_v4"
                    if st.session_state.get(last_key) != choice:
                        system_default, user_default = _load_generation_template(language, choice)
                        st.session_state[sys_key] = system_default
                        st.session_state[usr_key] = user_default
                        st.session_state[last_key] = choice
                    system = st.text_area("System prompt", height=120, key=sys_key)
                    user_template = st.text_area("User template", height=190, key=usr_key)
                    overrides = _ab_overrides(language, side)
                    variants[side] = (choice, system, user_template, overrides)

            if st.button(t["run"], type="primary", key=f"{language}_ab_gen_run_v4"):
                rows = []
                for side, label in [("a", "A"), ("b", "B")]:
                    choice, system, template, overrides = variants[side]
                    client = build_client(overrides)
                    rendered = template.replace("{input}", input_text)
                    payload = PromptPayload(
                        strategy_name=f"playground_ab_generation_{choice}",
                        instructions=system or None,
                        input_text=rendered,
                        output_mode="text",
                    )
                    response = client.classify(payload)
                    quality = _generation_quality_metrics(response.raw_output, required_keywords, output_format)
                    prompt_chars = len((system or "") + rendered)
                    row = {
                        "variant": label,
                        "template": display_map[choice],
                        "prompt_chars": prompt_chars,
                        "approx_prompt_tokens": round(prompt_chars / 4),
                        "tokens": response.total_tokens,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "latency_seconds": response.latency_seconds,
                        "temperature": overrides.get("temperature"),
                        "top_p": overrides.get("top_p"),
                        "top_k": overrides.get("top_k"),
                        "max_output_tokens": overrides.get("max_output_tokens"),
                        **quality,
                        "response": response.raw_output,
                    }
                    rows.append(row)
                    append_playground_history({"mode": "ab_generation", "provider": provider, "model": model, "input": input_text, **row})
                result_df = pd.DataFrame(rows)
                c1, c2 = st.columns(2)
                for col, row in zip((c1, c2), rows):
                    with col:
                        st.markdown(f"### Prompt {row['variant']} · {row['template']}")
                        st.write(row["response"])
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Tokens", row["tokens"])
                        m2.metric("Latency", f"{row['latency_seconds']:.3f}s")
                        m3.metric("Keyword coverage", f"{row['keyword_coverage']:.0%}")
                        m4, m5, m6 = st.columns(3)
                        m4.metric("Words", row["words"])
                        m5.metric("Bullets", row["bullets"])
                        m6.metric("Format", "✅" if row["format_ok"] else "❌")
                rc1, rc2, rc3 = st.columns(3)
                token_long = reshape_token_usage(result_df, id_vars=["variant", "template"])
                rc1.plotly_chart(px.bar(token_long, x="variant", y="token_count", color="token_type", barmode="stack", title="Token usage"), use_container_width=True)
                rc2.plotly_chart(px.bar(result_df, x="variant", y="latency_seconds", color="variant", title="Latency"), use_container_width=True)
                rc3.plotly_chart(px.bar(result_df, x="variant", y="keyword_coverage", color="variant", range_y=[0,1.05], title="Keyword coverage"), use_container_width=True)
                st.dataframe(result_df.drop(columns=["response"]), use_container_width=True, hide_index=True)

    with st.expander(t["history"], expanded=False):
        history = load_playground_history(50)
        if history.empty:
            st.info("Még nincs Playground history." if language == "hu" else "No Playground history yet.")
        else:
            cols = [c for c in ["timestamp", "mode", "provider", "model", "prompt", "variant", "tokens", "latency_seconds", "input", "output"] if c in history]
            st.dataframe(history[cols], use_container_width=True, hide_index=True)
