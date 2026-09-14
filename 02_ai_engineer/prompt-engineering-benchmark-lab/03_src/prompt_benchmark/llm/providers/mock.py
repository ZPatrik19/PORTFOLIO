from __future__ import annotations

import hashlib
import json
import re

from prompt_benchmark.llm.base import BaseLLMClient
from prompt_benchmark.llm.schemas import LLMResponse
from prompt_benchmark.prompts.base import PromptPayload

class MockLLMClient(BaseLLMClient):
    """Deterministic prompt-sensitivity simulator for offline portfolio demos.

    This is deliberately *not* a real language model. It uses the synthetic
    scenario metadata plus the selected prompt strategy to emulate realistic
    differences in semantic accuracy, output validity, token overhead and
    latency. The purpose is to make the complete benchmark/UI meaningful before
    an external API key is configured. Real portfolio evidence should still be
    produced with Ollama, Groq, Gemini or OpenAI.
    """

    provider = "mock"

    _BASE_ACCURACY = {
        "p0_zero_shot": 0.75,
        "p1_definitions": 0.79,
        "p2_role": 0.78,
        "p3_few_shot": 0.82,
        "p4_constraints": 0.84,
        "p5_decision_policy": 0.87,
        "p6_json": 0.86,
        "p7_structured_output": 0.87,
        "p8_persona": 0.79,
        "p9_instruction_context": 0.83,
        "p10_format_audience_tone": 0.81,
        "p11_delimited_data": 0.84,
        "p12_contrastive_few_shot": 0.88,
        "p13_reasoning_model": 0.89,
        "p14_tree_branch_vote": 0.85,
        "p15_grammar_constrained": 0.86,
        "p16_full_advanced_template": 0.91,
    }

    _CASE_ADJUSTMENT = {
        "easy_clear": 0.11,
        "implicit_request": -0.01,
        "ambiguous_boundary": -0.12,
        "multi_intent_primary": -0.10,
        "noisy_typo": -0.05,
        "long_context": -0.10,
        "prompt_injection": -0.15,
        "resolved_history": -0.09,
        "negation_correction": -0.11,
        "quoted_thread": -0.10,
        "multilingual_mixed": -0.06,
        "telegraphic_short": -0.04,
        "primary_last": -0.13,
        "primary_first": -0.10,
        "conditional_distractor": -0.12,
        "code_log_noise": -0.14,
        "label_word_attack": -0.17,
        "double_negation": -0.15,
    }

    _CASE_BONUS = {
        "p1_definitions": {"implicit_request": 0.03, "ambiguous_boundary": 0.03},
        "p2_role": {"prompt_injection": 0.02},
        "p3_few_shot": {"implicit_request": 0.05, "noisy_typo": 0.03, "ambiguous_boundary": 0.04},
        "p4_constraints": {"multi_intent_primary": 0.06, "prompt_injection": 0.05, "long_context": 0.02, "label_word_attack": 0.07, "conditional_distractor": 0.05},
        "p5_decision_policy": {"ambiguous_boundary": 0.09, "multi_intent_primary": 0.09, "resolved_history": 0.07, "negation_correction": 0.08, "quoted_thread": 0.06, "primary_last": 0.09, "primary_first": 0.07, "conditional_distractor": 0.10, "double_negation": 0.08},
        "p6_json": {"ambiguous_boundary": 0.08, "multi_intent_primary": 0.08, "resolved_history": 0.06},
        "p7_structured_output": {"ambiguous_boundary": 0.08, "multi_intent_primary": 0.08, "resolved_history": 0.06},
        "p8_persona": {"implicit_request": 0.01},
        "p9_instruction_context": {"long_context": 0.07, "multi_intent_primary": 0.05, "resolved_history": 0.05, "quoted_thread": 0.07, "multilingual_mixed": 0.03},
        "p10_format_audience_tone": {"long_context": 0.02},
        "p11_delimited_data": {"prompt_injection": 0.16, "long_context": 0.05, "resolved_history": 0.03, "quoted_thread": 0.08, "code_log_noise": 0.12, "label_word_attack": 0.18},
        "p12_contrastive_few_shot": {"ambiguous_boundary": 0.13, "multi_intent_primary": 0.10, "implicit_request": 0.05, "negation_correction": 0.07, "telegraphic_short": 0.04, "conditional_distractor": 0.08, "double_negation": 0.09},
        "p13_reasoning_model": {"ambiguous_boundary": 0.14, "multi_intent_primary": 0.12, "long_context": 0.09, "resolved_history": 0.09, "negation_correction": 0.10, "quoted_thread": 0.08, "primary_last": 0.11, "primary_first": 0.09, "conditional_distractor": 0.11, "code_log_noise": 0.07, "double_negation": 0.12},
        "p14_tree_branch_vote": {"ambiguous_boundary": 0.07, "multi_intent_primary": 0.06, "long_context": 0.05, "prompt_injection": 0.04, "primary_last": 0.07, "conditional_distractor": 0.08, "label_word_attack": 0.07, "double_negation": 0.08},
        "p15_grammar_constrained": {"ambiguous_boundary": 0.07, "multi_intent_primary": 0.07},
        "p16_full_advanced_template": {"implicit_request": 0.06, "ambiguous_boundary": 0.14, "multi_intent_primary": 0.13, "noisy_typo": 0.06, "long_context": 0.12, "prompt_injection": 0.15, "resolved_history": 0.11, "negation_correction": 0.12, "quoted_thread": 0.11, "multilingual_mixed": 0.08, "telegraphic_short": 0.06, "primary_last": 0.13, "primary_first": 0.10, "conditional_distractor": 0.13, "code_log_noise": 0.13, "label_word_attack": 0.17, "double_negation": 0.14},
    }

    _FORMAT_VALIDITY = {
        "p0_zero_shot": 0.925,
        "p1_definitions": 0.945,
        "p2_role": 0.965,
        "p3_few_shot": 0.972,
        "p4_constraints": 0.988,
        "p5_decision_policy": 0.992,
        "p6_json": 0.965,
        "p7_structured_output": 1.0,
        "p8_persona": 0.958,
        "p9_instruction_context": 0.982,
        "p10_format_audience_tone": 0.994,
        "p11_delimited_data": 0.991,
        "p12_contrastive_few_shot": 0.991,
        "p13_reasoning_model": 0.994,
        "p14_tree_branch_vote": 0.995,
        "p15_grammar_constrained": 1.0,
        "p16_full_advanced_template": 1.0,
    }

    _CONFUSIONS = {
        "api": "technical",
        "technical": "api",
        "billing": "cancellation",
        "cancellation": "billing",
        "complaint": "technical",
        "upgrade": "billing",
    }

    def __init__(
        self,
        model: str = "mock-prompt-sensitive-simulator-v2",
        temperature: float | None = 0.0,
        top_p: float | None = 1.0,
        top_k: int | None = 40,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    @staticmethod
    def _stable_unit(*parts: object) -> float:
        raw = "|".join(str(part) for part in parts).encode("utf-8")
        value = int(hashlib.sha256(raw).hexdigest()[:16], 16)
        return value / float(0xFFFFFFFFFFFFFFFF)

    @staticmethod
    def _parent_strategy(strategy_name: str) -> str:
        if strategy_name.startswith("p14_tree_branch_vote"):
            return "p14_tree_branch_vote"
        return strategy_name

    @staticmethod
    def _extract_ticket(text: str) -> str:
        t = text.lower()
        if "<input_data>" in t and "</input_data>" in t:
            return t.split("<input_data>", 1)[1].split("</input_data>", 1)[0].strip()
        if "<ticket>" in t and "</ticket>" in t:
            return t.split("<ticket>", 1)[1].split("</ticket>", 1)[0].strip()
        if "[input data]" in t:
            chunk = t.split("[input data]", 1)[1]
            return chunk.split("[output]", 1)[0].strip()
        for marker in ("message:", "ticket:"):
            if marker in t:
                chunk = t.rsplit(marker, 1)[-1]
                for end in ("\nreturn", "\nallowed labels:", "\noutput:"):
                    chunk = chunk.split(end, 1)[0]
                return chunk.strip()
        return t

    @classmethod
    def _heuristic_predict(cls, text: str) -> str:
        """Fallback used for free-form Playground tickets without benchmark metadata."""
        msg = cls._extract_ticket(text)
        rules = [
            ("cancellation", ["cancel", "terminate", "not renew", "future renew", "end my membership"]),
            ("billing", ["charged", "invoice", "refund", "payment", "credit card", "statement", "transaction"]),
            ("api", [" api", "api ", "sdk", "endpoint", "api key", "rate limit", "webhook"]),
            ("upgrade", ["upgrade", "enterprise", "higher plan", "premium", "more seats", "capacity"]),
            ("technical", ["crash", "freez", "white screen", "error", "login", "not working", "stopped working"]),
            ("complaint", ["disappointed", "frustrat", "unacceptable", "terrible", "complain", "escalat", "manager"]),
        ]
        for label, keywords in rules:
            if any(keyword in msg for keyword in keywords):
                return label
        return "complaint"

    def _sampling_penalty(self) -> float:
        penalty = 0.0
        if self.temperature is not None and self.temperature > 0.2:
            penalty += min(0.13, (self.temperature - 0.2) * 0.07)
        if self.top_p is not None and self.top_p < 0.75:
            penalty += min(0.07, (0.75 - self.top_p) * 0.18)
        if self.top_k is not None:
            if self.top_k < 15:
                penalty += 0.035
            elif self.top_k > 80:
                penalty += 0.02
        return penalty

    @staticmethod
    def _custom_prompt_features(payload: PromptPayload) -> dict[str, bool]:
        text = f"{payload.instructions or ''}\n{payload.input_text}".lower()
        return {
            "definitions": any(k in text for k in ["category definitions", "kategória", "billing:", "technical:"]),
            "few_shot": any(k in text for k in ["examples:", "példák:", "example:", "category:"]),
            "constraints": any(k in text for k in ["rules:", "constraints", "szabály", "exactly one", "pontosan egy"]),
            "decision_policy": any(k in text for k in ["decision policy", "döntési", "primary intent", "elsődleges szándék"]),
            "delimited": any(k in text for k in ["<ticket>", "<input_data>", "[input data]", "<adat>"]),
            "persona": any(k in text for k in ["you are", "te egy", "szakértő", "specialist"]),
            "json": payload.output_mode == "json" or "json" in text,
            "structured": payload.structured_output,
            "reasoning": bool(payload.reasoning_effort) or any(k in text for k in ["reasoning", "döntési policy", "internally"]),
        }

    def _custom_base_accuracy(self, payload: PromptPayload, case_type: str) -> float:
        f = self._custom_prompt_features(payload)
        base = 0.73
        base += 0.035 if f["definitions"] else 0.0
        base += 0.045 if f["few_shot"] else 0.0
        base += 0.035 if f["constraints"] else 0.0
        base += 0.045 if f["decision_policy"] else 0.0
        base += 0.018 if f["persona"] else 0.0
        base += 0.02 if f["json"] else 0.0
        base += 0.025 if f["structured"] else 0.0
        base += 0.045 if f["reasoning"] else 0.0
        if f["delimited"] and case_type in {"prompt_injection", "long_context", "resolved_history", "code_log_noise", "label_word_attack"}:
            base += 0.10
        if f["decision_policy"] and case_type in {"ambiguous_boundary", "multi_intent_primary", "resolved_history", "primary_last", "primary_first", "conditional_distractor", "double_negation"}:
            base += 0.07
        if f["few_shot"] and case_type in {"implicit_request", "noisy_typo", "ambiguous_boundary", "conditional_distractor", "double_negation"}:
            base += 0.05
        return base

    def _semantic_probability(self, payload: PromptPayload) -> float:
        strategy = self._parent_strategy(payload.strategy_name)
        case_type = str(payload.metadata.get("case_type") or "easy_clear")
        if strategy.startswith("custom_"):
            base = self._custom_base_accuracy(payload, case_type)
        else:
            base = self._BASE_ACCURACY.get(strategy, 0.78)
        base += self._CASE_ADJUSTMENT.get(case_type, 0.0)
        base += self._CASE_BONUS.get(strategy, {}).get(case_type, 0.0)

        # Branch specialization makes the branch-and-vote strategy genuinely
        # diverse rather than three identical calls.
        if payload.strategy_name.endswith("branch1") and case_type in {"easy_clear", "noisy_typo"}:
            base += 0.05
        elif payload.strategy_name.endswith("branch2") and case_type in {"ambiguous_boundary", "multi_intent_primary", "resolved_history"}:
            base += 0.06
        elif payload.strategy_name.endswith("branch3") and case_type in {"ambiguous_boundary", "multi_intent_primary", "prompt_injection"}:
            base += 0.08

        base -= self._sampling_penalty()
        return max(0.35, min(0.995, base))

    def _format_probability(self, payload: PromptPayload) -> float:
        strategy = self._parent_strategy(payload.strategy_name)
        if strategy.startswith("custom_"):
            f = self._custom_prompt_features(payload)
            probability = 0.93 + (0.035 if f["constraints"] else 0.0) + (0.02 if f["json"] else 0.0) + (0.015 if f["structured"] else 0.0)
        else:
            probability = self._FORMAT_VALIDITY.get(strategy, 0.96)
        if self.temperature is not None and self.temperature > 0.7 and not payload.structured_output:
            probability -= min(0.08, (self.temperature - 0.7) * 0.05)
        return max(0.70, min(1.0, probability))

    @staticmethod
    def _approx_tokens(text: str) -> int:
        # A deliberately provider-neutral approximation used only in mock mode.
        units = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        return max(1, int(round(len(units) * 1.28)))

    def _simulated_latency(self, payload: PromptPayload, input_tokens: int, output_tokens: int) -> float:
        strategy = self._parent_strategy(payload.strategy_name)
        overhead = {
            "p0_zero_shot": 0.03,
            "p3_few_shot": 0.06,
            "p5_decision_policy": 0.08,
            "p7_structured_output": 0.08,
            "p12_contrastive_few_shot": 0.10,
            "p13_reasoning_model": 0.32,
            "p14_tree_branch_vote": 0.10,
            "p15_grammar_constrained": 0.09,
            "p16_full_advanced_template": 0.14,
        }.get(strategy, 0.05)
        jitter = self._stable_unit(payload.strategy_name, payload.input_text, "latency") * 0.06
        return round(0.18 + input_tokens * 0.00042 + output_tokens * 0.0025 + overhead + jitter, 4)

    def classify(self, payload: PromptPayload) -> LLMResponse:
        if payload.output_mode == "text":
            prompt_text = f"{payload.instructions or ''}\n{payload.input_text}"
            input_tokens = self._approx_tokens(prompt_text)
            topic = payload.input_text.strip().replace("\n", " ")[:180]
            raw = (
                "[MOCK TEXT GENERATION] This is a deterministic offline simulation, not a real LLM response. "
                f"The requested topic/input was: {topic}. "
                "Connect Gemini, Groq, OpenRouter, OpenAI or Ollama to generate real free-form text."
            )
            output_tokens = self._approx_tokens(raw)
            return LLMResponse(
                raw_output=raw,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                latency_seconds=self._simulated_latency(payload, input_tokens, output_tokens),
                model=self.model,
                provider=self.provider,
                token_source="estimated_mock",
                latency_source="simulated_mock",
            )

        metadata = payload.metadata or {}
        true_label = str(metadata.get("true_label") or "")
        secondary_label = str(metadata.get("secondary_label") or "")
        case_type = str(metadata.get("case_type") or "free_form")

        # Free-form playground input has no ground-truth metadata, so use the
        # transparent keyword fallback rather than fabricating a known answer.
        if true_label not in self._CONFUSIONS or case_type not in self._CASE_ADJUSTMENT:
            predicted = self._heuristic_predict(payload.input_text)
        else:
            correct_probability = self._semantic_probability(payload)
            semantic_draw = self._stable_unit(payload.strategy_name, metadata.get("scenario_id"), payload.input_text, "semantic")
            if semantic_draw <= correct_probability:
                predicted = true_label
            else:
                predicted = secondary_label if secondary_label in self._CONFUSIONS and case_type in {
                    "ambiguous_boundary", "multi_intent_primary", "prompt_injection", "resolved_history",
                    "primary_last", "primary_first", "conditional_distractor", "label_word_attack", "double_negation"
                } else self._CONFUSIONS[true_label]

        format_draw = self._stable_unit(payload.strategy_name, metadata.get("scenario_id"), payload.input_text, "format")
        format_valid = format_draw <= self._format_probability(payload)

        if payload.output_mode == "json":
            if format_valid:
                raw = json.dumps({"label": predicted})
            else:
                raw = f"{{label: '{predicted}'}}"  # intentionally malformed JSON
        else:
            if format_valid:
                raw = predicted
            else:
                raw = f"The routing category is {predicted}."

        prompt_text = f"{payload.instructions or ''}\n{payload.input_text}"
        input_tokens = self._approx_tokens(prompt_text)
        output_tokens = self._approx_tokens(raw)
        if payload.reasoning_effort:
            # Simulate hidden reasoning-token overhead without exposing any CoT.
            output_tokens += 24
        latency = self._simulated_latency(payload, input_tokens, output_tokens)
        return LLMResponse(
            raw_output=raw,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            latency_seconds=latency,
            model=self.model,
            provider=self.provider,
            token_source="estimated_mock",
            latency_source="simulated_mock",
        )
