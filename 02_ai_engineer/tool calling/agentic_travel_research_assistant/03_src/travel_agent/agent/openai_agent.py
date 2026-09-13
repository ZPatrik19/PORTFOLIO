from __future__ import annotations

import json
import time
from travel_agent.i18n import language_instruction, normalize_language, t
from travel_agent.models import AgentRun, ToolCallRecord
from travel_agent.tools import ToolRegistry
from .prompts import SYSTEM_PROMPT


class OpenAITravelAgent:
    """Real LLM-driven orchestration using the OpenAI Responses API.

    The model chooses tools and arguments. Python remains responsible for validating
    and executing functions. Tool outputs are returned to the model until it emits a
    final textual answer or the safety step limit is reached.
    """

    def __init__(self, api_key: str | None, model: str, registry: ToolRegistry | None = None, max_steps: int = 8, client=None, language: str = "en") -> None:
        if client is None:
            if not api_key:
                raise ValueError("api_key is required when no client is injected")
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        self.client = client
        self.model = model
        self.registry = registry or ToolRegistry()
        self.max_steps = max_steps
        self.language = normalize_language(language)

    def run(self, query: str) -> AgentRun:
        started = time.perf_counter()
        trace: list[ToolCallRecord] = []

        instructions = f"{SYSTEM_PROMPT}\n\n{language_instruction(self.language)}"
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=query,
            tools=self.registry.schemas(),
            tool_choice="auto",
            parallel_tool_calls=True,
        )

        for step in range(1, self.max_steps + 1):
            function_calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
            if not function_calls:
                return AgentRun(
                    query=query,
                    answer=response.output_text or "No textual answer returned.",
                    trace=trace,
                    total_latency_ms=(time.perf_counter() - started) * 1000,
                    model=self.model,
                    metadata={"methodology": "openai_direct", "language": self.language},
                )

            tool_outputs = []
            for call in function_calls:
                try:
                    arguments = json.loads(call.arguments)
                except json.JSONDecodeError as exc:
                    arguments = {}
                    output, latency_ms, success, error = {"error": f"Invalid JSON arguments: {exc}"}, 0.0, False, str(exc)
                else:
                    output, latency_ms, success, error = self.registry.execute(call.name, arguments)

                trace.append(ToolCallRecord(
                    step=step,
                    name=call.name,
                    arguments=arguments,
                    output=output,
                    latency_ms=latency_ms,
                    success=success,
                    error=error,
                ))
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(output, ensure_ascii=False),
                })

            response = self.client.responses.create(
                model=self.model,
                instructions=instructions,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=self.registry.schemas(),
                tool_choice="auto",
                parallel_tool_calls=True,
            )

        return AgentRun(
            query=query,
            answer=t("stopped", self.language),
            trace=trace,
            total_latency_ms=(time.perf_counter() - started) * 1000,
            model=self.model,
            metadata={"methodology": "openai_direct", "language": self.language},
        )
