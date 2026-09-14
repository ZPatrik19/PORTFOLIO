from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any

import pandas as pd
from tqdm import tqdm

from prompt_benchmark.data.benchmark_suites import representative_stratified_subset
from prompt_benchmark.data.validation import validate_benchmark_frame
from prompt_benchmark.evaluation.parsing import ParsedPrediction, parse_prediction
from prompt_benchmark.llm.client import BaseLLMClient
from prompt_benchmark.llm.schemas import LLMResponse
from prompt_benchmark.prompts.base import PromptStrategy
from prompt_benchmark.utils.pricing import calculate_estimated_cost

LOGGER = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "sample_id", "text", "true_label", "case_type", "difficulty",
    "secondary_label", "scenario_id", "scenario_notes",
    "predicted_label", "correct", "raw_response", "valid_output", "valid_json",
    "input_tokens", "output_tokens", "total_tokens", "token_source",
    "latency_seconds", "latency_source", "estimated_cost_usd",
    "provider", "model", "prompt_strategy",
    "temperature", "top_p", "top_k", "reasoning_effort", "branch_count",
    "timestamp", "error",
]


def _execute_strategy(
    strategy: PromptStrategy,
    ticket: str,
    client: BaseLLMClient,
    sample_context: dict[str, object] | None = None,
) -> tuple[LLMResponse, ParsedPrediction, str | None, int]:
    """Execute either a normal prompt or a branch-and-vote strategy.

    Benchmark metadata is attached to ``PromptPayload.metadata``. Real provider
    adapters ignore it; the deterministic mock simulator uses it to model
    prompt-sensitive behavior across synthetic scenario types.
    """
    context = dict(sample_context or {})
    if hasattr(strategy, "build_branches"):
        payloads = [replace(p, metadata=context) for p in strategy.build_branches(ticket)]  # type: ignore[attr-defined]
        responses: list[LLMResponse] = []
        parsed_items: list[ParsedPrediction] = []
        for payload in payloads:
            response = client.classify(payload)
            responses.append(response)
            parsed_items.append(parse_prediction(response.raw_output, payload.output_mode) if not response.error else parse_prediction("", payload.output_mode))

        valid_labels = [item.label for item in parsed_items if item.valid_output and item.label]
        if valid_labels:
            counts = Counter(valid_labels)
            max_count = max(counts.values())
            tied = {label for label, count in counts.items() if count == max_count}
            voted = next(label for label in valid_labels if label in tied)
            parsed = ParsedPrediction(voted, True, None)
        else:
            voted = None
            parsed = ParsedPrediction(None, False, None)

        raw = json.dumps(
            {
                "branch_predictions": [item.label for item in parsed_items],
                "branch_outputs": [response.raw_output for response in responses],
                "majority_vote": voted,
            },
            ensure_ascii=False,
        )
        errors = [response.error for response in responses if response.error]
        combined = LLMResponse(
            raw_output=raw,
            input_tokens=sum(r.input_tokens for r in responses),
            output_tokens=sum(r.output_tokens for r in responses),
            total_tokens=sum(r.total_tokens for r in responses),
            latency_seconds=sum(r.latency_seconds for r in responses),
            model=client.model,
            provider=client.provider,
            error=" | ".join(errors) if errors else None,
            token_source=responses[0].token_source if responses else "unknown",
            latency_source=responses[0].latency_source if responses else "unknown",
        )
        return combined, parsed, None, len(payloads)

    payload = replace(strategy.build(ticket), metadata=context)
    response = client.classify(payload)
    parsed = parse_prediction(response.raw_output, payload.output_mode) if not response.error else parse_prediction("", payload.output_mode)
    return response, parsed, payload.reasoning_effort, 1


def run_strategy(
    benchmark: pd.DataFrame,
    strategy: PromptStrategy,
    client: BaseLLMClient,
    output_path: str | Path,
    input_price_per_million: float,
    output_price_per_million: float,
    limit: int | None = None,
    force: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Run one prompt strategy with resumable per-request checkpointing."""
    validate_benchmark_frame(benchmark)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work = representative_stratified_subset(benchmark, limit) if limit else benchmark.copy()
    LOGGER.info("Starting benchmark strategy=%s provider=%s model=%s rows=%s output=%s", strategy.name, client.provider, client.model, len(work), output_path)

    if force and output_path.exists():
        output_path.unlink()

    existing = pd.DataFrame(columns=RESULT_COLUMNS)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if len(existing):
            existing_models = set(existing["model"].astype(str))
            existing_strategies = set(existing["prompt_strategy"].astype(str))
            existing_providers = set(existing.get("provider", pd.Series(["unknown"])).astype(str))
            if existing_models != {client.model} or existing_strategies != {strategy.name} or existing_providers != {client.provider}:
                raise ValueError(
                    f"Cached file {output_path} belongs to provider/model/strategy "
                    f"{existing_providers}/{existing_models}/{existing_strategies}, not "
                    f"{client.provider}/{client.model}/{strategy.name}. Use --force to replace it."
                )

            settings = client.generation_settings()
            for col in ("temperature", "top_p", "top_k"):
                if col not in existing.columns:
                    continue
                cached_values = existing[col].dropna().unique().tolist()
                if len(cached_values) > 1:
                    raise ValueError(f"Cached file {output_path} contains multiple {col} values; use a dedicated sweep output path.")
                cached = cached_values[0] if cached_values else None
                current = settings.get(col)

                def _setting_equal(left: object, right: object) -> bool:
                    if left is None or right is None:
                        return left is None and right is None
                    try:
                        return abs(float(left) - float(right)) <= 1e-12
                    except (TypeError, ValueError):
                        return str(left) == str(right)

                if not _setting_equal(cached, current):
                    raise ValueError(
                        f"Cached {col}={cached} differs from current {col}={current}. "
                        "Use --force or a separate sweep output path."
                    )

            current_identity = work[["sample_id", "text", "true_label"]].copy()
            current_identity["sample_id"] = current_identity["sample_id"].astype(str)
            cached_identity = existing[["sample_id", "text", "true_label"]].copy()
            cached_identity["sample_id"] = cached_identity["sample_id"].astype(str)
            current_ids = set(current_identity["sample_id"])
            cached_ids = set(cached_identity["sample_id"])
            extra_cached_ids = cached_ids - current_ids
            if extra_cached_ids:
                preview = ", ".join(sorted(extra_cached_ids)[:5])
                raise ValueError(
                    f"Cached file {output_path} contains rows outside the current benchmark selection "
                    f"(for example: {preview}). Use --force or a separate output path when changing the run profile/limit."
                )
            overlap = cached_identity.merge(current_identity, on="sample_id", how="inner", suffixes=("_cached", "_current"))
            mismatch = overlap[(overlap["text_cached"].astype(str) != overlap["text_current"].astype(str)) | (overlap["true_label_cached"].astype(str) != overlap["true_label_current"].astype(str))]
            if not mismatch.empty:
                raise ValueError(f"Cached file {output_path} points to different data. Use --force.")

            # Cost is derived from token usage and current pricing, so refresh it
            # even when the expensive model prediction itself is reused.
            existing["estimated_cost_usd"] = [
                calculate_estimated_cost(int(i), int(o), input_price_per_million, output_price_per_million)
                for i, o in zip(existing["input_tokens"], existing["output_tokens"])
            ]
            existing.to_csv(output_path, index=False)

    completed = set(existing["sample_id"].astype(str)) if len(existing) else set()
    settings = client.generation_settings()
    total = len(work)
    if progress_callback is not None:
        progress_callback({
            "event": "start",
            "strategy": strategy.name,
            "completed": len(completed & set(work["sample_id"].astype(str))),
            "total": total,
            "provider": client.provider,
            "model": client.model,
        })

    processed_now = 0
    for row in tqdm(work.itertuples(index=False), total=len(work), desc=f"{client.provider}:{strategy.name}"):
        sample_id = str(row.sample_id)
        if sample_id in completed:
            continue

        sample_context = {
            "true_label": str(row.true_label),
            "case_type": str(getattr(row, "case_type", "")),
            "difficulty": str(getattr(row, "difficulty", "")),
            "secondary_label": str(getattr(row, "secondary_label", "")),
            "scenario_id": str(getattr(row, "scenario_id", sample_id)),
            "scenario_notes": str(getattr(row, "scenario_notes", "")),
        }
        response, parsed, reasoning_effort, branch_count = _execute_strategy(
            strategy, str(row.text), client, sample_context=sample_context
        )
        cost = calculate_estimated_cost(response.input_tokens, response.output_tokens, input_price_per_million, output_price_per_million)
        result_row = {
            "sample_id": sample_id,
            "text": str(row.text),
            "true_label": str(row.true_label),
            "case_type": sample_context["case_type"],
            "difficulty": sample_context["difficulty"],
            "secondary_label": sample_context["secondary_label"],
            "scenario_id": sample_context["scenario_id"],
            "scenario_notes": sample_context["scenario_notes"],
            "predicted_label": parsed.label,
            "correct": parsed.label == str(row.true_label),
            "raw_response": response.raw_output,
            "valid_output": parsed.valid_output,
            "valid_json": parsed.valid_json,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "token_source": response.token_source,
            "latency_seconds": response.latency_seconds,
            "latency_source": response.latency_source,
            "estimated_cost_usd": cost,
            "provider": response.provider,
            "model": response.model,
            "prompt_strategy": strategy.name,
            "temperature": settings.get("temperature"),
            "top_p": settings.get("top_p"),
            "top_k": settings.get("top_k"),
            "reasoning_effort": reasoning_effort,
            "branch_count": branch_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": response.error,
        }
        pd.DataFrame([result_row], columns=RESULT_COLUMNS).to_csv(output_path, mode="a", header=not output_path.exists(), index=False)
        completed.add(sample_id)
        processed_now += 1
        if progress_callback is not None:
            progress_callback({
                "event": "sample",
                "strategy": strategy.name,
                "sample_id": sample_id,
                "completed": len(completed & set(work["sample_id"].astype(str))),
                "processed_now": processed_now,
                "total": total,
                "row": result_row,
            })

    final = pd.read_csv(output_path) if output_path.exists() else existing
    if not final.empty:
        wanted_ids = set(work["sample_id"].astype(str))
        final = final[final["sample_id"].astype(str).isin(wanted_ids)].copy().reset_index(drop=True)
    LOGGER.info("Completed benchmark strategy=%s rows=%s output=%s", strategy.name, len(final), output_path)
    if progress_callback is not None:
        progress_callback({
            "event": "complete",
            "strategy": strategy.name,
            "completed": min(len(final), total),
            "total": total,
            "rows": len(final),
        })
    return final
