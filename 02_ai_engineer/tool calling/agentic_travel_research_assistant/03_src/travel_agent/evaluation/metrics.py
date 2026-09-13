from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from math import isclose
from statistics import mean
from typing import Any


def _normalize(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _arg_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6)
    return _normalize(expected) == _normalize(actual)


def score_case(expected_calls: list[dict], actual_calls: list[dict], answer: str) -> dict[str, Any]:
    expected_names = [x["name"] for x in expected_calls]
    actual_names = [x["name"] for x in actual_calls]
    exp_counter, act_counter = Counter(expected_names), Counter(actual_names)

    true_positive = sum((exp_counter & act_counter).values())
    total_expected = sum(exp_counter.values())
    total_actual = sum(act_counter.values())
    tool_precision = true_positive / total_actual if total_actual else (1.0 if not total_expected else 0.0)
    tool_recall = true_positive / total_expected if total_expected else (1.0 if not total_actual else 0.0)
    tool_f1 = 2 * tool_precision * tool_recall / (tool_precision + tool_recall) if (tool_precision + tool_recall) else 0.0
    tool_set_exact = float(exp_counter == act_counter)
    unnecessary = max(total_actual - true_positive, 0)
    unnecessary_rate = unnecessary / total_actual if total_actual else 0.0

    # Pair expected and actual calls of the same name in order, then score every expected argument.
    actual_by_name: dict[str, list[dict]] = {}
    for call in actual_calls:
        actual_by_name.setdefault(call["name"], []).append(call.get("arguments", {}))
    used_idx = Counter()
    arg_correct = 0
    arg_total = 0
    for exp in expected_calls:
        name = exp["name"]
        idx = used_idx[name]
        used_idx[name] += 1
        actual_args = actual_by_name.get(name, [])
        candidate = actual_args[idx] if idx < len(actual_args) else {}
        for key, exp_value in exp.get("arguments", {}).items():
            arg_total += 1
            if key in candidate and _arg_match(exp_value, candidate[key]):
                arg_correct += 1
    argument_accuracy = arg_correct / arg_total if arg_total else 1.0

    all_success = all(call.get("success", False) for call in actual_calls)
    task_success = float(tool_set_exact == 1.0 and argument_accuracy == 1.0 and all_success and bool(answer.strip()))
    return {
        "tool_selection_accuracy": tool_set_exact,
        "tool_precision": tool_precision,
        "tool_recall": tool_recall,
        "tool_f1": tool_f1,
        "argument_accuracy": argument_accuracy,
        "task_success": task_success,
        "unnecessary_tool_call_rate": unnecessary_rate,
        "num_calls": total_actual,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [
        "tool_selection_accuracy", "tool_precision", "tool_recall", "tool_f1",
        "argument_accuracy", "task_success", "unnecessary_tool_call_rate", "num_calls", "latency_ms",
    ]
    out = {name: mean(float(row[name]) for row in rows) if rows else 0.0 for name in metric_names}
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    if latencies:
        p95_index = min(len(latencies) - 1, max(0, int(round(0.95 * len(latencies) + 0.5)) - 1))
        out["p95_latency_ms"] = latencies[p95_index]
    else:
        out["p95_latency_ms"] = 0.0
    out["cases"] = float(len(rows))
    return out
