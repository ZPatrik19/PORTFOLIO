from pathlib import Path

from travel_agent.models import AgentRun, ToolCallRecord
from travel_agent.usage import UsageStore


def _sample_run() -> AgentRun:
    return AgentRun(
        query="3 napra megyek Bécsbe. Kell esernyő?",
        answer="A következő napokra időjárási adatokat találtam.",
        trace=[
            ToolCallRecord(
                step=1,
                name="get_weather",
                arguments={"city": "Vienna", "days": 3},
                output={"days": [{"date": "2026-09-13", "condition": "rain"}]},
                latency_ms=12.5,
                success=True,
            )
        ],
        total_latency_ms=21.0,
        model="offline",
        metadata={"methodology": "plan_execute"},
    )


def test_usage_store_persists_runs_and_statistics(tmp_path: Path):
    db = tmp_path / "usage.sqlite3"
    store = UsageStore(db)
    interaction_id = store.record_run(
        _sample_run(),
        methodology="plan_execute",
        language="hu",
        data_mode="local",
        source="custom",
    )
    assert interaction_id == 1
    assert store.count() == 1

    summary = store.summary()
    assert summary["total_questions"] == 1
    assert summary["total_tool_calls"] == 1
    assert summary["custom_questions"] == 1
    assert summary["preset_questions"] == 0
    assert summary["tool_success_rate"] == 1.0

    tools = store.tool_usage()
    assert tools.iloc[0]["tool_name"] == "get_weather"
    assert int(tools.iloc[0]["calls"]) == 1

    recent = store.recent_interactions()
    assert recent.iloc[0]["question"].startswith("3 napra")
    assert recent.iloc[0]["answer"].startswith("A következő")

    # Persistence across a fresh store instance.
    reloaded = UsageStore(db)
    assert reloaded.count() == 1


def test_usage_store_clear_removes_history(tmp_path: Path):
    store = UsageStore(tmp_path / "usage.sqlite3")
    store.record_run(
        _sample_run(),
        methodology="plan_execute",
        language="hu",
        data_mode="local",
        source="preset",
        preset_id="weather_basic_01",
    )
    assert store.count() == 1
    store.clear()
    assert store.count() == 0
    assert store.tool_usage().empty


def test_usage_store_advanced_statistics(tmp_path: Path):
    store = UsageStore(tmp_path / "usage.sqlite3")
    first = _sample_run()
    store.record_run(first, methodology="plan_execute", language="hu", data_mode="local", source="custom")

    second = AgentRun(
        query="Vienna hotel and restaurant",
        answer="Found options.",
        trace=[
            ToolCallRecord(1, "search_hotels", {"city": "Vienna"}, {"results": []}, 8.0, True),
            ToolCallRecord(2, "search_restaurants", {"city": "Vienna"}, {"results": []}, 10.0, False, "provider_error"),
        ],
        total_latency_ms=40.0,
        model="offline",
        metadata={"methodology": "ml_router"},
    )
    store.record_run(second, methodology="ml_router", language="en", data_mode="auto", source="preset", preset_id="x")

    summary = store.summary()
    assert summary["total_questions"] == 2
    assert summary["total_tool_calls"] == 3
    assert summary["multi_tool_rate"] == 0.5
    assert summary["no_tool_rate"] == 0.0
    assert summary["run_latency_p95_ms"] >= summary["run_latency_p50_ms"]
    assert summary["unique_question_rate"] == 1.0

    sequences = store.tool_sequences()
    assert "get_weather" in sequences["tool_sequence"].tolist()
    assert any("search_hotels" in x for x in sequences["tool_sequence"].tolist())

    destinations = store.destination_usage()
    vienna = destinations.loc[destinations["city"] == "Vienna", "tool_calls"].iloc[0]
    assert int(vienna) == 3

    raw_calls = store.tool_call_history()
    assert len(raw_calls) == 3
    assert {"tool_name", "latency_ms", "methodology", "success"}.issubset(raw_calls.columns)

    errors = store.error_breakdown()
    assert errors.iloc[0]["tool_name"] == "search_restaurants"
    assert errors.iloc[0]["error"] == "provider_error"

    assert set(store.dimension_usage("language")["value"]) == {"hu", "en"}
    assert set(store.dimension_usage("data_mode")["value"]) == {"local", "auto"}
