from travel_agent.agent import PlanThenExecuteTravelAgent
from travel_agent.presets import CATEGORY_LABELS, PRESET_QUESTIONS


def test_preset_catalogue_has_30_unique_bilingual_scenarios():
    assert len(PRESET_QUESTIONS) == 30
    assert len({item["id"] for item in PRESET_QUESTIONS}) == 30
    assert set(item["category"] for item in PRESET_QUESTIONS) <= set(CATEGORY_LABELS)
    for item in PRESET_QUESTIONS:
        assert item["hu"].strip()
        assert item["en"].strip()
        assert item["expected_tools"]
        assert item["difficulty"] in {"basic", "intermediate", "advanced"}


def test_all_bilingual_presets_execute_expected_tool_route(monkeypatch):
    monkeypatch.setenv("TRAVEL_DATA_MODE", "local")
    for item in PRESET_QUESTIONS:
        for language in ("hu", "en"):
            run = PlanThenExecuteTravelAgent(language=language).run(item[language])
            assert run.tool_names == item["expected_tools"], (item["id"], language, run.tool_names)
            assert run.trace
            assert all(call.success for call in run.trace), (item["id"], language, run.trace)


def test_all_bilingual_presets_are_supported_by_saved_ml_router(monkeypatch):
    from travel_agent.agent import MLRouterTravelAgent

    monkeypatch.setenv("TRAVEL_DATA_MODE", "local")
    for language in ("hu", "en"):
        agent = MLRouterTravelAgent(language=language)
        for item in PRESET_QUESTIONS:
            run = agent.run(item[language])
            assert run.tool_names == item["expected_tools"], (item["id"], language, run.tool_names)
            assert all(call.success for call in run.trace), (item["id"], language, run.trace)
