"""EN: Custom prompt rendering, validation, persistence, and prompt-sensitive mock behavior.

HU: A saját promptok renderelését, validációját, mentését/betöltését és promptérzékeny mock viselkedését ellenőrzi.
"""

from pathlib import Path

import pytest

from prompt_benchmark.prompts.custom import CustomPromptStrategy, load_custom_prompt, save_custom_prompt


def test_custom_prompt_renders_ticket():
    """EN: Checks that a custom prompt correctly interpolates the ticket payload.

    HU: Ellenőrzi, hogy a custom prompt helyesen behelyettesíti a ticket tartalmát.
    """
    strategy = CustomPromptStrategy(
        display_name="Saját teszt",
        system_prompt="Te egy osztályozó vagy.",
        user_template="Ticket: {ticket}\nCsak címkét adj.",
    )
    payload = strategy.build("Please cancel")
    assert "Please cancel" in payload.input_text
    assert strategy.name.startswith("custom_")


def test_custom_prompt_requires_ticket_placeholder():
    """EN: Rejects custom templates that cannot inject the benchmark ticket, preventing meaningless runs.

    HU: Elutasítja az olyan custom template-et, amelybe a benchmark ticket nem illeszthető be.
    """
    with pytest.raises(ValueError):
        CustomPromptStrategy("bad", "", "Nincs helyőrző")


def test_custom_prompt_save_and_load(tmp_path: Path):
    """EN: Verifies lossless persistence and reload of custom prompt presets.

    HU: Ellenőrzi a custom prompt preset veszteségmentes mentését és visszatöltését.
    """
    strategy = CustomPromptStrategy(
        display_name="JSON saját",
        system_prompt="system",
        user_template='Ticket: {ticket}\nReturn JSON.',
        output_mode="json",
        structured_output=True,
    )
    path = save_custom_prompt(strategy, tmp_path)
    loaded = load_custom_prompt(path)
    assert loaded.display_name == strategy.display_name
    assert loaded.output_mode == "json"
    assert loaded.structured_output is True


def test_mock_custom_prompt_detects_advanced_components():
    """EN: Ensures the prompt-sensitive mock simulator recognizes advanced components in user-defined prompts.

    HU: Biztosítja, hogy a promptérzékeny mock szimulátor felismeri a felhasználói advanced prompt komponenseket.
    """
    from prompt_benchmark.llm.client import MockLLMClient

    client = MockLLMClient()
    basic = CustomPromptStrategy(
        display_name="basic_custom",
        system_prompt="",
        user_template="Classify: {ticket}",
    ).build("Please cancel")
    advanced = CustomPromptStrategy(
        display_name="advanced_custom",
        system_prompt="You are a specialist.",
        user_template=(
            "Category definitions: billing, cancellation, api, technical, complaint, upgrade.\n"
            "Rules: select exactly one. Decision policy: choose the primary intent internally.\n"
            "<ticket>{ticket}</ticket>\nReturn JSON."
        ),
        output_mode="json",
        structured_output=True,
        reasoning_effort="medium",
    ).build("Please cancel")
    basic = basic.__class__(**{**basic.__dict__, "metadata": {"true_label": "cancellation", "case_type": "ambiguous_boundary", "scenario_id": "x"}})
    advanced = advanced.__class__(**{**advanced.__dict__, "metadata": {"true_label": "cancellation", "case_type": "ambiguous_boundary", "scenario_id": "x"}})
    assert client._semantic_probability(advanced) > client._semantic_probability(basic)
