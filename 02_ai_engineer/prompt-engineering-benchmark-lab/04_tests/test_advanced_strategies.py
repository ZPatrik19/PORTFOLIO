"""EN: Advanced prompt-strategy contracts, branch-and-vote behavior, structured output, prompt isolation, and provider capability checks.

HU: Az advanced promptstratégiák szerződéseit, a branch-and-vote működést, a strukturált kimenetet, a prompt-izolációt és a provider-képességeket ellenőrzi.
"""

from pathlib import Path

import pytest

from prompt_benchmark.llm.client import MockLLMClient
from prompt_benchmark.llm.factory import PROVIDER_CAPABILITIES, create_llm_client
from prompt_benchmark.prompts import get_strategy, list_strategies


def test_advanced_strategy_count_and_names():
    """EN: Verifies that the advanced strategy registry exposes the expected number and stable strategy identifiers.

    HU: Ellenőrzi, hogy az advanced strategy registry a várt számú és stabil azonosítójú promptstratégiát adja vissza.
    """
    names = list_strategies()
    assert len(names) == 17
    assert "p14_tree_branch_vote" in names
    assert "p16_full_advanced_template" in names


def test_tree_strategy_has_three_independent_branches():
    """EN: Verifies that the Tree-of-Thought-inspired branch-and-vote strategy executes three independent decision branches rather than a single hidden reasoning trace.

    HU: Ellenőrzi, hogy a Tree-of-Thought ihletésű branch-and-vote stratégia három független döntési ágat futtat, nem egyetlen rejtett gondolatmenetet.
    """
    strategy = get_strategy("p14_tree_branch_vote")
    branches = strategy.build_branches("Please cancel my subscription")
    assert len(branches) == 3
    assert len({branch.instructions for branch in branches}) == 3


def test_full_template_uses_structured_output():
    """EN: Verifies that the full advanced template requests structured output so downstream parsing has a stable contract.

    HU: Ellenőrzi, hogy a teljes advanced template strukturált kimenetet kér, így a downstream parsing stabil szerződésre épül.
    """
    payload = get_strategy("p16_full_advanced_template").build("invoice issue")
    assert payload.structured_output is True
    assert payload.output_mode == "json"
    assert "<input_data>" in payload.input_text


def test_mock_extracts_delimited_ticket_not_prompt_definitions():
    """EN: Prevents the mock classifier from accidentally classifying label definitions or instructions instead of the delimited user ticket.

    HU: Megakadályozza, hogy a mock classifier a label-definíciókat vagy instrukciókat osztályozza a delimitált ticket helyett.
    """
    client = MockLLMClient()
    payload = get_strategy("p16_full_advanced_template").build("Please cancel my subscription")
    response = client.classify(payload)
    assert '"cancellation"' in response.raw_output


def test_provider_capability_matrix_exposes_top_k_support():
    """EN: Checks that provider capability metadata clearly states whether top_k is supported.

    HU: Ellenőrzi, hogy a provider-képességmátrix egyértelműen jelzi a top_k támogatását.
    """
    assert PROVIDER_CAPABILITIES["ollama"]["top_k"] is True
    assert PROVIDER_CAPABILITIES["gemini"]["top_k"] is True
    assert PROVIDER_CAPABILITIES["groq"]["top_k"] is False
    assert PROVIDER_CAPABILITIES["openai"]["top_k"] is False


def test_unsupported_groq_top_k_fails_before_api_call():
    """EN: Ensures unsupported Groq top_k configuration fails locally before any external request is sent.

    HU: Biztosítja, hogy nem támogatott Groq top_k esetén a rendszer még API-hívás előtt lokálisan hibázzon.
    """
    with pytest.raises(ValueError, match="does not support top_k"):
        create_llm_client("groq", {"max_output_tokens": 64, "random_seed": 42}, overrides={"top_k": 40})


def test_prompt_template_file_exists():
    """EN: Ensures the reusable prompt-template resource required by the UI and documentation is packaged in the repository.

    HU: Ellenőrzi, hogy a UI és dokumentáció által használt újrahasznosítható prompt template fájl megtalálható a repositoryban.
    """
    assert Path("configs/prompts/templates/full_prompt_template.yaml").exists()
