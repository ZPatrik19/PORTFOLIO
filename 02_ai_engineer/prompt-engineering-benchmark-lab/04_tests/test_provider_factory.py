"""EN: Provider registry completeness and credential-free mock client creation.

HU: A provider-registry teljességét és a credential nélküli mock kliens létrehozását ellenőrzi.
"""

from prompt_benchmark.llm.factory import SUPPORTED_PROVIDERS, create_llm_client
from prompt_benchmark.llm.client import MockLLMClient


def test_expected_providers_are_available():
    """EN: Checks that Mock, Ollama, Gemini, Groq, OpenRouter, and OpenAI are all registered.

    HU: Ellenőrzi, hogy Mock, Ollama, Gemini, Groq, OpenRouter és OpenAI mind regisztrálva vannak.
    """
    assert set(SUPPORTED_PROVIDERS) == {"mock", "ollama", "groq", "gemini", "openrouter", "openai"}


def test_mock_factory_requires_no_credentials():
    """EN: Ensures the offline mock provider can always be created without secrets.

    HU: Biztosítja, hogy az offline mock provider secret nélkül is létrehozható.
    """
    client = create_llm_client("mock", {"max_output_tokens": 64, "random_seed": 42})
    assert isinstance(client, MockLLMClient)
    assert client.provider == "mock"
