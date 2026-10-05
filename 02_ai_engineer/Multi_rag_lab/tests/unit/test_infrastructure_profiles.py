from __future__ import annotations

from rag_engine.platform.profiles import load_cuda_profiles, load_faiss_profiles, load_llm_profiles
from rag_engine.platform.runtime import ollama_profile_environment


def test_llm_profiles_have_expected_aliases() -> None:
    profiles = load_llm_profiles()
    assert profiles["balanced"]["base_model"] == "qwen3:4b"
    assert profiles["balanced"]["alias"] == "rag-qwen-balanced"
    assert profiles["low_memory"]["context_length"] == 2048


def test_ollama_environment_is_profile_driven() -> None:
    env = ollama_profile_environment("extended_context")
    assert env["OLLAMA_CONTEXT_LENGTH"] == "4096"
    assert env["OLLAMA_NUM_PARALLEL"] == "1"
    assert env["OLLAMA_HOST"] == "127.0.0.1:11434"


def test_faiss_profiles_keep_cpu_fallback() -> None:
    profiles = load_faiss_profiles()
    assert profiles["cpu"]["vector_device"] == "cpu"
    assert profiles["cuda_preferred"]["vector_device"] == "cuda"
    assert profiles["cuda_preferred"]["fallback_to_numpy"] is True


def test_cuda_auto_profile_keeps_faiss_cpu() -> None:
    profiles = load_cuda_profiles()
    assert profiles["auto"]["embedding_device"] == "auto"
    assert profiles["auto"]["vector_device"] == "cpu"
    assert profiles["cuda_strict"]["allow_cpu_fallback"] is False


def test_all_runtime_profiles_have_ui_metadata() -> None:
    for profiles in (load_llm_profiles(), load_faiss_profiles(), load_cuda_profiles()):
        assert profiles
        for name, profile in profiles.items():
            assert profile.get("display_name"), name
            assert "description" in profile, name


def test_faiss_profiles_have_persistent_output_directory() -> None:
    for name, profile in load_faiss_profiles().items():
        assert profile.get("output_dir"), name
