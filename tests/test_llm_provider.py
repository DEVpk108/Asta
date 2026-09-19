from ai.llm_provider import create_llm_provider
from ai.llama_cpp_engine import LlamaCppEngine


def test_default_llm_provider_is_llama_cpp(monkeypatch):
    monkeypatch.delenv("ASTA_LLM_PROVIDER", raising=False)
    provider = create_llm_provider()

    assert isinstance(provider, LlamaCppEngine)


def test_llm_provider_can_be_selected_explicitly(monkeypatch):
    monkeypatch.setenv("ASTA_LLM_PROVIDER", "llama_cpp")
    provider = create_llm_provider()

    assert isinstance(provider, LlamaCppEngine)
