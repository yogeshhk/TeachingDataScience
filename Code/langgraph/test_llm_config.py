"""Tests for llm_config.py. No network, no API keys, no running model server."""
import importlib

import pytest

import llm_config as cfg


def reload_with(monkeypatch, **env):
    for key in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_TEMPERATURE", "LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(cfg)


def test_defaults_are_local_and_free(monkeypatch):
    c = reload_with(monkeypatch)
    assert c.PROVIDER == "lmstudio"
    assert c.MODEL_NAME == "gemma-2-2b-it"
    assert c.BASE_URL == "http://localhost:1234/v1"


def test_llm_model_env_changes_the_one_shared_variable(monkeypatch):
    c = reload_with(monkeypatch, LLM_MODEL="qwen/qwen3-1.7b")
    assert c.MODEL_NAME == "qwen/qwen3-1.7b"
    assert c.get_llm().model_name == "qwen/qwen3-1.7b"


def test_lmstudio_builds_an_openai_compatible_client_without_a_key(monkeypatch):
    c = reload_with(monkeypatch)
    llm = c.get_llm()
    assert llm.model_name == "gemma-2-2b-it"
    assert str(llm.openai_api_base).rstrip("/") == "http://localhost:1234/v1"
    assert llm.temperature == 0.0


def test_ollama_uses_its_own_address_and_default_model(monkeypatch):
    c = reload_with(monkeypatch, LLM_PROVIDER="ollama")
    llm = c.get_llm()
    assert llm.model_name == "gemma2:2b"
    assert str(llm.openai_api_base).rstrip("/") == "http://localhost:11434/v1"


def test_temperature_override(monkeypatch):
    c = reload_with(monkeypatch, LLM_TEMPERATURE="0.7")
    assert c.get_llm().temperature == 0.7
    assert c.get_llm(temperature=0.1).temperature == 0.1


def test_unknown_provider_rejected(monkeypatch):
    c = reload_with(monkeypatch)
    with pytest.raises(ValueError):
        c.get_llm("openai")


def test_other_provider_keeps_its_own_default_model(monkeypatch):
    c = reload_with(monkeypatch)
    assert c.model_for("groq") == "openai/gpt-oss-20b"
    assert c.model_for("lmstudio") == "gemma-2-2b-it"


def test_unreachable_server_lists_no_models(monkeypatch):
    c = reload_with(monkeypatch)
    assert c.list_local_models("http://localhost:1/v1", timeout=1) == []


class TestFoldSystemPrompt:
    def test_system_is_merged_into_first_user_message(self):
        out = cfg.fold_system_prompt([("system", "Be brief."), ("user", "Hi")])
        assert out == [("user", "Be brief.\n\nHi")]

    def test_messages_without_system_pass_through(self):
        msgs = [("user", "Hi"), ("assistant", "Hello")]
        assert cfg.fold_system_prompt(msgs) == msgs

    def test_system_only_becomes_a_user_message(self):
        assert cfg.fold_system_prompt([("system", "Be brief.")]) == [("user", "Be brief.")]

    def test_later_messages_are_kept(self):
        out = cfg.fold_system_prompt([("system", "S"), ("user", "A"), ("assistant", "B"), ("user", "C")])
        assert out == [("user", "S\n\nA"), ("assistant", "B"), ("user", "C")]
