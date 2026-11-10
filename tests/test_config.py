from pathlib import Path

import pytest

from kg_generator.config import DEFAULT_MODEL, ConfigError, Settings


def test_missing_api_key_is_rejected():
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        Settings.from_env({"LLM_API_KEY": "  "})


def test_defaults_apply_when_only_key_is_set():
    settings = Settings.from_env({"LLM_API_KEY": "secret"})

    assert settings.llm_model == DEFAULT_MODEL
    assert settings.falkordb_port == 6379
    assert settings.ontology_path == Path("state/ontology.json")


def test_environment_overrides_defaults():
    settings = Settings.from_env(
        {"LLM_API_KEY": "k", "LLM_MODEL": "openai/gpt-4.1", "FALKORDB_PORT": "7000",
         "DATA_DIR": "data/batch_2", "log_level": "ignored", "LOG_LEVEL": "debug"}
    )

    assert settings.llm_model == "openai/gpt-4.1"
    assert settings.falkordb_port == 7000
    assert settings.data_dir == Path("data/batch_2")
    assert settings.log_level == "DEBUG"


def test_malformed_integer_gives_clear_error():
    with pytest.raises(ConfigError, match="FALKORDB_PORT"):
        Settings.from_env({"LLM_API_KEY": "k", "FALKORDB_PORT": "abc"})


def test_api_key_never_appears_in_repr():
    assert "super-secret" not in repr(Settings.from_env({"LLM_API_KEY": "super-secret"}))
