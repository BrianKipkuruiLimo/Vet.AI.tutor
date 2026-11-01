"""Tests for the typed settings in vet_ai_tutor.config."""
import pytest

from vet_ai_tutor import config


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Ignore any VET_* variables set on the machine running the tests."""
    import os
    for name in [n for n in os.environ if n.startswith("VET_")]:
        monkeypatch.delenv(name)


def load(env_file=None, **env):
    """Build settings from the given VET_* variables only (no real .env)."""
    return config.load_settings(env_file=env_file, **env)


def test_defaults_match_constants():
    s = load()
    assert s.llm_model == "qwen3:1.7b"
    assert s.retriever_k == 5 and s.chunk_size == 600
    assert s.offline_only is True and s.use_reranker is False
    assert s.cors_origins == "http://localhost:8000,http://127.0.0.1:8000"


def test_environment_variables_are_typed(monkeypatch):
    monkeypatch.setenv("VET_RETRIEVER_K", "8")
    monkeypatch.setenv("VET_LLM_REASONING", "yes")
    monkeypatch.setenv("VET_EMBEDDING_PREFIXES", "OFF")
    monkeypatch.setenv("VET_API_PORT", "9000")
    s = load()
    assert s.retriever_k == 8 and s.llm_reasoning is True
    assert s.embedding_prefixes == "off"
    assert s.cors_origins == "http://localhost:9000,http://127.0.0.1:9000"


@pytest.mark.parametrize("name, value, message", [
    ("VET_RETRIEVER_K", "0", "VET_RETRIEVER_K"),
    ("VET_RETRIEVER_K", "five", "VET_RETRIEVER_K"),
    ("VET_VECTOR_BACKEND", "pinecone", "VET_VECTOR_BACKEND"),
    ("VET_CHUNK_SIZE", "50", "VET_CHUNK_SIZE"),
    ("VET_LLM_NUM_CTX", "256", "VET_LLM_NUM_CTX"),
    ("VET_OFFLINE_ONLY", "maybe", "VET_OFFLINE_ONLY"),
])
def test_invalid_values_name_the_variable(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        load()


def test_cross_field_rules(monkeypatch):
    monkeypatch.setenv("VET_CHUNK_OVERLAP", "600")
    with pytest.raises(ValueError, match="VET_CHUNK_OVERLAP .* must be <"):
        load()
    monkeypatch.delenv("VET_CHUNK_OVERLAP")
    monkeypatch.setenv("VET_RETRIEVER_K", "10")
    monkeypatch.setenv("VET_RETRIEVAL_CANDIDATES", "5")
    with pytest.raises(ValueError, match="VET_RETRIEVAL_CANDIDATES"):
        load()


def test_env_file_is_read_and_environment_wins(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("VET_LLM_MODEL=qwen3:4b\nVET_RETRIEVER_K=7\n", encoding="utf-8")
    assert load(str(env_file)).llm_model == "qwen3:4b"
    monkeypatch.setenv("VET_RETRIEVER_K", "3")
    s = load(str(env_file))
    assert s.retriever_k == 3
    sources = {name: source for name, _, source in config.describe(s, env_file=str(env_file))}
    assert sources["VET_LLM_MODEL"] == ".env"
    assert sources["VET_RETRIEVER_K"] == "environment"
    assert sources["VET_CHUNK_SIZE"] == "default"


def test_unknown_variables_get_suggestions(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("VET_CHUNK_SIZ=800\n", encoding="utf-8")
    unknown = config.unknown_variables({"VET_LLM_MODLE": "x", "VET_LLM_MODEL": "y", "PATH": "/bin"},
                                       env_file=str(env_file))
    assert unknown == {"VET_LLM_MODLE": "VET_LLM_MODEL", "VET_CHUNK_SIZ": "VET_CHUNK_SIZE"}
    assert config.unknown_variables({"VET_COMPLETELY_DIFFERENT_THING": "1"}, env_file=None) == {
        "VET_COMPLETELY_DIFFERENT_THING": None}


def test_admin_token_is_masked():
    rows = {name: value for name, value, _ in config.describe(load(admin_token="s3cret"), environ={},
                                                             env_file=None)}
    assert rows["VET_ADMIN_TOKEN"] == "(set)"
    assert "s3cret" not in str(rows)


def test_env_example_documents_every_setting():
    text = config.env_example()
    for field, info in config.Settings.model_fields.items():
        assert f"# {config.env_name(field)}=" in text
        assert info.description in text
    assert "s3cret" not in text
    assert "# VET_OFFLINE_ONLY=1" in text and "# VET_ADMIN_TOKEN=\n" in text
