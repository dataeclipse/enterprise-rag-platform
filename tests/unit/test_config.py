import pytest
from pydantic import ValidationError

from rag.config import AuthConfig, Settings


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"auth": AuthConfig(secret_key="test-secret-key-for-unit-tests")}
    base.update(overrides)
    return Settings.model_validate(base)


def test_defaults() -> None:
    settings = make_settings()
    assert settings.env == "local"
    assert settings.qdrant.collection == "documents"
    assert settings.retrieval.rrf_k == 60
    assert settings.agents.max_correction_rounds == 2


def test_nested_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_AUTH__SECRET_KEY", "env-secret")
    monkeypatch.setenv("RAG_QDRANT__COLLECTION", "custom")
    monkeypatch.setenv("RAG_RETRIEVAL__DENSE_TOP_K", "50")
    settings = Settings()
    assert settings.qdrant.collection == "custom"
    assert settings.retrieval.dense_top_k == 50
    assert settings.auth.secret_key.get_secret_value() == "env-secret"


def test_secret_not_leaked_in_repr() -> None:
    settings = make_settings()
    assert "test-secret-key-for-unit-tests" not in repr(settings)
    assert "test-secret-key-for-unit-tests" not in settings.model_dump_json()


def test_missing_auth_secret_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_AUTH__SECRET_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_env_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(env="staging")
