import litellm
import pytest
from graphrag_sdk.models import GenerativeModelConfig

from kg_generator.llm import RetryingChatSession, expose_api_key, is_transient


class FakeModel:
    """Minimal stand-in for LiteModel (the real one calls the API in its constructor)."""

    model = "gemini/test"
    generation_config = GenerativeModelConfig()
    additional_params: dict = {}

    @staticmethod
    def parse_generate_content_response(response):
        return response


def fake_completion(outcomes):
    """Return a completion() replacement that raises or returns the queued outcomes in order."""
    def completion(**kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return completion


def test_transient_error_is_retried(monkeypatch):
    rate_limited = litellm.RateLimitError("slow down", llm_provider="gemini", model="test")
    reply = type("Reply", (), {"text": "ok"})()
    monkeypatch.setattr("graphrag_sdk.models.litellm.completion", fake_completion([rate_limited, reply]))
    delays = []
    session = RetryingChatSession(FakeModel(), "system", sleep=delays.append)

    assert session.send_message("hello").text == "ok"
    assert delays == [2.0]
    assert [m["role"] for m in session.get_chat_history()] == ["system", "user", "assistant"]


def test_permanent_error_is_not_retried(monkeypatch):
    auth_error = litellm.AuthenticationError("bad key", llm_provider="gemini", model="test")
    monkeypatch.setattr("graphrag_sdk.models.litellm.completion", fake_completion([auth_error]))
    session = RetryingChatSession(FakeModel(), sleep=lambda _: pytest.fail("should not retry"))

    with pytest.raises(ValueError):
        session.send_message("hello")


@pytest.mark.parametrize(
    ("message", "expected"),
    [("Requests per minute exceeded", True), ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", False)],
)
def test_daily_quota_is_not_worth_retrying(message, expected):
    wrapped = ValueError("Error during completion request")
    wrapped.__context__ = litellm.RateLimitError(message, llm_provider="gemini", model="test")

    assert is_transient(wrapped) is expected


def test_api_key_is_exposed_under_the_provider_variable():
    environ: dict[str, str] = {}

    expose_api_key("gemini/gemini-3.5-flash", "secret", environ)

    assert environ["GEMINI_API_KEY"] == "secret"
