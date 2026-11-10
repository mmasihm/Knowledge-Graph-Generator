"""LiteLLM-backed generative model with retries on transient provider errors."""

import logging
import os
import time
from collections.abc import Callable, MutableMapping

import litellm
from graphrag_sdk.models import GenerativeModelConfig
from graphrag_sdk.models.litellm import LiteModel, LiteModelChatSession

from kg_generator.config import Settings

logger = logging.getLogger(__name__)

TRANSIENT_ERRORS = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.BadGatewayError,
)


class RetryingChatSession(LiteModelChatSession):
    """Chat session that retries transient LLM errors with exponential backoff."""

    def __init__(
        self,
        model: LiteModel,
        system_instruction: str | None = None,
        max_attempts: int = 5,
        base_delay: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Wrap the SDK chat session with a retry policy."""
        super().__init__(model, system_instruction)
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._sleep = sleep

    def send_message(self, message: str):
        """Send a message, retrying when the provider fails transiently."""
        for attempt in range(1, self._max_attempts + 1):
            try:
                return super().send_message(message)
            except ValueError as error:
                # The SDK appends the user turn before calling the API; undo it so a retry doesn't duplicate it.
                self._chat_history.pop()
                if attempt == self._max_attempts or not is_transient(error):
                    raise
                delay = self._base_delay * 2 ** (attempt - 1)
                logger.warning("Transient LLM error (attempt %d/%d), retrying in %.0fs: %s",
                               attempt, self._max_attempts, delay, type(error.__context__).__name__)
                self._sleep(delay)
        raise AssertionError("unreachable")


class RetryingLiteModel(LiteModel):
    """LiteModel whose chat sessions retry transient errors."""

    def start_chat(self, system_instruction: str | None = None) -> RetryingChatSession:
        """Start a retrying chat session."""
        return RetryingChatSession(self, system_instruction)


def is_transient(error: BaseException) -> bool:
    """Return True if the error (or the one the SDK wrapped) is worth retrying."""
    return isinstance(error, TRANSIENT_ERRORS) or isinstance(error.__context__, TRANSIENT_ERRORS)


def expose_api_key(model_name: str, api_key: str, environ: MutableMapping[str, str] = os.environ) -> None:
    """Copy LLM_API_KEY into the provider-specific variable LiteLLM reads (e.g. GEMINI_API_KEY)."""
    for variable in litellm.validate_environment(model_name)["missing_keys"]:
        environ[variable] = api_key


def create_model(settings: Settings) -> RetryingLiteModel:
    """Build the generative model used for both ontology creation and extraction."""
    expose_api_key(settings.llm_model, settings.llm_api_key)
    return RetryingLiteModel(
        settings.llm_model,
        # Temperature 0 makes extraction as repeatable as the provider allows.
        generation_config=GenerativeModelConfig(temperature=0),
        additional_params={"timeout": 300},
    )
