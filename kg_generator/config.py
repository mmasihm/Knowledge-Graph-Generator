"""Application settings, read from environment variables (optionally via a .env file)."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL = "gemini/gemini-3.5-flash"


class ConfigError(ValueError):
    """Raised when a required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration for the pipeline."""

    # repr=False keeps the key out of logs and tracebacks.
    llm_api_key: str = field(repr=False)
    llm_model: str = DEFAULT_MODEL
    graph_name: str = "knowledge_graph"
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    data_dir: Path = Path("data/batch_1")
    state_dir: Path = Path("state")
    pdf_strategy: str = "fast"
    chunk_max_chars: int = 8000
    log_level: str = "INFO"

    @property
    def ontology_path(self) -> Path:
        """Path of the saved ontology JSON."""
        return self.state_dir / "ontology.json"

    @property
    def tracker_path(self) -> Path:
        """Path of the ingested-files ledger."""
        return self.state_dir / "ingested_files.json"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "Settings":
        """Build settings from a mapping of environment variables."""
        api_key = env.get("LLM_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("LLM_API_KEY is not set (see .env.example).")
        return cls(
            llm_api_key=api_key,
            llm_model=env.get("LLM_MODEL") or DEFAULT_MODEL,
            graph_name=env.get("GRAPH_NAME", cls.graph_name),
            falkordb_host=env.get("FALKORDB_HOST", cls.falkordb_host),
            falkordb_port=_to_int(env, "FALKORDB_PORT", cls.falkordb_port),
            data_dir=Path(env.get("DATA_DIR", str(cls.data_dir))),
            state_dir=Path(env.get("STATE_DIR", str(cls.state_dir))),
            pdf_strategy=env.get("PDF_STRATEGY", cls.pdf_strategy),
            chunk_max_chars=_to_int(env, "CHUNK_MAX_CHARS", cls.chunk_max_chars),
            log_level=env.get("LOG_LEVEL", cls.log_level).upper(),
        )


def load_settings(env_file: str = ".env") -> Settings:
    """Load .env (if present) into the process environment and build settings."""
    load_dotenv(env_file)
    return Settings.from_env(os.environ)


def _to_int(env: Mapping[str, str], name: str, default: int) -> int:
    """Read an integer variable, failing with a clear message if malformed."""
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}.") from exc
