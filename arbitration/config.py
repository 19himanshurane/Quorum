"""Environment-driven configuration. No provider SDKs are imported here."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CriticConfig:
    name: str
    provider: str  # "groq" | "mistral" | "nvidia" | "openai" | "ollama" | "mock"
    model: str


@dataclass(frozen=True)
class Settings:
    provider_mode: str
    accuracy_critic: CriticConfig
    logic_critic: CriticConfig
    completeness_critic: CriticConfig
    adjudicator: CriticConfig
    ollama_base_url: str
    db_path: str
    max_retries: int
    request_timeout_seconds: float


def _mode() -> str:
    return os.getenv("ARBITRATION_PROVIDER_MODE", "mock").strip().lower()


def load_settings() -> Settings:
    mode = _mode()

    def provider_for(default_provider: str) -> str:
        return default_provider if mode == "live" else "mock"

    # NVIDIA NIM's free-tier queueing made it unusably slow in practice (51s-180s+
    # per call, sometimes longer than our request timeout). Both accuracy and
    # completeness now run on Groq - fast and reliable - but on different model
    # families (OpenAI's gpt-oss vs Alibaba's Qwen) so they don't share blind spots.
    accuracy = CriticConfig(
        name="accuracy_critic",
        provider=provider_for(os.getenv("ACCURACY_CRITIC_PROVIDER", "groq")),
        model=os.getenv("ACCURACY_CRITIC_MODEL", "openai/gpt-oss-120b"),
    )
    logic = CriticConfig(
        name="logic_critic",
        provider=provider_for(os.getenv("LOGIC_CRITIC_PROVIDER", "mistral")),
        model=os.getenv("LOGIC_CRITIC_MODEL", "mistral-large-latest"),
    )
    completeness = CriticConfig(
        name="completeness_critic",
        provider=provider_for(os.getenv("COMPLETENESS_CRITIC_PROVIDER", "groq")),
        model=os.getenv("COMPLETENESS_CRITIC_MODEL", "qwen/qwen3.6-27b"),
    )
    adjudicator = CriticConfig(
        name="adjudicator",
        provider=provider_for(os.getenv("ADJUDICATOR_PROVIDER", "mistral")),
        model=os.getenv("ADJUDICATOR_MODEL", "mistral-large-latest"),
    )

    return Settings(
        provider_mode=mode,
        accuracy_critic=accuracy,
        logic_critic=logic,
        completeness_critic=completeness,
        adjudicator=adjudicator,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        db_path=os.getenv("ARBITRATION_DB_PATH", str(PROJECT_ROOT / "data" / "arbitration.db")),
        max_retries=int(os.getenv("CRITIC_MAX_RETRIES", "2")),
        # The openai SDK's own default timeout is ~10 minutes; without an explicit
        # override here, a slow/unresponsive provider hangs each retry attempt for
        # that long instead of failing fast into the next attempt / graceful
        # degradation path.
        request_timeout_seconds=float(os.getenv("CRITIC_REQUEST_TIMEOUT_SECONDS", "30")),
    )
