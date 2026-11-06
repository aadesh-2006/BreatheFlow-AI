"""Typed application configuration for BreatheFlow AI.

Safely loads configuration from environment variables without exposing
sensitive credentials in logs or console output.
"""

from dataclasses import dataclass
import os
from typing import Mapping, Optional


@dataclass(frozen=True)
class BreatheFlowConfig:
    """Immutable application configuration."""

    # Gnani AI API Credentials & Endpoints
    gnani_api_key: Optional[str] = None
    gnani_base_url: str = "https://api.vachana.ai"

    # Gnani Prisma STT Configuration
    stt_model: str = "prisma-v2.5"
    stt_language_code: str = "en-IN"

    # Gnani Timbre TTS Configuration
    tts_model: str = "timbre-v2.5"
    tts_language_code: str = "en-IN"
    tts_voice: str = "Kaveri"
    tts_speed: float = 0.95

    # Local Ollama / Qwen3-4B Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"

    # Application Behavior
    log_level: str = "INFO"

    def __repr__(self) -> str:
        """Mask sensitive API key to prevent accidental leakage in logs."""
        key_status = "<SET>" if self.gnani_api_key else "<NOT SET>"
        return (
            f"BreatheFlowConfig("
            f"gnani_api_key='{key_status}', "
            f"gnani_base_url='{self.gnani_base_url}', "
            f"stt_model='{self.stt_model}', "
            f"stt_language_code='{self.stt_language_code}', "
            f"tts_model='{self.tts_model}', "
            f"tts_language_code='{self.tts_language_code}', "
            f"tts_voice='{self.tts_voice}', "
            f"tts_speed={self.tts_speed}, "
            f"ollama_host='{self.ollama_host}', "
            f"ollama_model='{self.ollama_model}', "
            f"log_level='{self.log_level}')"
        )


def load_config(env: Optional[Mapping[str, str]] = None) -> BreatheFlowConfig:
    """Load configuration from provided mapping or system environment.

    Args:
        env: Optional mapping of environment variables (defaults to os.environ).

    Returns:
        Validated BreatheFlowConfig instance.
    """
    source = os.environ if env is None else env

    api_key = source.get("GNANI_API_KEY", "").strip() or None
    base_url = source.get("GNANI_BASE_URL", "https://api.vachana.ai").strip()

    stt_model = source.get("GNANI_STT_MODEL", "prisma-v2.5").strip()
    stt_language_code = source.get("GNANI_STT_LANGUAGE_CODE", "en-IN").strip()

    tts_model = source.get("GNANI_TTS_MODEL", "timbre-v2.5").strip()
    tts_language_code = source.get("GNANI_TTS_LANGUAGE_CODE", "en-IN").strip()
    tts_voice = source.get("GNANI_TTS_VOICE", "Kaveri").strip()

    try:
        tts_speed = float(source.get("GNANI_TTS_SPEED", "0.95"))
        if tts_speed <= 0:
            raise ValueError
    except (ValueError, TypeError):
        tts_speed = 0.95

    ollama_host = source.get("OLLAMA_HOST", "http://localhost:11434").strip()
    ollama_model = source.get("OLLAMA_MODEL", "qwen3:4b").strip()
    log_level = source.get("LOG_LEVEL", "INFO").strip().upper()

    return BreatheFlowConfig(
        gnani_api_key=api_key,
        gnani_base_url=base_url,
        stt_model=stt_model,
        stt_language_code=stt_language_code,
        tts_model=tts_model,
        tts_language_code=tts_language_code,
        tts_voice=tts_voice,
        tts_speed=tts_speed,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
        log_level=log_level,
    )
