"""Provider interfaces and concrete adapters for STT, TTS, and conversational reasoning."""

from breatheflow.providers.base import (
    ConversationEngine,
    ConversationTurn,
    STTProvider,
    STTResult,
    TTSAudio,
    TTSProvider,
)
from breatheflow.providers.exceptions import (
    GnaniAPIError,
    GnaniTimeoutError,
    OllamaConnectionError,
    OllamaError,
    OllamaResponseError,
    ProviderError,
)
from breatheflow.providers.gnani_stt import GnaniSTTAdapter
from breatheflow.providers.gnani_tts import GnaniTTSAdapter
from breatheflow.providers.ollama_brain import OllamaBrainAdapter

__all__ = [
    "ConversationEngine",
    "ConversationTurn",
    "GnaniAPIError",
    "GnaniSTTAdapter",
    "GnaniTTSAdapter",
    "GnaniTimeoutError",
    "OllamaBrainAdapter",
    "OllamaConnectionError",
    "OllamaError",
    "OllamaResponseError",
    "ProviderError",
    "STTProvider",
    "STTResult",
    "TTSAudio",
    "TTSProvider",
]
