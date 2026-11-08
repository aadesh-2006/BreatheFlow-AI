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
    GeminiAuthError,
    GeminiConnectionError,
    GeminiError,
    GeminiRateLimitError,
    GeminiResponseError,
    GnaniAPIError,
    GnaniTimeoutError,
    OllamaConnectionError,
    OllamaError,
    OllamaResponseError,
    ProviderError,
)
from breatheflow.providers.gemini_brain import GeminiBrainAdapter
from breatheflow.providers.gnani_stt import GnaniSTTAdapter
from breatheflow.providers.gnani_tts import GnaniTTSAdapter
from breatheflow.providers.ollama_brain import OllamaBrainAdapter

__all__ = [
    "ConversationEngine",
    "ConversationTurn",
    "GeminiAuthError",
    "GeminiBrainAdapter",
    "GeminiConnectionError",
    "GeminiError",
    "GeminiRateLimitError",
    "GeminiResponseError",
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
