"""Abstract provider contracts for speech input, speech synthesis, and reasoning.

Decouples BreatheFlow core execution from specific cloud vendor SDKs and APIs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class STTResult:
    """Result of speech transcription."""

    transcript: str
    detected_language: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TTSAudio:
    """Synthesized speech audio payload."""

    audio_bytes: bytes
    content_type: str = "audio/wav"
    sample_rate: int = 24000
    duration_s: Optional[float] = None


@dataclass(frozen=True)
class ConversationTurn:
    """Result of an empathetic conversational turn."""

    spoken_response: str
    suggested_protocol: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None


class STTProvider(ABC):
    """Abstract speech-to-text transcription provider."""

    @abstractmethod
    def transcribe(
        self, audio_bytes: bytes, language_code: Optional[str] = None
    ) -> STTResult:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Binary audio payload (e.g. WAV, PCM).
            language_code: Optional language code (e.g. 'en-IN').

        Returns:
            STTResult containing the transcribed text.
        """
        pass


class TTSProvider(ABC):
    """Abstract text-to-speech synthesis provider."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        language_code: Optional[str] = None,
    ) -> TTSAudio:
        """Synthesize text into speech audio bytes.

        Args:
            text: Text to synthesize.
            voice: Voice persona identifier.
            speed: Speaking rate multiplier.
            language_code: Target language code.

        Returns:
            TTSAudio containing audio bytes and format metadata.
        """
        pass


class ConversationEngine(ABC):
    """Abstract conversational brain for intent understanding and guidance."""

    @abstractmethod
    def generate_turn(
        self,
        user_text: str,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Generate empathetic spoken guidance and optional protocol recommendations.

        Args:
            user_text: Transcribed user input.
            session_context: Optional state metadata (e.g., current activity,
                previous protocol, physiological estimate).

        Returns:
            ConversationTurn with spoken response and optional protocol parameters.
        """
        pass
