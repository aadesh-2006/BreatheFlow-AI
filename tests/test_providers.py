"""Unit tests for provider abstract base classes and contracts."""

import unittest
from typing import Any, Dict, Optional
from breatheflow.providers.base import (
    ConversationEngine,
    ConversationTurn,
    STTProvider,
    STTResult,
    TTSAudio,
    TTSProvider,
)


class MockSTT(STTProvider):
    """Mock implementation of STTProvider for testing."""

    def transcribe(
        self, audio_bytes: bytes, language_code: Optional[str] = None
    ) -> STTResult:
        if not audio_bytes:
            return STTResult(transcript="", confidence=0.0)
        return STTResult(
            transcript="I want to do box breathing",
            detected_language=language_code or "en-IN",
            confidence=0.98,
        )


class MockTTS(TTSProvider):
    """Mock implementation of TTSProvider for testing."""

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        language_code: Optional[str] = None,
    ) -> TTSAudio:
        # Returns simulated mock WAV bytes
        simulated_wav = b"RIFF" + b"\x00" * 36 + text.encode("utf-8")
        return TTSAudio(
            audio_bytes=simulated_wav,
            content_type="audio/wav",
            sample_rate=24000,
            duration_s=2.5,
        )


class MockBrain(ConversationEngine):
    """Mock implementation of ConversationEngine for testing."""

    def generate_turn(
        self,
        user_text: str,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        return ConversationTurn(
            spoken_response="Let's start Box Breathing together.",
            suggested_protocol={"protocol": "box_breathing", "cycles": 4},
            raw_response="Mocked LLM raw output",
        )


class TestProviderContracts(unittest.TestCase):
    """Verify abstract contracts enforce implementation and return expected types."""

    def test_cannot_instantiate_abstract_classes(self) -> None:
        with self.assertRaises(TypeError):
            STTProvider()  # type: ignore[abstract]
        with self.assertRaises(TypeError):
            TTSProvider()  # type: ignore[abstract]
        with self.assertRaises(TypeError):
            ConversationEngine()  # type: ignore[abstract]

    def test_mock_stt_contract(self) -> None:
        stt = MockSTT()
        result = stt.transcribe(b"fake_audio_pcm_bytes", language_code="en-IN")
        self.assertEqual(result.transcript, "I want to do box breathing")
        self.assertEqual(result.detected_language, "en-IN")
        self.assertEqual(result.confidence, 0.98)

    def test_mock_tts_contract(self) -> None:
        tts = MockTTS()
        audio = tts.synthesize("Welcome to BreatheFlow", voice="Kaveri")
        self.assertTrue(audio.audio_bytes.startswith(b"RIFF"))
        self.assertEqual(audio.content_type, "audio/wav")
        self.assertEqual(audio.sample_rate, 24000)

    def test_mock_brain_contract(self) -> None:
        brain = MockBrain()
        turn = brain.generate_turn("I feel tense", session_context={"state": "idle"})
        self.assertEqual(turn.spoken_response, "Let's start Box Breathing together.")
        self.assertIsNotNone(turn.suggested_protocol)
        if turn.suggested_protocol:
            self.assertEqual(turn.suggested_protocol["protocol"], "box_breathing")


if __name__ == "__main__":
    unittest.main()
