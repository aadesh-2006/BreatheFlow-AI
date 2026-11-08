"""Unit tests for BreatheFlow dialogue and session orchestrator."""

import unittest
import unittest.mock
from typing import Any, Dict, Optional

from breatheflow.config import BreatheFlowConfig
from breatheflow.core.controller import (
    BreathingPhase,
    BreathingProtocol,
    DeterministicBreathingController,
)
from breatheflow.core.orchestrator import (
    BreatheFlowOrchestrator,
    OrchestratorStatus,
)
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
    OllamaConnectionError,
    OllamaResponseError,
)
from breatheflow.providers.gemini_brain import GeminiBrainAdapter


class MockSTT(STTProvider):
    def __init__(self, transcript: str = "Start box breathing", fail: bool = False):
        self.transcript = transcript
        self.fail = fail
        self.call_count = 0

    def transcribe(
        self, audio_bytes: bytes, language_code: Optional[str] = None
    ) -> STTResult:
        self.call_count += 1
        if self.fail:
            raise GnaniAPIError("STT service unavailable", status_code=503)
        return STTResult(transcript=self.transcript, confidence=0.98)


class MockBrain(ConversationEngine):
    def __init__(
        self,
        spoken: str = "Let's do Box Breathing.",
        protocol: Optional[Dict[str, Any]] = None,
        fail: bool = False,
    ):
        self.spoken = spoken
        self.protocol = protocol or {
            "action": "start_session",
            "protocol": "box_breathing",
            "inhale_s": 4.0,
            "hold_in_s": 4.0,
            "exhale_s": 4.0,
            "hold_out_s": 4.0,
            "cycles": 4,
        }
        self.fail = fail
        self.last_context: Optional[Dict[str, Any]] = None

    def generate_turn(
        self, user_text: str, session_context: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        self.last_context = session_context
        if self.fail:
            raise OllamaConnectionError("Failed to reach Ollama")
        return ConversationTurn(
            spoken_response=self.spoken, suggested_protocol=self.protocol
        )


class MockTTS(TTSProvider):
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.last_synthesized_text: Optional[str] = None

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        language_code: Optional[str] = None,
    ) -> TTSAudio:
        self.last_synthesized_text = text
        if self.fail:
            raise GnaniAPIError("TTS quota exceeded", status_code=429)
        return TTSAudio(audio_bytes=b"RIFF_AUDIO_DATA", content_type="audio/wav")


class TestBreatheFlowOrchestrator(unittest.TestCase):
    """Test full voice turn orchestration, provider handling, and controller independence."""

    def setUp(self) -> None:
        self.stt = MockSTT()
        self.brain = MockBrain()
        self.tts = MockTTS()
        self.controller = DeterministicBreathingController()
        self.config = BreatheFlowConfig(gnani_api_key="test_key")
        self.orchestrator = BreatheFlowOrchestrator(
            stt=self.stt,
            brain=self.brain,
            tts=self.tts,
            controller=self.controller,
            config=self.config,
        )

    def test_successful_turn_starts_controller(self) -> None:
        audio_in = b"RIFF_INPUT_BYTES"
        result = self.orchestrator.process_voice_turn(audio_in)

        self.assertEqual(result.status, OrchestratorStatus.SUCCESS)
        self.assertEqual(result.transcript, "Start box breathing")
        self.assertEqual(result.spoken_response, "Let's do Box Breathing.")
        self.assertIsNotNone(result.audio_response)
        if result.audio_response:
            self.assertEqual(result.audio_response.audio_bytes, b"RIFF_AUDIO_DATA")

        # Verify controller was started with recommended protocol
        self.assertTrue(self.controller.is_active)
        self.assertEqual(self.controller.phase, BreathingPhase.INHALE)
        self.assertEqual(self.controller.current_cycle, 1)
        self.assertIsNotNone(result.controller_tick)

    def test_controller_independence_when_already_active(self) -> None:
        """Verify that if the controller is already running, an LLM suggestion CANNOT overwrite it."""
        # Manually start controller on coherent breathing
        coherent = BreathingProtocol.coherent_breathing(cycles=5, duration_s=5.5)
        self.controller.start(coherent)
        # Advance 2 seconds into coherent breathing
        self.controller.step(2.0)
        self.assertEqual(self.controller.protocol.name, "Coherent Breathing")
        self.assertEqual(self.controller.phase, BreathingPhase.INHALE)

        # Process a voice turn where LLM recommends Box Breathing
        audio_in = b"RIFF_INPUT_BYTES"
        result = self.orchestrator.process_voice_turn(audio_in)

        # The turn succeeds conversationally
        self.assertEqual(result.status, OrchestratorStatus.SUCCESS)
        # BUT the active protocol remains UNCHANGED (Coherent Breathing, not Box Breathing)
        self.assertEqual(self.controller.protocol.name, "Coherent Breathing")
        self.assertEqual(self.controller.protocol.inhale_s, 5.5)
        # And timing in controller was NOT reset or altered
        self.assertEqual(self.controller.phase, BreathingPhase.INHALE)
        self.assertIsNone(result.controller_tick)

    def test_session_context_passed_to_brain(self) -> None:
        # Start a protocol
        self.controller.start(BreathingProtocol.box_breathing(cycles=4))
        self.controller.step(4.0)  # Move to HOLD_IN

        self.orchestrator.process_voice_turn(b"I feel good")
        ctx = self.brain.last_context
        self.assertIsNotNone(ctx)
        if ctx:
            self.assertEqual(ctx["controller_phase"], "hold_in")
            self.assertTrue(ctx["is_active"])
            self.assertEqual(ctx["current_cycle"], 1)
            self.assertEqual(ctx["total_cycles"], 4)

    def test_empty_transcript_handling(self) -> None:
        stt = MockSTT(transcript="")
        orchestrator = BreatheFlowOrchestrator(
            stt=stt,
            brain=self.brain,
            tts=self.tts,
            controller=self.controller,
        )

        result = orchestrator.process_voice_turn(b"SILENCE_AUDIO")
        self.assertEqual(result.status, OrchestratorStatus.EMPTY_TRANSCRIPT)
        self.assertEqual(result.transcript, "")
        self.assertIn("didn't catch that", result.spoken_response)
        self.assertIsNotNone(result.audio_response)
        # Controller was NOT started
        self.assertFalse(self.controller.is_active)

    def test_stt_failure_handling(self) -> None:
        stt = MockSTT(fail=True)
        orchestrator = BreatheFlowOrchestrator(
            stt=stt,
            brain=self.brain,
            tts=self.tts,
            controller=self.controller,
        )

        result = orchestrator.process_voice_turn(b"SOME_AUDIO")
        self.assertEqual(result.status, OrchestratorStatus.STT_ERROR)
        self.assertIn("Speech transcription failed", str(result.error_message))
        self.assertFalse(self.controller.is_active)

    def test_brain_failure_handling(self) -> None:
        brain = MockBrain(fail=True)
        orchestrator = BreatheFlowOrchestrator(
            stt=self.stt,
            brain=brain,
            tts=self.tts,
            controller=self.controller,
        )

        result = orchestrator.process_voice_turn(b"SOME_AUDIO")
        self.assertEqual(result.status, OrchestratorStatus.BRAIN_ERROR)
        self.assertEqual(result.transcript, "Start box breathing")
        # Generates fallback calming response and synthesizes audio
        self.assertIn("comfortable breath", result.spoken_response)
        self.assertIsNotNone(result.audio_response)
        self.assertIn("Brain reasoning failed", str(result.error_message))

    def test_tts_failure_handling(self) -> None:
        tts = MockTTS(fail=True)
        orchestrator = BreatheFlowOrchestrator(
            stt=self.stt,
            brain=self.brain,
            tts=tts,
            controller=self.controller,
        )

        result = orchestrator.process_voice_turn(b"SOME_AUDIO")
        self.assertEqual(result.status, OrchestratorStatus.TTS_ERROR)
        # Transcript and spoken text are still preserved
        self.assertEqual(result.transcript, "Start box breathing")
        self.assertEqual(result.spoken_response, "Let's do Box Breathing.")
        self.assertIsNone(result.audio_response)
        self.assertIn("Speech synthesis failed", str(result.error_message))
        self.assertFalse(self.controller.is_active)
        self.assertIsNone(result.controller_tick)

    def test_tts_failure_cleans_up_active_controller_and_allows_subsequent_turn(self) -> None:
        """Verify that when TTS fails after starting a protocol, controller is stopped and not left active.

        Also verify that a subsequent turn can successfully initialize a protocol once TTS recovers.
        """
        failing_tts = MockTTS(fail=True)
        orchestrator = BreatheFlowOrchestrator(
            stt=self.stt,
            brain=self.brain,
            tts=failing_tts,
            controller=self.controller,
        )

        # 1. Turn with failing TTS attempting to start box breathing
        result_fail = orchestrator.process_voice_turn(b"SOME_AUDIO")
        self.assertEqual(result_fail.status, OrchestratorStatus.TTS_ERROR)
        self.assertIsNone(result_fail.audio_response)
        self.assertIsNone(result_fail.controller_tick)
        self.assertFalse(self.controller.is_active)
        self.assertEqual(self.controller.phase, BreathingPhase.TERMINATED)

        # 2. Recovery: TTS works on the subsequent turn, successfully starting a session
        failing_tts.fail = False
        result_success = orchestrator.process_voice_turn(b"SOME_AUDIO")
        self.assertEqual(result_success.status, OrchestratorStatus.SUCCESS)
        self.assertIsNotNone(result_success.audio_response)
        self.assertIsNotNone(result_success.controller_tick)
        self.assertTrue(self.controller.is_active)
        self.assertEqual(self.controller.phase, BreathingPhase.INHALE)
        self.assertEqual(self.controller.current_cycle, 1)

    def test_tts_failure_does_not_stop_pre_existing_active_session(self) -> None:
        """Verify that TTS failure during a mid-session turn does NOT abort a pre-existing active session."""
        # Manually start a session beforehand
        protocol = BreathingProtocol.coherent_breathing(cycles=5, duration_s=5.5)
        self.controller.start(protocol)
        self.controller.step(2.0)  # 2s into inhale
        self.assertTrue(self.controller.is_active)
        self.assertEqual(self.controller.phase, BreathingPhase.INHALE)

        # Execute a turn where TTS fails
        failing_tts = MockTTS(fail=True)
        orchestrator = BreatheFlowOrchestrator(
            stt=self.stt,
            brain=self.brain,
            tts=failing_tts,
            controller=self.controller,
        )

        result = orchestrator.process_voice_turn(b"SOME_AUDIO")
        self.assertEqual(result.status, OrchestratorStatus.TTS_ERROR)
        # Pre-existing session was NOT stopped; still active in inhale phase with original protocol
        self.assertTrue(self.controller.is_active)
        self.assertEqual(self.controller.phase, BreathingPhase.INHALE)
        self.assertEqual(self.controller.protocol.name, "Coherent Breathing")
        self.assertIsNone(result.controller_tick)

    def test_turn_history_tracking(self) -> None:
        self.orchestrator.process_voice_turn(b"AUDIO_1")
        history = self.orchestrator.turn_history
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["user"], "Start box breathing")
        self.assertEqual(history[0]["assistant"], "Let's do Box Breathing.")

    def test_bounded_history_eviction(self) -> None:
        """Verify that turn history does not grow unbounded and retains chronological order."""
        orchestrator = BreatheFlowOrchestrator(
            stt=self.stt,
            brain=self.brain,
            tts=self.tts,
            controller=self.controller,
            max_history_turns=3,
        )

        for i in range(5):
            self.stt.transcript = f"Turn {i}"
            orchestrator.process_voice_turn(b"AUDIO")

        history = orchestrator.turn_history
        self.assertEqual(len(history), 3)
        # Chronological order must be preserved: turns 2, 3, 4 retained; turns 0, 1 evicted
        self.assertEqual(history[0]["user"], "Turn 2")
        self.assertEqual(history[1]["user"], "Turn 3")
        self.assertEqual(history[2]["user"], "Turn 4")

    def test_invalid_suggested_protocol_while_idle(self) -> None:
        """Verify that malformed or negative protocol params fail safely without crashing."""
        malformed_protocol = {
            "action": "start_session",
            "protocol": "broken",
            "inhale_s": -2.0,  # Invalid negative duration
            "hold_in_s": 0.0,
            "exhale_s": 4.0,
            "hold_out_s": 0.0,
            "cycles": 0,  # Invalid zero cycles
        }
        brain = MockBrain(spoken="Try this broken protocol.", protocol=malformed_protocol)
        orchestrator = BreatheFlowOrchestrator(
            stt=self.stt,
            brain=brain,
            tts=self.tts,
            controller=self.controller,
        )

        result = orchestrator.process_voice_turn(b"AUDIO")
        self.assertEqual(result.status, OrchestratorStatus.SUCCESS)
        self.assertEqual(result.spoken_response, "Try this broken protocol.")
        # Controller remains IDLE and controller_tick is None
        self.assertEqual(self.controller.phase, BreathingPhase.IDLE)
        self.assertIsNone(result.controller_tick)

    def test_max_history_turns_validation(self) -> None:
        with self.assertRaises(ValueError):
            BreatheFlowOrchestrator(
                stt=self.stt,
                brain=self.brain,
                tts=self.tts,
                max_history_turns=0,
            )

    def test_orchestrator_initializes_simplified_protocol(self) -> None:
        """Verify orchestrator resolves simplified protocol payloads to standard presets."""
        simplified = {"action": "start_session", "protocol": "box_breathing"}
        brain = MockBrain(spoken="Starting box breathing.", protocol=simplified)
        orchestrator = BreatheFlowOrchestrator(
            stt=self.stt,
            brain=brain,
            tts=self.tts,
            controller=self.controller,
        )

        result = orchestrator.process_voice_turn(b"AUDIO")
        self.assertEqual(result.status, OrchestratorStatus.SUCCESS)
        self.assertTrue(self.controller.is_active)
        self.assertEqual(self.controller.phase, BreathingPhase.INHALE)
        self.assertIsNotNone(self.controller.protocol)
        if self.controller.protocol:
            self.assertEqual(self.controller.protocol.name, "Box Breathing")
            self.assertEqual(self.controller.protocol.inhale_s, 4.0)
            self.assertEqual(self.controller.protocol.cycles, 4)

    def test_empty_content_brain_error_not_success(self) -> None:
        """Verify empty model response error is handled as BRAIN_ERROR, not SUCCESS."""
        class EmptyContentBrain(ConversationEngine):
            def generate_turn(self, user_text: str, session_context: Optional[Dict[str, Any]] = None) -> ConversationTurn:
                raise OllamaResponseError("Ollama completed generation with empty content")

        orchestrator = BreatheFlowOrchestrator(
            stt=self.stt,
            brain=EmptyContentBrain(),
            tts=self.tts,
            controller=self.controller,
        )

        result = orchestrator.process_voice_turn(b"AUDIO")
        self.assertEqual(result.status, OrchestratorStatus.BRAIN_ERROR)
        self.assertIn("comfortable breath", result.spoken_response)
        self.assertIn("Brain reasoning failed", str(result.error_message))
        self.assertFalse(self.controller.is_active)
        self.assertEqual(self.controller.phase, BreathingPhase.IDLE)

    def test_active_session_rejects_simplified_protocol_switch(self) -> None:
        """Verify running controller rejects simplified protocol suggestion and preserves active session."""
        coherent = BreathingProtocol.coherent_breathing(cycles=5, duration_s=5.5)
        self.controller.start(coherent)

        simplified = {"action": "start_session", "protocol": "box_breathing"}
        brain = MockBrain(spoken="Switch to box breathing.", protocol=simplified)
        orchestrator = BreatheFlowOrchestrator(
            stt=self.stt,
            brain=brain,
            tts=self.tts,
            controller=self.controller,
        )

        result = orchestrator.process_voice_turn(b"AUDIO")
        self.assertEqual(result.status, OrchestratorStatus.SUCCESS)
        self.assertTrue(self.controller.is_active)
        self.assertEqual(self.controller.protocol.name, "Coherent Breathing")
        self.assertIsNone(result.controller_tick)

    def test_orchestrator_with_gemini_brain_adapter(self) -> None:
        """Verify orchestrator seamlessly runs with GeminiBrainAdapter."""
        mock_client = unittest.mock.MagicMock()
        mock_response = unittest.mock.MagicMock()
        mock_response.text = (
            '{"spoken_response": "Let\'s take a grounding box breath together.", '
            '"action": "start_session", "protocol": "box_breathing"}'
        )
        mock_client.models.generate_content.return_value = mock_response

        gemini_brain = GeminiBrainAdapter(
            api_key="test-key",
            model="gemini-3.5-flash-lite",
            client=mock_client,
        )
        orchestrator = BreatheFlowOrchestrator(
            stt=self.stt,
            brain=gemini_brain,
            tts=self.tts,
            controller=self.controller,
        )

        result = orchestrator.process_voice_turn(b"AUDIO")
        self.assertEqual(result.status, OrchestratorStatus.SUCCESS)
        self.assertEqual(
            result.spoken_response, "Let's take a grounding box breath together."
        )
        self.assertTrue(self.controller.is_active)
        self.assertEqual(self.controller.phase, BreathingPhase.INHALE)
        self.assertEqual(self.controller.protocol.name, "box_breathing")
        self.assertEqual(self.controller.protocol.inhale_s, 4.0)


if __name__ == "__main__":
    unittest.main()
