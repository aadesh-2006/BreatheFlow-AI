"""BreatheFlow AI — Dialogue and Session Orchestrator.

Coordinates voice input, conversational reasoning, voice synthesis,
and the deterministic breathing session controller into a coherent turn loop.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional

from breatheflow.config import BreatheFlowConfig, load_config
from breatheflow.core.controller import (
    BreathingPhase,
    BreathingProtocol,
    ControllerTick,
    DeterministicBreathingController,
)
from breatheflow.providers.base import (
    ConversationEngine,
    ConversationTurn,
    STTProvider,
    STTResult,
    TTSAudio,
    TTSProvider,
)
from breatheflow.providers.exceptions import ProviderError

logger = logging.getLogger(__name__)


class OrchestratorStatus(str, Enum):
    """Execution status of an orchestration voice turn."""

    SUCCESS = "success"
    EMPTY_TRANSCRIPT = "empty_transcript"
    STT_ERROR = "stt_error"
    BRAIN_ERROR = "brain_error"
    TTS_ERROR = "tts_error"


@dataclass(frozen=True)
class OrchestratorTurnResult:
    """Immutable result of a single voice interaction turn."""

    status: OrchestratorStatus
    transcript: str
    spoken_response: str
    audio_response: Optional[TTSAudio] = None
    suggested_protocol: Optional[Dict[str, Any]] = None
    controller_tick: Optional[ControllerTick] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BreatheFlowOrchestrator:
    """Coordinates STT, Brain, TTS, and Deterministic Controller."""

    def __init__(
        self,
        stt: STTProvider,
        brain: ConversationEngine,
        tts: TTSProvider,
        controller: Optional[DeterministicBreathingController] = None,
        config: Optional[BreatheFlowConfig] = None,
        max_history_turns: int = 50,
    ) -> None:
        if max_history_turns < 1:
            raise ValueError(f"max_history_turns must be at least 1, got {max_history_turns}")
        self._stt = stt
        self._brain = brain
        self._tts = tts
        self._controller = controller or DeterministicBreathingController()
        self._config = config or load_config()
        self._max_history_turns = max_history_turns
        self._turn_history: deque[Dict[str, str]] = deque(maxlen=max_history_turns)

    @property
    def controller(self) -> DeterministicBreathingController:
        """Access the deterministic breathing session controller."""
        return self._controller

    @property
    def max_history_turns(self) -> int:
        """Configured maximum number of conversational turns retained in memory."""
        return self._max_history_turns

    @property
    def turn_history(self) -> List[Dict[str, str]]:
        """Chronologically ordered history of conversational turns in current session."""
        return list(self._turn_history)

    def _build_session_context(self) -> Dict[str, Any]:
        """Construct session state context for conversational brain without diagnostic data."""
        active_protocol_name = (
            self._controller.protocol.name if self._controller.protocol else None
        )
        return {
            "controller_phase": self._controller.phase.value,
            "is_active": self._controller.is_active,
            "current_cycle": self._controller.current_cycle,
            "total_cycles": (
                self._controller.protocol.cycles if self._controller.protocol else 0
            ),
            "active_protocol": active_protocol_name,
        }

    def _try_initialize_protocol(
        self, suggested: Optional[Dict[str, Any]]
    ) -> Optional[ControllerTick]:
        """Safely start a recommended protocol on the controller IF currently idle/completed.

        Never alters timing of an already-active breathing session.
        """
        if not suggested or suggested.get("action") != "start_session":
            return None

        # Safety gate: LLM cannot interrupt or alter an active running cycle
        if self._controller.is_active or self._controller.phase == BreathingPhase.PAUSED:
            logger.info(
                "Controller is currently active (%s). Ignoring LLM protocol start suggestion.",
                self._controller.phase.value,
            )
            return None

        try:
            if "inhale_s" in suggested and "exhale_s" in suggested:
                protocol = BreathingProtocol(
                    name=suggested.get("protocol", "Guided Breathing"),
                    inhale_s=float(suggested["inhale_s"]),
                    hold_in_s=float(suggested.get("hold_in_s", 0.0)),
                    exhale_s=float(suggested["exhale_s"]),
                    hold_out_s=float(suggested.get("hold_out_s", 0.0)),
                    cycles=int(suggested.get("cycles", 4)),
                )
            else:
                proto_name = str(suggested.get("protocol", ""))
                preset = BreathingProtocol.from_name(proto_name)
                if not preset:
                    logger.warning("Unknown protocol recommendation: %s", proto_name)
                    return None
                protocol = preset

            return self._controller.start(protocol)
        except (ValueError, KeyError, TypeError) as e:
            logger.warning("Failed to initialize suggested protocol: %s", e)
            return None

    def process_voice_turn(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None,
    ) -> OrchestratorTurnResult:
        """Process a single turn: Audio In -> STT -> Brain -> TTS -> Audio Out.

        Args:
            audio_bytes: Inbound spoken user audio bytes.
            language_code: Optional language override (e.g. 'en-IN').

        Returns:
            OrchestratorTurnResult with transcript, spoken text, audio, and controller state.
        """
        lang = language_code or self._config.stt_language_code

        # 1. Speech-to-Text Transcription
        try:
            stt_result = self._stt.transcribe(audio_bytes, language_code=lang)
        except Exception as e:
            logger.error("STT provider failure: %s", e)
            return OrchestratorTurnResult(
                status=OrchestratorStatus.STT_ERROR,
                transcript="",
                spoken_response="",
                error_message=f"Speech transcription failed: {e}",
            )

        transcript = (stt_result.transcript or "").strip()

        # 2. Empty Transcript Handling
        if not transcript:
            fallback_text = "I didn't catch that. Could you please say that again?"
            try:
                audio_res = self._tts.synthesize(
                    text=fallback_text,
                    voice=self._config.tts_voice,
                    speed=self._config.tts_speed,
                    language_code=self._config.tts_language_code,
                )
            except Exception:
                audio_res = None

            return OrchestratorTurnResult(
                status=OrchestratorStatus.EMPTY_TRANSCRIPT,
                transcript="",
                spoken_response=fallback_text,
                audio_response=audio_res,
                metadata={"confidence": stt_result.confidence},
            )

        # 3. Conversational Reasoning via Brain
        context = self._build_session_context()
        try:
            brain_turn = self._brain.generate_turn(
                user_text=transcript, session_context=context
            )
            spoken_text = brain_turn.spoken_response
            suggested_proto = brain_turn.suggested_protocol
        except Exception as e:
            logger.error("Brain provider failure: %s", e)
            fallback_text = (
                "I'm here with you. Take a slow, comfortable breath in, and let it out gently."
            )
            try:
                audio_res = self._tts.synthesize(
                    text=fallback_text,
                    voice=self._config.tts_voice,
                    speed=self._config.tts_speed,
                    language_code=self._config.tts_language_code,
                )
            except Exception:
                audio_res = None

            return OrchestratorTurnResult(
                status=OrchestratorStatus.BRAIN_ERROR,
                transcript=transcript,
                spoken_response=fallback_text,
                audio_response=audio_res,
                error_message=f"Brain reasoning failed: {e}",
            )

        # 4. Controller Action Mapping (Deterministic & Guarded)
        tick = self._try_initialize_protocol(suggested_proto)

        # Record turn history
        self._turn_history.append({"user": transcript, "assistant": spoken_text})

        # 5. Text-to-Speech Synthesis
        try:
            tts_audio = self._tts.synthesize(
                text=spoken_text,
                voice=self._config.tts_voice,
                speed=self._config.tts_speed,
                language_code=self._config.tts_language_code,
            )
            tts_error = None
            status = OrchestratorStatus.SUCCESS
        except Exception as e:
            logger.error("TTS provider failure: %s", e)
            # Roll back protocol initialization if TTS fails so controller is not left in an orphaned active session
            if tick is not None and self._controller.is_active:
                self._controller.stop()
                tick = None
            tts_audio = None
            tts_error = f"Speech synthesis failed: {e}"
            status = OrchestratorStatus.TTS_ERROR

        return OrchestratorTurnResult(
            status=status,
            transcript=transcript,
            spoken_response=spoken_text,
            audio_response=tts_audio,
            suggested_protocol=suggested_proto,
            controller_tick=tick,
            error_message=tts_error,
            metadata={
                "stt_confidence": stt_result.confidence,
                "controller_phase": self._controller.phase.value,
            },
        )
