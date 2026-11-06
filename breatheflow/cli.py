"""Command-line interface (CLI) for running BreatheFlow AI voice turns.

Supports testing with live providers (Gnani Prisma, Gnani Timbre, Ollama Qwen3-4B)
or offline mock providers for testing environments.
"""

import argparse
import os
import sys
from typing import Optional

from breatheflow.config import load_config
from breatheflow.core.controller import DeterministicBreathingController
from breatheflow.core.orchestrator import BreatheFlowOrchestrator, OrchestratorStatus
from breatheflow.providers.base import (
    ConversationEngine,
    ConversationTurn,
    STTProvider,
    STTResult,
    TTSAudio,
    TTSProvider,
)
from breatheflow.providers.gnani_stt import GnaniSTTAdapter
from breatheflow.providers.gnani_tts import GnaniTTSAdapter
from breatheflow.providers.ollama_brain import OllamaBrainAdapter


class CLIMockSTT(STTProvider):
    """Offline mock STT provider for CLI testing."""

    def transcribe(
        self, audio_bytes: bytes, language_code: Optional[str] = None
    ) -> STTResult:
        return STTResult(
            transcript="I am feeling anxious before my exam. Can we do box breathing?",
            detected_language="en-IN",
            confidence=0.99,
        )


class CLIMockBrain(ConversationEngine):
    """Offline mock conversational brain for CLI testing."""

    def generate_turn(
        self, user_text: str, session_context: Optional[dict] = None
    ) -> ConversationTurn:
        spoken = "I hear you. Let's do Box Breathing together to steady your focus. Sit comfortably."
        protocol = {
            "action": "start_session",
            "protocol": "box_breathing",
            "inhale_s": 4.0,
            "hold_in_s": 4.0,
            "exhale_s": 4.0,
            "hold_out_s": 4.0,
            "cycles": 4,
        }
        return ConversationTurn(spoken_response=spoken, suggested_protocol=protocol)


class CLIMockTTS(TTSProvider):
    """Offline mock TTS provider for CLI testing."""

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        language_code: Optional[str] = None,
    ) -> TTSAudio:
        # Minimal valid 44-byte WAV header + silent PCM bytes
        wav_header = (
            b"RIFF\x2c\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
            b"\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x08\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00"
        )
        return TTSAudio(
            audio_bytes=wav_header,
            content_type="audio/wav",
            sample_rate=16000,
            duration_s=1.0,
        )


def main(argv: Optional[list[str]] = None) -> int:
    """CLI Entry point."""
    parser = argparse.ArgumentParser(
        description="BreatheFlow AI — Voice-First Adaptive Breathing Companion CLI"
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        help="Path to an input audio file (WAV format recommended).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output_response.wav",
        help="Path to save synthesized audio response (default: output_response.wav).",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Language code override (e.g. en-IN, hi-IN).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run using offline mock providers (does not require live Gnani API key or Ollama).",
    )

    args = parser.parse_args(argv)
    config = load_config()

    print("==================================================")
    print("  BreatheFlow AI — Voice Companion CLI")
    print("==================================================")

    if args.mock:
        print("[Mode] Running with MOCK providers (offline test).")
        stt = CLIMockSTT()
        brain = CLIMockBrain()
        tts = CLIMockTTS()
    else:
        print("[Mode] Running with LIVE providers.")
        if not config.gnani_api_key:
            print(
                "[ERROR] GNANI_API_KEY is not set in environment or .env!\n"
                "Please set GNANI_API_KEY or run with --mock for offline testing.",
                file=sys.stderr,
            )
            return 1

        stt = GnaniSTTAdapter(
            api_key=config.gnani_api_key,
            base_url=config.gnani_base_url,
            default_model=config.stt_model,
            default_language=config.stt_language_code,
        )
        tts = GnaniTTSAdapter(
            api_key=config.gnani_api_key,
            base_url=config.gnani_base_url,
            default_model=config.tts_model,
            default_voice=config.tts_voice,
            default_language=config.tts_language_code,
            default_speed=config.tts_speed,
        )
        brain = OllamaBrainAdapter(
            ollama_host=config.ollama_host,
            model=config.ollama_model,
        )

    # Read audio input
    if args.audio:
        if not os.path.exists(args.audio):
            print(f"[ERROR] Audio file not found: {args.audio}", file=sys.stderr)
            return 1
        with open(args.audio, "rb") as f:
            audio_bytes = f.read()
        print(f"[Input] Loaded {len(audio_bytes)} bytes from {args.audio}")
    else:
        if args.mock:
            audio_bytes = b"MOCK_AUDIO_PAYLOAD"
            print("[Input] Using simulated input audio.")
        else:
            print(
                "[ERROR] --audio path must be provided when running in live mode.",
                file=sys.stderr,
            )
            return 1

    controller = DeterministicBreathingController()
    orchestrator = BreatheFlowOrchestrator(
        stt=stt,
        brain=brain,
        tts=tts,
        controller=controller,
        config=config,
    )

    print("\n--- Processing Voice Turn ---")
    result = orchestrator.process_voice_turn(
        audio_bytes=audio_bytes,
        language_code=args.language,
    )

    print(f"Status:             {result.status.value}")
    print(f"Recognized Speech:  \"{result.transcript}\"")
    print(f"Assistant Response: \"{result.spoken_response}\"")

    if result.suggested_protocol:
        proto = result.suggested_protocol
        print(
            f"Breathing Protocol: {proto.get('protocol')} "
            f"(Inhale: {proto.get('inhale_s')}s, Hold: {proto.get('hold_in_s')}s, "
            f"Exhale: {proto.get('exhale_s')}s, Cycles: {proto.get('cycles')})"
        )

    if result.controller_tick:
        tick = result.controller_tick
        print(
            f"Controller State:   Phase: {tick.phase.value}, "
            f"Cycle: {tick.current_cycle}/{tick.total_cycles}, "
            f"Phase Remaining: {tick.phase_remaining_s}s"
        )
    else:
        print(f"Controller State:   Phase: {controller.phase.value}")

    if result.audio_response:
        try:
            with open(args.output, "wb") as f:
                f.write(result.audio_response.audio_bytes)
            print(
                f"Audio Output:       Saved {len(result.audio_response.audio_bytes)} bytes to {args.output}"
            )
        except OSError as e:
            print(
                f"[ERROR] Failed to write audio output file '{args.output}': {e}",
                file=sys.stderr,
            )
            return 1

    if result.error_message:
        print(f"[Notice] {result.error_message}", file=sys.stderr)

    print("-----------------------------\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
