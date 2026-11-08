"""Local Ollama / Qwen3-4B Conversational Brain Adapter.

Provides voice-first conversational intelligence, technique recommendation,
and structured protocol output using the local Ollama runtime.
"""

import json
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from breatheflow.core.controller import BreathingProtocol
from breatheflow.providers.base import ConversationEngine, ConversationTurn
from breatheflow.providers.exceptions import OllamaConnectionError, OllamaResponseError

SYSTEM_PROMPT = """You are BreatheFlow AI, a voice-first adaptive breathing companion.
Your mission is to guide users through calming, evidence-based breathing exercises with warmth, empathy, and brevity.

CORE GUIDELINES:
1. VOICE-FIRST CONCISENESS: Keep internal reasoning minimal. Keep your spoken responses under 40 words (1 to 3 short, calming sentences). Never output long paragraphs or markdown bullet lists, as your text is synthesized directly into speech.
2. TECHNIQUE MATCHING:
   - Box Breathing: for focus, grounding, or pre-task anxiety (identifier: "box_breathing").
   - 4-7-8 Breathing: for relaxation, sleep, or deep unwinding (identifier: "four_seven_eight").
   - Coherent Breathing: for steady autonomic balance (identifier: "coherent_breathing").
   - Physiological Sigh: for rapid acute reset (identifier: "physiological_sigh").
3. STRUCTURED PROTOCOL OUTPUT: When recommending or initiating a breathing session, output your short spoken guidance followed by a minimal JSON block:
```json
{
  "action": "start_session",
  "protocol": "box_breathing"
}
```
Do not output timing values or cycle counts; standard protocols are pre-configured.
4. STRICT SAFETY & NON-MEDICAL BOUNDARIES:
   - You are an educational wellness companion, NOT a medical doctor or diagnostic tool.
   - NEVER make clinical diagnoses (e.g., do not say "You have panic disorder" or "Your pulse indicates arrhythmia").
   - You CANNOT change active session timers directly. All timing is executed by the deterministic controller.
   - If the user reports severe physical distress (chest pain, severe shortness of breath) or a psychological crisis, compassionately advise them to seek professional medical emergency care immediately.
"""


class OllamaBrainAdapter(ConversationEngine):
    """Conversational reasoning engine backed by local Ollama Qwen3-4B."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        temperature: float = 0.6,
        timeout: float = 60.0,
        num_predict: int = 512,
    ) -> None:
        self._host = ollama_host.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._num_predict = num_predict

    @property
    def host(self) -> str:
        """Ollama server host URL."""
        return self._host

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    @property
    def timeout(self) -> float:
        """HTTP request timeout in seconds."""
        return self._timeout

    @property
    def num_predict(self) -> int:
        """Token generation limit."""
        return self._num_predict

    def _validate_protocol_payload(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate protocol command structure and parameter boundaries.

        Supports both simplified recommendations ({"action": "start_session", "protocol": "box_breathing"})
        and explicit parameterizations with full durations and cycle counts.
        """
        if not isinstance(data, dict):
            return None

        action = data.get("action")
        if action != "start_session":
            return None

        protocol_name = data.get("protocol")
        if not protocol_name or not isinstance(protocol_name, str):
            return None

        # Check if full explicit parameters are provided (backward compatibility)
        has_custom_timings = "inhale_s" in data and "exhale_s" in data

        if has_custom_timings:
            try:
                inhale_s = float(data.get("inhale_s", 0))
                hold_in_s = float(data.get("hold_in_s", 0))
                exhale_s = float(data.get("exhale_s", 0))
                hold_out_s = float(data.get("hold_out_s", 0))
                cycles = int(data.get("cycles", 0))

                if inhale_s <= 0 or exhale_s <= 0 or hold_in_s < 0 or hold_out_s < 0:
                    return None
                if cycles < 1 or cycles > 50:
                    return None

                return {
                    "action": action,
                    "protocol": protocol_name.strip(),
                    "inhale_s": inhale_s,
                    "hold_in_s": hold_in_s,
                    "exhale_s": exhale_s,
                    "hold_out_s": hold_out_s,
                    "cycles": cycles,
                }
            except (ValueError, TypeError):
                return None

        # Simplified recommendation: resolve using existing deterministic protocol presets
        preset = BreathingProtocol.from_name(protocol_name)
        if preset is None:
            return None

        return {
            "action": action,
            "protocol": protocol_name.strip(),
            "inhale_s": preset.inhale_s,
            "hold_in_s": preset.hold_in_s,
            "exhale_s": preset.exhale_s,
            "hold_out_s": preset.hold_out_s,
            "cycles": preset.cycles,
        }

    def _extract_spoken_and_protocol(self, text: str) -> tuple[str, Optional[Dict[str, Any]]]:
        """Extract clean speech text and parsed protocol JSON from raw LLM completion."""
        # Find json block: ```json ... ```
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, text, re.DOTALL)

        suggested_protocol = None
        if match:
            raw_json = match.group(1)
            try:
                parsed = json.loads(raw_json)
                suggested_protocol = self._validate_protocol_payload(parsed)
            except json.JSONDecodeError:
                suggested_protocol = None

        # Clean spoken text by stripping out code blocks and excessive whitespace
        clean_spoken = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
        # Fallback if the entire response was inside the code block
        if not clean_spoken:
            clean_spoken = "Let's take a comfortable breath together."

        return clean_spoken, suggested_protocol

    def _format_session_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format session context into a concise, token-efficient key-value string.

        Avoids dumping raw JSON syntax (braces, quotes, nulls, zero-cycles)
        that inflates prompts and triggers excessive reasoning tokens.
        """
        if not context:
            return ""

        # Specialized concise handling for orchestrator session state
        if "controller_phase" in context:
            phase = str(context.get("controller_phase", "idle"))
            is_active = bool(context.get("is_active", False))
            parts = [f"controller_phase: {phase}", f"is_active: {str(is_active).lower()}"]

            active_protocol = context.get("active_protocol")
            if active_protocol:
                parts.append(f"protocol: {active_protocol}")

            total_cycles = context.get("total_cycles", 0)
            if total_cycles and total_cycles > 0:
                current_cycle = context.get("current_cycle", 1)
                parts.append(f"cycle: {current_cycle}/{total_cycles}")

            # Include any other non-standard keys passed in context
            standard_keys = {
                "controller_phase",
                "is_active",
                "active_protocol",
                "total_cycles",
                "current_cycle",
            }
            for k, v in context.items():
                if k not in standard_keys and v is not None:
                    parts.append(f"{k}: {v}")

            return ", ".join(parts)

        # General dictionary formatting for other callers
        parts = []
        for k, v in context.items():
            if v is not None:
                parts.append(f"{k}: {v}")
        return ", ".join(parts)

    def generate_turn(
        self,
        user_text: str,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Query local Ollama Qwen3-4B for empathetic guidance and protocol recommendation.

        Args:
            user_text: Spoken utterance transcribed from user.
            session_context: Optional context metadata (e.g. current activity, session status).

        Returns:
            ConversationTurn with spoken response and optional validated protocol dict.

        Raises:
            OllamaConnectionError: If unable to reach Ollama endpoint.
            OllamaResponseError: If response format is invalid.
        """
        user_content = user_text.strip()
        if session_context:
            context_str = self._format_session_context(session_context)
            if context_str:
                user_content = f"[Context: {context_str}]\n{user_content}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._num_predict,
            },
        }

        url = f"{self._host}/api/chat"
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            raise OllamaConnectionError(
                f"Failed to connect to local Ollama server at {self._host}: {e}"
            ) from e
        except Exception as e:
            raise OllamaResponseError(f"Unexpected error communicating with Ollama: {e}") from e

        try:
            parsed = json.loads(body)
            raw_content = parsed["message"]["content"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise OllamaResponseError(
                f"Invalid or malformed response from Ollama API: {body[:200]}"
            ) from e

        if not raw_content or not raw_content.strip():
            raise OllamaResponseError(
                "Ollama completed generation with empty content (generation truncated or token budget exhausted)."
            )

        spoken, protocol = self._extract_spoken_and_protocol(raw_content)

        return ConversationTurn(
            spoken_response=spoken,
            suggested_protocol=protocol,
            raw_response=raw_content,
        )
