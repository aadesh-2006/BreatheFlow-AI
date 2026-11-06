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

from breatheflow.providers.base import ConversationEngine, ConversationTurn
from breatheflow.providers.exceptions import OllamaConnectionError, OllamaResponseError

SYSTEM_PROMPT = """You are BreatheFlow AI, a voice-first adaptive breathing companion.
Your mission is to guide users through calming, evidence-based breathing exercises with warmth, empathy, and brevity.

CORE GUIDELINES:
1. VOICE-FIRST CONCISENESS: Keep your spoken responses under 40 words (1 to 3 short, calming sentences). Never output long paragraphs or markdown bullet lists, as your text is synthesized directly into speech.
2. TECHNIQUE MATCHING:
   - Box Breathing (4s Inhale, 4s Hold, 4s Exhale, 4s Hold): for focus, grounding, or pre-task anxiety.
   - 4-7-8 Breathing (4s Inhale, 7s Hold, 8s Exhale, 0s Hold): for relaxation, sleep, or deep unwinding.
   - Coherent Breathing (5.5s Inhale, 0s Hold, 5.5s Exhale, 0s Hold): for steady autonomic balance.
   - Physiological Sigh (2.5s Inhale, 1s Hold, 6s Exhale, 0s Hold): for rapid acute reset.
3. STRUCTURED PROTOCOL OUTPUT: When recommending or initiating a breathing session, output your short spoken guidance followed by a JSON block:
```json
{
  "action": "start_session",
  "protocol": "box_breathing",
  "inhale_s": 4.0,
  "hold_in_s": 4.0,
  "exhale_s": 4.0,
  "hold_out_s": 4.0,
  "cycles": 4
}
```
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
        timeout: float = 30.0,
    ) -> None:
        self._host = ollama_host.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout

    @property
    def host(self) -> str:
        """Ollama server host URL."""
        return self._host

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    def _validate_protocol_payload(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate protocol command structure and parameter boundaries.

        Returns valid payload dictionary or None if invalid.
        """
        if not isinstance(data, dict):
            return None

        action = data.get("action")
        if action != "start_session":
            return None

        protocol_name = data.get("protocol")
        if not protocol_name or not isinstance(protocol_name, str):
            return None

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
            context_str = json.dumps(session_context)
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
                "num_predict": 150,
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

        spoken, protocol = self._extract_spoken_and_protocol(raw_content)

        return ConversationTurn(
            spoken_response=spoken,
            suggested_protocol=protocol,
            raw_response=raw_content,
        )
