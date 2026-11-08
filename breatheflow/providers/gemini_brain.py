"""Google Gemini API Conversational Brain Adapter.

Provides voice-first conversational intelligence, technique recommendation,
and structured protocol output using the official Google Gen AI Python SDK.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from breatheflow.core.controller import BreathingProtocol
from breatheflow.providers.base import ConversationEngine, ConversationTurn
from breatheflow.providers.exceptions import (
    GeminiAuthError,
    GeminiConnectionError,
    GeminiError,
    GeminiRateLimitError,
    GeminiResponseError,
)

logger = logging.getLogger(__name__)

GEMINI_SYSTEM_PROMPT = """You are BreatheFlow AI, a voice-first adaptive breathing companion.
Your mission is to guide users through calming, evidence-based breathing exercises with warmth, empathy, and brevity.

CORE GUIDELINES:
1. VOICE-FIRST CONCISENESS: Keep your spoken responses under 40 words (1 to 3 short, calming sentences). Never output bullet points, long paragraphs, or markdown formatting in spoken responses, as your text is synthesized directly into speech.
2. TECHNIQUE MATCHING:
   - Box Breathing: for focus, grounding, or pre-task anxiety (protocol: "box_breathing").
   - 4-7-8 Breathing: for relaxation, sleep, or deep unwinding (protocol: "four_seven_eight").
   - Coherent Breathing: for steady autonomic balance (protocol: "coherent_breathing").
   - Physiological Sigh: for rapid acute reset (protocol: "physiological_sigh").
3. PROTOCOL RECOMMENDATION:
   - When recommending or initiating a breathing session, specify action="start_session" and the protocol identifier.
   - Do not specify timing numbers or cycle counts; standard protocols are pre-configured.
   - If no session is being initiated (e.g. conversational check-in or debrief), leave action and protocol null.
4. STRICT SAFETY & NON-MEDICAL BOUNDARIES:
   - You are an educational wellness companion, NOT a medical doctor or diagnostic tool.
   - NEVER make clinical diagnoses (e.g., do not say "You have panic disorder" or "Your pulse indicates arrhythmia").
   - You CANNOT change active session timers directly. All timing is executed by the deterministic controller.
   - If the user reports severe physical distress (chest pain, severe shortness of breath) or a psychological crisis, compassionately advise them to seek professional medical emergency care immediately.
"""


class BreathingTurnSchema(BaseModel):
    """Structured response schema for BreatheFlow conversational turn."""

    spoken_response: str = Field(
        description="Empathetic, calming spoken guidance for the user. Strictly under 40 words. No bullet points or markdown."
    )
    action: Optional[str] = Field(
        default=None,
        description="Set to 'start_session' when recommending or initiating a breathing technique; otherwise null.",
    )
    protocol: Optional[str] = Field(
        default=None,
        description="Standard breathing technique identifier: 'box_breathing', 'four_seven_eight', 'coherent_breathing', or 'physiological_sigh'. Null if no session is being initiated.",
    )


class GeminiBrainAdapter(ConversationEngine):
    """Conversational reasoning engine backed by the Google Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.5-flash-lite",
        temperature: float = 0.6,
        timeout: float = 30.0,
        client: Optional[Any] = None,
    ) -> None:
        raw_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self._api_key = raw_key.strip() if raw_key else None
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._client = client

    @property
    def model(self) -> str:
        """Active Gemini model identifier."""
        return self._model

    @property
    def timeout(self) -> float:
        """Request timeout in seconds."""
        return self._timeout

    def __repr__(self) -> str:
        """Mask API key in string representation to prevent leakage."""
        key_status = "<SET>" if self._api_key else "<NOT SET>"
        return (
            f"GeminiBrainAdapter("
            f"model='{self._model}', "
            f"api_key='{key_status}', "
            f"temperature={self._temperature}, "
            f"timeout={self._timeout})"
        )

    def _get_client(self) -> Any:
        """Retrieve existing client or initialize Google Gen AI SDK client."""
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise GeminiAuthError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key."
            )
        try:
            from google import genai

            return genai.Client(api_key=self._api_key)
        except Exception as e:
            raise GeminiConnectionError(
                f"Failed to initialize Google Gen AI client: {e}"
            ) from e

    def _format_session_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format session context into a concise key-value string."""
        if not context:
            return ""

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

        parts = []
        for k, v in context.items():
            if v is not None:
                parts.append(f"{k}: {v}")
        return ", ".join(parts)

    def _parse_turn_response(
        self, raw_text: str
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        """Parse structured LLM completion into clean spoken text and validated protocol."""
        data: Optional[Dict[str, Any]] = None

        # 1. Direct JSON parse
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # 2. Markdown fence fallback
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    data = None

        if isinstance(data, dict):
            spoken = str(data.get("spoken_response", "")).strip()
            action = data.get("action")
            protocol_name = data.get("protocol")

            # Fallback if spoken_response key was missing but plain text exists
            if not spoken:
                clean_text = re.sub(r"```.*?```", "", raw_text, flags=re.DOTALL).strip()
                spoken = clean_text or "Let's take a comfortable breath together."

            suggested_protocol = None
            if action == "start_session" and protocol_name and isinstance(protocol_name, str):
                preset = BreathingProtocol.from_name(protocol_name)
                if preset is not None:
                    suggested_protocol = {
                        "action": "start_session",
                        "protocol": protocol_name.strip(),
                        "inhale_s": preset.inhale_s,
                        "hold_in_s": preset.hold_in_s,
                        "exhale_s": preset.exhale_s,
                        "hold_out_s": preset.hold_out_s,
                        "cycles": preset.cycles,
                    }
                else:
                    logger.warning(
                        "Gemini recommended unrecognized breathing protocol: %s",
                        protocol_name,
                    )
                    suggested_protocol = None

            return spoken, suggested_protocol

        # 3. Unstructured text fallback
        clean_spoken = re.sub(r"```.*?```", "", raw_text, flags=re.DOTALL).strip()
        if not clean_spoken:
            clean_spoken = "Let's take a comfortable breath together."
        return clean_spoken, None

    def generate_turn(
        self,
        user_text: str,
        session_context: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Query Google Gemini API for empathetic guidance and protocol recommendation.

        Args:
            user_text: Spoken utterance transcribed from user.
            session_context: Optional context metadata (e.g. current activity, session status).

        Returns:
            ConversationTurn with spoken response and optional validated protocol dict.

        Raises:
            GeminiAuthError: If API key is missing or authentication fails.
            GeminiRateLimitError: If API quota or rate limits are exceeded.
            GeminiConnectionError: If network connection or timeout occurs.
            GeminiResponseError: If response is empty or malformed.
        """
        client = self._get_client()

        user_content = user_text.strip()
        if session_context:
            context_str = self._format_session_context(session_context)
            if context_str:
                user_content = f"[Context: {context_str}]\n{user_content}"

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                system_instruction=GEMINI_SYSTEM_PROMPT,
                temperature=self._temperature,
                response_mime_type="application/json",
                response_schema=BreathingTurnSchema,
                http_options=types.HttpOptions(timeout=int(self._timeout * 1000)),
            )

            response = client.models.generate_content(
                model=self._model,
                contents=user_content,
                config=config,
            )
        except GeminiError:
            raise
        except Exception as e:
            err_msg = str(e)
            err_lower = err_msg.lower()

            if "429" in err_msg or "resource_exhausted" in err_lower or "quota" in err_lower or "rate limit" in err_lower:
                raise GeminiRateLimitError(
                    f"Gemini API rate limit or quota exceeded: {e}"
                ) from e

            if "401" in err_msg or "403" in err_msg or "unauthorized" in err_lower or "api_key_invalid" in err_lower or "api key not valid" in err_lower:
                raise GeminiAuthError(f"Gemini authentication failed: {e}") from e

            if "timeout" in err_lower or "timed out" in err_lower or "connection" in err_lower or "connect" in err_lower:
                raise GeminiConnectionError(
                    f"Failed to communicate with Gemini API: {e}"
                ) from e

            raise GeminiResponseError(f"Error from Gemini API: {e}") from e

        raw_text = getattr(response, "text", None)
        if not raw_text or not raw_text.strip():
            raise GeminiResponseError(
                "Gemini API returned empty response content."
            )

        spoken, protocol = self._parse_turn_response(raw_text)

        return ConversationTurn(
            spoken_response=spoken,
            suggested_protocol=protocol,
            raw_response=raw_text,
        )
