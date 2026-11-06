"""Gnani Timbre v2.5 Text-to-Speech (TTS) REST Adapter.

Synthesizes spoken guidance into audio bytes via Gnani's Vachana API platform.
"""

import json
import socket
import urllib.error
import urllib.request
from typing import Optional

from breatheflow.providers.base import TTSAudio, TTSProvider
from breatheflow.providers.exceptions import GnaniAPIError, GnaniTimeoutError


class GnaniTTSAdapter(TTSProvider):
    """REST adapter for Gnani Timbre v2.5 Text-to-Speech."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.vachana.ai",
        default_model: str = "timbre-v2.5",
        default_voice: str = "Kaveri",
        default_language: str = "en-IN",
        default_speed: float = 0.95,
        sample_rate: int = 24000,
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key.strip() if api_key else None
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._default_voice = default_voice
        self._default_language = default_language
        self._default_speed = default_speed
        self._sample_rate = sample_rate
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        """API base endpoint URL."""
        return self._base_url

    @property
    def default_voice(self) -> str:
        """Default voice persona."""
        return self._default_voice

    @property
    def default_language(self) -> str:
        """Default language code."""
        return self._default_language

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        language_code: Optional[str] = None,
    ) -> TTSAudio:
        """Synthesize text into WAV audio bytes.

        Args:
            text: Text string to synthesize.
            voice: Voice persona (e.g. 'Kaveri' for en-IN, 'Nalini' for hi-IN).
            speed: Speaking rate multiplier (default 0.95 for calm pacing).
            language_code: Language code target (default 'en-IN').

        Returns:
            TTSAudio containing audio bytes and format metadata.

        Raises:
            ValueError: If text or api_key is missing/empty.
            GnaniAPIError: If API returns error status or empty audio payload.
            GnaniTimeoutError: If request times out.
        """
        if not self._api_key:
            raise ValueError("Gnani API key is required for TTS synthesis.")

        if not text or not text.strip():
            raise ValueError("Text to synthesize cannot be empty.")

        target_voice = voice or self._default_voice
        target_lang = language_code or self._default_language
        target_speed = speed if speed is not None else self._default_speed

        payload = {
            "text": text.strip(),
            "model": self._default_model,
            "language": target_lang,
            "voice": target_voice,
            "speed": target_speed,
            "audio_config": {
                "sample_rate": self._sample_rate,
                "num_channels": 1,
                "encoding": "linear_pcm",
                "container": "wav",
            },
        }

        url = f"{self._base_url}/api/v1/tts/inference"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key-ID": self._api_key,
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                status_code = response.status
                content_type = response.headers.get("Content-Type", "audio/wav")
                audio_bytes = response.read()
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = ""
            raise GnaniAPIError(
                f"Gnani TTS API failed with HTTP {e.code}: {err_body}",
                status_code=e.code,
                response_body=err_body,
            ) from e
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            raise GnaniTimeoutError(f"Gnani TTS request timed out after {self._timeout}s: {e}") from e
        except Exception as e:
            raise GnaniAPIError(f"Unexpected error during Gnani TTS request: {e}") from e

        if not audio_bytes or len(audio_bytes) == 0:
            raise GnaniAPIError(
                "Received empty audio response from Gnani TTS API.",
                status_code=status_code,
            )

        # Validate that the response is binary audio, not a JSON error response returned with HTTP 200
        is_json_header = "application/json" in content_type.lower()
        starts_with_json = audio_bytes.strip().startswith((b"{", b"["))

        if is_json_header or starts_with_json:
            try:
                decoded_str = audio_bytes.decode("utf-8")
                parsed = json.loads(decoded_str)
                if isinstance(parsed, dict):
                    err_msg = (
                        parsed.get("message")
                        or parsed.get("error")
                        or parsed.get("detail")
                        or "API returned JSON response instead of audio"
                    )
                    raise GnaniAPIError(
                        f"Gnani TTS returned error response: {err_msg}",
                        status_code=status_code,
                        response_body=decoded_str,
                    )
            except UnicodeDecodeError:
                pass
            except json.JSONDecodeError:
                if is_json_header:
                    raise GnaniAPIError(
                        "Gnani TTS returned malformed JSON with Content-Type application/json",
                        status_code=status_code,
                        response_body=audio_bytes.decode("utf-8", errors="replace"),
                    )

        return TTSAudio(
            audio_bytes=audio_bytes,
            content_type=content_type,
            sample_rate=self._sample_rate,
        )
