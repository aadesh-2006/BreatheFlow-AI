"""Gnani Prisma v2.5 Speech-to-Text (STT) REST Adapter.

Converts speech audio into text via Gnani's Vachana API platform.
"""

import json
import socket
import urllib.error
import urllib.request
import uuid
from typing import Optional

from breatheflow.providers.base import STTProvider, STTResult
from breatheflow.providers.exceptions import GnaniAPIError, GnaniTimeoutError


class GnaniSTTAdapter(STTProvider):
    """REST adapter for Gnani Prisma v2.5 Speech-to-Text."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.vachana.ai",
        default_model: str = "prisma-v2.5",
        default_language: str = "en-IN",
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key.strip() if api_key else None
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._default_language = default_language
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        """API base endpoint URL."""
        return self._base_url

    @property
    def default_language(self) -> str:
        """Default transcription language code."""
        return self._default_language

    def _build_multipart_payload(
        self,
        audio_bytes: bytes,
        language_code: str,
        model: str,
        boundary: str,
        filename: str = "audio.wav",
    ) -> bytes:
        """Construct a multipart/form-data payload using standard library byte concatenation."""
        lines = []

        # audio_file field
        lines.append(f"--{boundary}".encode("utf-8"))
        lines.append(
            f'Content-Disposition: form-data; name="audio_file"; filename="{filename}"'.encode("utf-8")
        )
        lines.append(b"Content-Type: audio/wav")
        lines.append(b"")
        lines.append(audio_bytes)

        # language_code field
        lines.append(f"--{boundary}".encode("utf-8"))
        lines.append(b'Content-Disposition: form-data; name="language_code"')
        lines.append(b"")
        lines.append(language_code.encode("utf-8"))

        # model field
        lines.append(f"--{boundary}".encode("utf-8"))
        lines.append(b'Content-Disposition: form-data; name="model"')
        lines.append(b"")
        lines.append(model.encode("utf-8"))

        # format field (transcribe mode for clean assistant text)
        lines.append(f"--{boundary}".encode("utf-8"))
        lines.append(b'Content-Disposition: form-data; name="format"')
        lines.append(b"")
        lines.append(b"transcribe")

        lines.append(f"--{boundary}--".encode("utf-8"))
        lines.append(b"")

        return b"\r\n".join(lines)

    def transcribe(
        self, audio_bytes: bytes, language_code: Optional[str] = None
    ) -> STTResult:
        """Transcribe audio bytes to text using Gnani Prisma v2.5.

        Args:
            audio_bytes: Binary audio payload.
            language_code: Target language (defaults to self._default_language).

        Returns:
            STTResult with transcribed text.

        Raises:
            ValueError: If api_key is missing or invalid.
            GnaniAPIError: If API returns error status code or malformed payload.
            GnaniTimeoutError: If request times out.
        """
        if not self._api_key:
            raise ValueError("Gnani API key is required for STT transcription.")

        if not audio_bytes or len(audio_bytes) == 0:
            return STTResult(transcript="", confidence=0.0)

        target_lang = language_code or self._default_language
        boundary = f"----BreatheFlowBoundary{uuid.uuid4().hex}"
        payload = self._build_multipart_payload(
            audio_bytes=audio_bytes,
            language_code=target_lang,
            model=self._default_model,
            boundary=boundary,
        )

        url = f"{self._base_url}/stt/v3"
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-API-Key-ID": self._api_key,
            "User-Agent": "BreatheFlow-AI/0.1",
        }

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                status_code = response.status
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = ""
            raise GnaniAPIError(
                f"Gnani STT API failed with HTTP {e.code}: {err_body}",
                status_code=e.code,
                response_body=err_body,
            ) from e
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            raise GnaniTimeoutError(f"Gnani STT request timed out after {self._timeout}s: {e}") from e
        except Exception as e:
            raise GnaniAPIError(f"Unexpected error during Gnani STT request: {e}") from e

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as e:
            raise GnaniAPIError(
                f"Invalid JSON returned from Gnani STT API: {body[:200]}",
                status_code=status_code,
                response_body=body,
            ) from e

        if not isinstance(parsed, dict):
            raise GnaniAPIError("Malformed response: expected JSON object", status_code=status_code)

        # Check for explicit failure flag in response
        if parsed.get("success") is False:
            err_msg = parsed.get("message") or parsed.get("error") or "Unknown API error"
            raise GnaniAPIError(f"Gnani STT reported failure: {err_msg}", status_code=status_code)

        transcript = parsed.get("transcript", "")
        detected_lang = parsed.get("detected_language_code")
        confidence = parsed.get("confidence")

        metadata = {
            "request_id": parsed.get("request_id"),
            "processing_time": parsed.get("processing_time"),
            "model": parsed.get("model", self._default_model),
        }

        return STTResult(
            transcript=transcript,
            detected_language=detected_lang,
            confidence=float(confidence) if confidence is not None else None,
            metadata=metadata,
        )
