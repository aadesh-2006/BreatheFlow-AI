"""Mocked HTTP tests for Gnani Prisma STT and Gnani Timbre TTS adapters."""

import io
import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch
import unittest

from breatheflow.providers.exceptions import GnaniAPIError, GnaniTimeoutError
from breatheflow.providers.gnani_stt import GnaniSTTAdapter
from breatheflow.providers.gnani_tts import GnaniTTSAdapter


class TestGnaniSTTAdapter(unittest.TestCase):
    """Test Gnani Prisma v2.5 STT adapter with mocked HTTP transport."""

    def setUp(self) -> None:
        self.api_key = "test_gnani_secret_key"
        self.adapter = GnaniSTTAdapter(api_key=self.api_key)

    def test_missing_api_key_raises_value_error(self) -> None:
        adapter = GnaniSTTAdapter(api_key=None)
        with self.assertRaises(ValueError):
            adapter.transcribe(b"dummy_audio_bytes")

    def test_empty_audio_returns_empty_transcript(self) -> None:
        res = self.adapter.transcribe(b"")
        self.assertEqual(res.transcript, "")
        self.assertEqual(res.confidence, 0.0)

    @patch("urllib.request.urlopen")
    def test_successful_transcription(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_payload = {
            "success": True,
            "request_id": "req-12345",
            "transcript": "Hello, I want to begin box breathing.",
            "model": "prisma-v2.5",
            "detected_language_code": "en-IN",
            "processing_time": 0.45,
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        audio_bytes = b"RIFF" + b"\x00" * 32
        result = self.adapter.transcribe(audio_bytes, language_code="en-IN")

        self.assertEqual(result.transcript, "Hello, I want to begin box breathing.")
        self.assertEqual(result.detected_language, "en-IN")
        self.assertEqual(result.metadata["request_id"], "req-12345")
        self.assertEqual(result.metadata["processing_time"], 0.45)

        # Verify request headers and URL
        called_req = mock_urlopen.call_args[0][0]
        self.assertEqual(called_req.get_header("X-api-key-id"), self.api_key)
        self.assertIn("multipart/form-data", called_req.get_header("Content-type"))
        self.assertTrue(called_req.full_url.endswith("/stt/v3"))

    @patch("urllib.request.urlopen")
    def test_http_401_unauthorized_error(self, mock_urlopen: MagicMock) -> None:
        err = urllib.error.HTTPError(
            url="https://api.vachana.ai/stt/v3",
            code=401,
            msg="Unauthorized",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error": "Invalid API Key"}'),
        )
        mock_urlopen.side_effect = err

        with self.assertRaises(GnaniAPIError) as ctx:
            self.adapter.transcribe(b"sample_audio")

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("401", str(ctx.exception))
        # Ensure secret key is never leaked in error message
        self.assertNotIn(self.api_key, str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_timeout_raises_gnani_timeout_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = TimeoutError("Connection timed out")

        with self.assertRaises(GnaniTimeoutError):
            self.adapter.transcribe(b"sample_audio")

    @patch("urllib.request.urlopen")
    def test_malformed_json_response(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b"<html>502 Bad Gateway</html>"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with self.assertRaises(GnaniAPIError):
            self.adapter.transcribe(b"sample_audio")

    @patch("urllib.request.urlopen")
    def test_api_reported_failure_flag(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_payload = {
            "success": False,
            "message": "Unsupported sample rate",
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with self.assertRaises(GnaniAPIError) as ctx:
            self.adapter.transcribe(b"sample_audio")

        self.assertIn("Unsupported sample rate", str(ctx.exception))


class TestGnaniTTSAdapter(unittest.TestCase):
    """Test Gnani Timbre v2.5 TTS adapter with mocked HTTP transport."""

    def setUp(self) -> None:
        self.api_key = "test_gnani_secret_key"
        self.adapter = GnaniTTSAdapter(api_key=self.api_key)

    def test_missing_api_key_raises_value_error(self) -> None:
        adapter = GnaniTTSAdapter(api_key=None)
        with self.assertRaises(ValueError):
            adapter.synthesize("Take a deep breath.")

    def test_empty_text_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.adapter.synthesize("")
        with self.assertRaises(ValueError):
            self.adapter.synthesize("   ")

    @patch("urllib.request.urlopen")
    def test_successful_synthesis(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "audio/wav"}
        simulated_audio = b"RIFF" + b"\x00" * 44 + b"PCM_AUDIO_DATA"
        mock_response.read.return_value = simulated_audio
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = self.adapter.synthesize(
            text="Inhale slowly through your nose.",
            voice="Kaveri",
            speed=0.9,
            language_code="en-IN",
        )

        self.assertEqual(result.audio_bytes, simulated_audio)
        self.assertEqual(result.content_type, "audio/wav")
        self.assertEqual(result.sample_rate, 24000)

        # Inspect request
        called_req = mock_urlopen.call_args[0][0]
        self.assertEqual(called_req.get_header("X-api-key-id"), self.api_key)
        self.assertEqual(called_req.get_header("Content-type"), "application/json")
        self.assertTrue(called_req.full_url.endswith("/api/v1/tts/inference"))

        # Verify JSON body parameters
        sent_body = json.loads(called_req.data.decode("utf-8"))
        self.assertEqual(sent_body["text"], "Inhale slowly through your nose.")
        self.assertEqual(sent_body["voice"], "Kaveri")
        self.assertEqual(sent_body["speed"], 0.9)
        self.assertEqual(sent_body["language"], "en-IN")
        self.assertEqual(sent_body["model"], "timbre-v2.5")

    @patch("urllib.request.urlopen")
    def test_empty_audio_response_raises_error(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "audio/wav"}
        mock_response.read.return_value = b""
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with self.assertRaises(GnaniAPIError) as ctx:
            self.adapter.synthesize("Inhale.")

        self.assertIn("empty audio response", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_http_error_handling(self, mock_urlopen: MagicMock) -> None:
        err = urllib.error.HTTPError(
            url="https://api.vachana.ai/api/v1/tts/inference",
            code=403,
            msg="Forbidden",
            hdrs={},  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error": "Forbidden quota exceeded"}'),
        )
        mock_urlopen.side_effect = err

        with self.assertRaises(GnaniAPIError) as ctx:
            self.adapter.synthesize("Breathe.")

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertNotIn(self.api_key, str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_timeout_handling(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = TimeoutError("TTS timeout")
        with self.assertRaises(GnaniTimeoutError):
            self.adapter.synthesize("Hold your breath.")


if __name__ == "__main__":
    unittest.main()
