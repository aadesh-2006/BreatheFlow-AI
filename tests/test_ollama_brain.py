"""Mocked HTTP tests for local Ollama / Qwen3-4B conversational brain adapter."""

import json
from unittest.mock import MagicMock, patch
import unittest

from breatheflow.providers.exceptions import OllamaConnectionError, OllamaResponseError
from breatheflow.providers.ollama_brain import OllamaBrainAdapter


class TestOllamaBrainAdapter(unittest.TestCase):
    """Test Ollama conversational reasoning and protocol extraction."""

    def setUp(self) -> None:
        self.adapter = OllamaBrainAdapter(ollama_host="http://localhost:11434", model="qwen3:4b")

    @patch("urllib.request.urlopen")
    def test_successful_turn_with_protocol(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        llm_content = (
            "I hear you. Let's start with Box Breathing to steady your focus. Get comfortable.\n\n"
            "```json\n"
            "{\n"
            '  "action": "start_session",\n'
            '  "protocol": "box_breathing",\n'
            '  "inhale_s": 4.0,\n'
            '  "hold_in_s": 4.0,\n'
            '  "exhale_s": 4.0,\n'
            '  "hold_out_s": 4.0,\n'
            '  "cycles": 4\n'
            "}\n"
            "```"
        )
        mock_payload = {
            "model": "qwen3:4b",
            "message": {
                "role": "assistant",
                "content": llm_content,
            },
            "done": True,
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        turn = self.adapter.generate_turn("I feel tense before my presentation.")

        # Spoken text should be clean, without the JSON code block
        self.assertEqual(
            turn.spoken_response,
            "I hear you. Let's start with Box Breathing to steady your focus. Get comfortable.",
        )
        self.assertIsNotNone(turn.suggested_protocol)
        if turn.suggested_protocol:
            self.assertEqual(turn.suggested_protocol["action"], "start_session")
            self.assertEqual(turn.suggested_protocol["protocol"], "box_breathing")
            self.assertEqual(turn.suggested_protocol["inhale_s"], 4.0)
            self.assertEqual(turn.suggested_protocol["cycles"], 4)

    @patch("urllib.request.urlopen")
    def test_conversational_turn_without_protocol(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        llm_content = "You are doing great. Keep that steady pace, breathing smoothly."
        mock_payload = {
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": llm_content},
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        turn = self.adapter.generate_turn("How am I doing?")
        self.assertEqual(turn.spoken_response, llm_content)
        self.assertIsNone(turn.suggested_protocol)

    @patch("urllib.request.urlopen")
    def test_rejection_of_invalid_protocol_parameters(self, mock_urlopen: MagicMock) -> None:
        """Verify that malformed or dangerous protocol parameters are discarded."""
        mock_response = MagicMock()
        mock_response.status = 200

        # LLM generated negative duration and zero cycles
        llm_content = (
            "Let's try this.\n\n"
            "```json\n"
            "{\n"
            '  "action": "start_session",\n'
            '  "protocol": "faulty_technique",\n'
            '  "inhale_s": -2.0,\n'
            '  "hold_in_s": 0.0,\n'
            '  "exhale_s": 4.0,\n'
            '  "hold_out_s": 0.0,\n'
            '  "cycles": 0\n'
            "}\n"
            "```"
        )
        mock_payload = {
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": llm_content},
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        turn = self.adapter.generate_turn("Help me breathe.")
        self.assertEqual(turn.spoken_response, "Let's try this.")
        # Must be rejected and discarded as None
        self.assertIsNone(turn.suggested_protocol)

    @patch("urllib.request.urlopen")
    def test_connection_error_raises_ollama_connection_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = ConnectionRefusedError("Connection refused to localhost:11434")

        with self.assertRaises(OllamaConnectionError):
            self.adapter.generate_turn("Hello")

    @patch("urllib.request.urlopen")
    def test_malformed_response_raises_ollama_response_error(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"unexpected": "format"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response

        with self.assertRaises(OllamaResponseError):
            self.adapter.generate_turn("Hello")

    @patch("urllib.request.urlopen")
    def test_context_passed_safely(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_payload = {
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": "I notice your steady breathing."},
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        context = {"session_state": "in_progress", "technique": "box_breathing"}
        turn = self.adapter.generate_turn("I feel calmer now.", session_context=context)

        # Inspect request
        called_req = mock_urlopen.call_args[0][0]
        sent_body = json.loads(called_req.data.decode("utf-8"))
        user_msg = sent_body["messages"][1]["content"]

        self.assertIn("session_state", user_msg)
        self.assertIn("box_breathing", user_msg)
        self.assertIn("I feel calmer now.", user_msg)


if __name__ == "__main__":
    unittest.main()
