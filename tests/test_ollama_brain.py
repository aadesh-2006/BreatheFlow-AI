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

    def test_default_token_limit_and_timeout(self) -> None:
        """Verify default token limit is 512 and timeout is 60 seconds."""
        adapter = OllamaBrainAdapter()
        self.assertEqual(adapter.num_predict, 512)
        self.assertEqual(adapter.timeout, 60.0)

    def test_override_num_predict_and_timeout(self) -> None:
        """Verify caller can override num_predict and timeout."""
        adapter = OllamaBrainAdapter(num_predict=256, timeout=45.0)
        self.assertEqual(adapter.num_predict, 256)
        self.assertEqual(adapter.timeout, 45.0)

    @patch("urllib.request.urlopen")
    def test_configured_num_predict_sent_in_request(self, mock_urlopen: MagicMock) -> None:
        """Verify the configured num_predict and timeout are sent in the HTTP request."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_payload = {
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": "Breathe gently."},
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Test with default adapter (512 tokens, 60s timeout)
        self.adapter.generate_turn("I need help.")
        called_req = mock_urlopen.call_args[0][0]
        sent_body = json.loads(called_req.data.decode("utf-8"))
        self.assertEqual(sent_body["options"]["num_predict"], 512)
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 60.0)

        # Test with customized adapter (1024 tokens, 90s timeout)
        custom_adapter = OllamaBrainAdapter(num_predict=1024, timeout=90.0)
        custom_adapter.generate_turn("I need more help.")
        custom_req = mock_urlopen.call_args[0][0]
        custom_sent_body = json.loads(custom_req.data.decode("utf-8"))
        self.assertEqual(custom_sent_body["options"]["num_predict"], 1024)
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 90.0)

    @patch("urllib.request.urlopen")
    def test_idle_session_context_formatting(self, mock_urlopen: MagicMock) -> None:
        """Verify idle session context is formatted concisely without raw JSON metadata."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_payload = {
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": "Hello."},
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        idle_context = {
            "controller_phase": "idle",
            "is_active": False,
            "current_cycle": 0,
            "total_cycles": 0,
            "active_protocol": None,
        }
        self.adapter.generate_turn("I feel anxious before my exam.", session_context=idle_context)

        called_req = mock_urlopen.call_args[0][0]
        sent_body = json.loads(called_req.data.decode("utf-8"))
        user_msg = sent_body["messages"][1]["content"]

        self.assertIn("[Context: controller_phase: idle, is_active: false]\n", user_msg)
        self.assertNotIn("null", user_msg)
        self.assertNotIn("{", user_msg)
        self.assertNotIn("}", user_msg)
        self.assertNotIn("total_cycles", user_msg)

    @patch("urllib.request.urlopen")
    def test_active_session_context_formatting(self, mock_urlopen: MagicMock) -> None:
        """Verify active session context preserves controller phase, protocol, and cycle."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_payload = {
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": "Keep breathing smoothly."},
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        active_context = {
            "controller_phase": "inhale",
            "is_active": True,
            "current_cycle": 2,
            "total_cycles": 4,
            "active_protocol": "box_breathing",
        }
        self.adapter.generate_turn("Can I slow down?", session_context=active_context)

        called_req = mock_urlopen.call_args[0][0]
        sent_body = json.loads(called_req.data.decode("utf-8"))
        user_msg = sent_body["messages"][1]["content"]

        self.assertIn(
            "[Context: controller_phase: inhale, is_active: true, protocol: box_breathing, cycle: 2/4]\n",
            user_msg,
        )
        self.assertIn("controller_phase: inhale", user_msg)
        self.assertIn("protocol: box_breathing", user_msg)
        self.assertIn("cycle: 2/4", user_msg)

    @patch("urllib.request.urlopen")
    def test_empty_or_none_context_omits_prefix(self, mock_urlopen: MagicMock) -> None:
        """Verify empty or None context does not prepend any [Context: ...] tag."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_payload = {
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": "Take a breath."},
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # None context
        self.adapter.generate_turn("Hello", session_context=None)
        req_none = mock_urlopen.call_args[0][0]
        body_none = json.loads(req_none.data.decode("utf-8"))
        self.assertEqual(body_none["messages"][1]["content"], "Hello")

        # Empty dict context
        self.adapter.generate_turn("Hello again", session_context={})
        req_empty = mock_urlopen.call_args[0][0]
        body_empty = json.loads(req_empty.data.decode("utf-8"))
        self.assertEqual(body_empty["messages"][1]["content"], "Hello again")

    def test_context_serialization_compactness(self) -> None:
        """Verify formatted context is significantly more compact than raw JSON."""
        idle_context = {
            "controller_phase": "idle",
            "is_active": False,
            "current_cycle": 0,
            "total_cycles": 0,
            "active_protocol": None,
        }
        raw_json = json.dumps(idle_context)
        formatted = self.adapter._format_session_context(idle_context)

        self.assertEqual(formatted, "controller_phase: idle, is_active: false")
        # Formatted string should be less than half the size of raw JSON
        self.assertLess(len(formatted), len(raw_json) * 0.5)

    @patch("urllib.request.urlopen")
    def test_simplified_protocol_json_extraction(self, mock_urlopen: MagicMock) -> None:
        """Verify simplified protocol without timings is extracted and resolved to presets."""
        mock_response = MagicMock()
        mock_response.status = 200
        llm_content = (
            "Let's do Box Breathing to center your mind.\n\n"
            "```json\n"
            "{\n"
            '  "action": "start_session",\n'
            '  "protocol": "box_breathing"\n'
            "}\n"
            "```"
        )
        mock_payload = {
            "model": "qwen3:4b",
            "message": {"role": "assistant", "content": llm_content},
        }
        mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        turn = self.adapter.generate_turn("I am nervous before an exam.")
        self.assertEqual(turn.spoken_response, "Let's do Box Breathing to center your mind.")
        self.assertIsNotNone(turn.suggested_protocol)
        if turn.suggested_protocol:
            self.assertEqual(turn.suggested_protocol["action"], "start_session")
            self.assertEqual(turn.suggested_protocol["protocol"], "box_breathing")
            self.assertEqual(turn.suggested_protocol["inhale_s"], 4.0)
            self.assertEqual(turn.suggested_protocol["hold_in_s"], 4.0)
            self.assertEqual(turn.suggested_protocol["exhale_s"], 4.0)
            self.assertEqual(turn.suggested_protocol["hold_out_s"], 4.0)
            self.assertEqual(turn.suggested_protocol["cycles"], 4)

    def test_simplified_protocol_resolution_presets(self) -> None:
        """Verify all standard protocols resolve accurately from simplified recommendations."""
        presets = [
            ("box_breathing", 4.0, 4.0, 4.0, 4.0, 4),
            ("four_seven_eight", 4.0, 7.0, 8.0, 0.0, 4),
            ("coherent_breathing", 5.5, 0.0, 5.5, 0.0, 5),
            ("physiological_sigh", 2.5, 1.0, 6.0, 0.0, 3),
        ]
        for name, inhale, hold_in, exhale, hold_out, cycles in presets:
            payload = {"action": "start_session", "protocol": name}
            resolved = self.adapter._validate_protocol_payload(payload)
            self.assertIsNotNone(resolved, f"Failed for {name}")
            if resolved:
                self.assertEqual(resolved["inhale_s"], inhale)
                self.assertEqual(resolved["hold_in_s"], hold_in)
                self.assertEqual(resolved["exhale_s"], exhale)
                self.assertEqual(resolved["hold_out_s"], hold_out)
                self.assertEqual(resolved["cycles"], cycles)

    def test_full_payload_backward_compatibility(self) -> None:
        """Verify explicit custom durations and cycle counts remain supported."""
        payload = {
            "action": "start_session",
            "protocol": "custom_technique",
            "inhale_s": 3.0,
            "hold_in_s": 1.5,
            "exhale_s": 4.5,
            "hold_out_s": 0.5,
            "cycles": 6,
        }
        resolved = self.adapter._validate_protocol_payload(payload)
        self.assertIsNotNone(resolved)
        if resolved:
            self.assertEqual(resolved["protocol"], "custom_technique")
            self.assertEqual(resolved["inhale_s"], 3.0)
            self.assertEqual(resolved["hold_in_s"], 1.5)
            self.assertEqual(resolved["cycles"], 6)

    def test_unknown_protocol_rejection(self) -> None:
        """Verify unrecognized protocol names are rejected as None."""
        payload = {"action": "start_session", "protocol": "unknown_dangerous_breath"}
        self.assertIsNone(self.adapter._validate_protocol_payload(payload))

    @patch("urllib.request.urlopen")
    def test_empty_or_whitespace_content_raises_response_error(self, mock_urlopen: MagicMock) -> None:
        """Verify empty or whitespace-only LLM output raises OllamaResponseError."""
        for empty_text in ["", "   ", "\n\t  \n"]:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_payload = {
                "model": "qwen3:4b",
                "message": {"role": "assistant", "content": empty_text},
            }
            mock_response.read.return_value = json.dumps(mock_payload).encode("utf-8")
            mock_urlopen.return_value.__enter__.return_value = mock_response

            with self.assertRaises(OllamaResponseError):
                self.adapter.generate_turn("Hello")


if __name__ == "__main__":
    unittest.main()
