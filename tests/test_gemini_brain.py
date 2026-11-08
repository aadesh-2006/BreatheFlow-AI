"""Mocked offline tests for Gemini API conversational brain adapter."""

import os
import unittest
from unittest.mock import MagicMock, patch

from breatheflow.providers.exceptions import (
    GeminiAuthError,
    GeminiConnectionError,
    GeminiRateLimitError,
    GeminiResponseError,
)
from breatheflow.providers.gemini_brain import GeminiBrainAdapter


class TestGeminiBrainAdapter(unittest.TestCase):
    """Test Gemini conversational reasoning and protocol extraction."""

    def setUp(self) -> None:
        self.mock_client = MagicMock()
        self.adapter = GeminiBrainAdapter(
            api_key="fake-test-key",
            model="gemini-3.5-flash-lite",
            client=self.mock_client,
        )

    def test_successful_conversational_response_without_protocol(self) -> None:
        """Verify conversational response without breathing protocol recommendation."""
        mock_response = MagicMock()
        mock_response.text = (
            '{"spoken_response": "You are doing well. Take a slow, smooth breath.", "action": null, "protocol": null}'
        )
        self.mock_client.models.generate_content.return_value = mock_response

        turn = self.adapter.generate_turn("How am I doing?")

        self.assertEqual(
            turn.spoken_response, "You are doing well. Take a slow, smooth breath."
        )
        self.assertIsNone(turn.suggested_protocol)
        self.assertEqual(turn.raw_response, mock_response.text)

    def test_valid_simplified_protocol_recommendation(self) -> None:
        """Verify simplified protocol recommendation is resolved to standard presets."""
        mock_response = MagicMock()
        mock_response.text = (
            '{"spoken_response": "I am here with you. Let\'s do Box Breathing to steady your focus.", '
            '"action": "start_session", "protocol": "box_breathing"}'
        )
        self.mock_client.models.generate_content.return_value = mock_response

        turn = self.adapter.generate_turn("I feel tense before my exam.")

        self.assertEqual(
            turn.spoken_response,
            "I am here with you. Let's do Box Breathing to steady your focus.",
        )
        self.assertIsNotNone(turn.suggested_protocol)
        if turn.suggested_protocol:
            self.assertEqual(turn.suggested_protocol["action"], "start_session")
            self.assertEqual(turn.suggested_protocol["protocol"], "box_breathing")
            self.assertEqual(turn.suggested_protocol["inhale_s"], 4.0)
            self.assertEqual(turn.suggested_protocol["hold_in_s"], 4.0)
            self.assertEqual(turn.suggested_protocol["exhale_s"], 4.0)
            self.assertEqual(turn.suggested_protocol["hold_out_s"], 4.0)
            self.assertEqual(turn.suggested_protocol["cycles"], 4)

    def test_invalid_or_unknown_protocol_rejection(self) -> None:
        """Verify unrecognized protocol name is strictly rejected as None."""
        mock_response = MagicMock()
        mock_response.text = (
            '{"spoken_response": "Let us try this.", "action": "start_session", "protocol": "invalid_hyperventilation"}'
        )
        self.mock_client.models.generate_content.return_value = mock_response

        turn = self.adapter.generate_turn("Help me breathe fast.")

        self.assertEqual(turn.spoken_response, "Let us try this.")
        self.assertIsNone(turn.suggested_protocol)

    def test_empty_model_output_raises_gemini_response_error(self) -> None:
        """Verify empty or whitespace-only model response raises GeminiResponseError."""
        for empty_val in ["", "   ", "\n\t  "]:
            mock_response = MagicMock()
            mock_response.text = empty_val
            self.mock_client.models.generate_content.return_value = mock_response

            with self.assertRaises(GeminiResponseError):
                self.adapter.generate_turn("Hello")

    def test_missing_api_key_raises_auth_error(self) -> None:
        """Verify instantiation/execution without API key raises GeminiAuthError."""
        with patch.dict(os.environ, {}, clear=True):
            adapter_no_key = GeminiBrainAdapter(api_key=None, client=None)
            with self.assertRaises(GeminiAuthError):
                adapter_no_key.generate_turn("Hello")

    def test_rate_limit_error_handling(self) -> None:
        """Verify 429 / quota exceeded error raises GeminiRateLimitError."""
        self.mock_client.models.generate_content.side_effect = Exception(
            "429 RESOURCE_EXHAUSTED: Quota exceeded for quota metric 'Generate Content API requests'"
        )

        with self.assertRaises(GeminiRateLimitError):
            self.adapter.generate_turn("Hello")

    def test_network_connection_error_handling(self) -> None:
        """Verify network timeout or connection failure raises GeminiConnectionError."""
        self.mock_client.models.generate_content.side_effect = Exception(
            "Connection reset by peer: timed out while connecting"
        )

        with self.assertRaises(GeminiConnectionError):
            self.adapter.generate_turn("Hello")

    def test_api_key_not_exposed_in_repr_or_logs(self) -> None:
        """Verify secret API key is never exposed in adapter string representation."""
        secret_key = "AIzaSySecretApiKey1234567890"
        adapter_with_secret = GeminiBrainAdapter(api_key=secret_key)
        repr_str = repr(adapter_with_secret)

        self.assertNotIn(secret_key, repr_str)
        self.assertIn("api_key='<SET>'", repr_str)
        self.assertIn("model='gemini-3.5-flash-lite'", repr_str)

        adapter_without_secret = GeminiBrainAdapter(api_key=None)
        self.assertIn("api_key='<NOT SET>'", repr(adapter_without_secret))

    def test_context_passed_compactly_to_content(self) -> None:
        """Verify session context is formatted compactly and passed into model call."""
        mock_response = MagicMock()
        mock_response.text = (
            '{"spoken_response": "Steady your breath.", "action": null, "protocol": null}'
        )
        self.mock_client.models.generate_content.return_value = mock_response

        context = {
            "controller_phase": "inhale",
            "is_active": True,
            "active_protocol": "box_breathing",
            "current_cycle": 2,
            "total_cycles": 4,
        }
        self.adapter.generate_turn("How am I doing?", session_context=context)

        # Inspect call args
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        contents_sent = call_kwargs["contents"]

        self.assertIn("[Context: controller_phase: inhale, is_active: true", contents_sent)
        self.assertIn("protocol: box_breathing", contents_sent)
        self.assertIn("cycle: 2/4", contents_sent)
        self.assertIn("How am I doing?", contents_sent)

    def test_markdown_code_fence_json_fallback(self) -> None:
        """Verify adapter cleanly parses JSON inside markdown code blocks."""
        mock_response = MagicMock()
        mock_response.text = (
            "Here is your breathing guidance:\n\n"
            "```json\n"
            '{\n  "spoken_response": "Let\'s do 4-7-8 breathing to unwind.",\n'
            '  "action": "start_session",\n'
            '  "protocol": "four_seven_eight"\n}\n'
            "```"
        )
        self.mock_client.models.generate_content.return_value = mock_response

        turn = self.adapter.generate_turn("I cannot sleep.")
        self.assertEqual(turn.spoken_response, "Let's do 4-7-8 breathing to unwind.")
        self.assertIsNotNone(turn.suggested_protocol)
        if turn.suggested_protocol:
            self.assertEqual(turn.suggested_protocol["protocol"], "four_seven_eight")
            self.assertEqual(turn.suggested_protocol["inhale_s"], 4.0)
            self.assertEqual(turn.suggested_protocol["hold_in_s"], 7.0)
            self.assertEqual(turn.suggested_protocol["exhale_s"], 8.0)
            self.assertEqual(turn.suggested_protocol["cycles"], 4)


if __name__ == "__main__":
    unittest.main()
