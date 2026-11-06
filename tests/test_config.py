"""Unit tests for BreatheFlow configuration."""

import unittest
from breatheflow.config import BreatheFlowConfig, load_config


class TestBreatheFlowConfig(unittest.TestCase):
    """Test configuration loading, validation, and secret masking."""

    def test_default_config(self) -> None:
        config = load_config(env={})
        self.assertIsNone(config.gnani_api_key)
        self.assertEqual(config.gnani_base_url, "https://api.vachana.ai")
        self.assertEqual(config.stt_model, "prisma-v2.5")
        self.assertEqual(config.stt_language_code, "en-IN")
        self.assertEqual(config.tts_model, "timbre-v2.5")
        self.assertEqual(config.tts_language_code, "en-IN")
        self.assertEqual(config.tts_voice, "Kaveri")
        self.assertEqual(config.tts_speed, 0.95)
        self.assertEqual(config.ollama_host, "http://localhost:11434")
        self.assertEqual(config.ollama_model, "qwen3:4b")
        self.assertEqual(config.log_level, "INFO")

    def test_env_override(self) -> None:
        custom_env = {
            "GNANI_API_KEY": "gnani_super_secret_test_key_123",
            "GNANI_BASE_URL": "https://custom.vachana.ai",
            "GNANI_STT_MODEL": "prisma-v3.0",
            "GNANI_STT_LANGUAGE_CODE": "hi-IN",
            "GNANI_TTS_MODEL": "timbre-v3.0",
            "GNANI_TTS_VOICE": "Nalini",
            "GNANI_TTS_SPEED": "1.1",
            "OLLAMA_HOST": "http://192.168.1.100:11434",
            "OLLAMA_MODEL": "qwen3:4b",
            "LOG_LEVEL": "DEBUG",
        }
        config = load_config(env=custom_env)
        self.assertEqual(config.gnani_api_key, "gnani_super_secret_test_key_123")
        self.assertEqual(config.gnani_base_url, "https://custom.vachana.ai")
        self.assertEqual(config.stt_model, "prisma-v3.0")
        self.assertEqual(config.stt_language_code, "hi-IN")
        self.assertEqual(config.tts_voice, "Nalini")
        self.assertEqual(config.tts_speed, 1.1)
        self.assertEqual(config.ollama_host, "http://192.168.1.100:11434")
        self.assertEqual(config.log_level, "DEBUG")

    def test_secret_masking_in_repr(self) -> None:
        raw_secret = "gnani_real_secret_token_abcdef"
        config = load_config(env={"GNANI_API_KEY": raw_secret})
        repr_str = repr(config)

        # The actual key string must NEVER appear in repr
        self.assertNotIn(raw_secret, repr_str)
        self.assertIn("gnani_api_key='<SET>'", repr_str)

        empty_config = load_config(env={})
        self.assertIn("gnani_api_key='<NOT SET>'", repr(empty_config))


if __name__ == "__main__":
    unittest.main()
