"""Unit tests for the BreatheFlow CLI."""

import io
import os
import tempfile
import unittest
from unittest.mock import patch

from breatheflow.cli import main


class TestCLI(unittest.TestCase):
    """Test CLI execution paths, argument handling, and error conditions."""

    def test_mock_execution_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "response.wav")
            code = main(["--mock", "--output", out_path])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_missing_audio_file_returns_error(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            code = main(["--mock", "--audio", "nonexistent_file_path_123.wav"])
            self.assertEqual(code, 1)
            self.assertIn("Audio file not found", mock_stderr.getvalue())

    def test_missing_credentials_in_live_mode_returns_error(self) -> None:
        # Without --mock and with no GNANI_API_KEY in env
        with patch.dict(os.environ, {"GNANI_API_KEY": ""}, clear=True):
            with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
                code = main(["--audio", "dummy.wav"])
                self.assertEqual(code, 1)
                self.assertIn("GNANI_API_KEY is not set", mock_stderr.getvalue())

    def test_invalid_output_directory_returns_error(self) -> None:
        """Verify that attempting to write to a nonexistent nested directory returns error cleanly."""
        invalid_out = os.path.join("nonexistent_directory_xyz", "sub", "output.wav")
        # Ensure the directory definitely does not exist
        if os.path.exists("nonexistent_directory_xyz"):
            os.rmdir("nonexistent_directory_xyz")

        with patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:
            code = main(["--mock", "--output", invalid_out])
            self.assertEqual(code, 1)
            self.assertIn("Failed to write audio output file", mock_stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
