"""Exceptions raised by BreatheFlow provider adapters."""


class ProviderError(Exception):
    """Base exception for external provider adapters."""

    pass


class GnaniAPIError(ProviderError):
    """Raised when a Gnani API request fails or returns an error response."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class GnaniTimeoutError(ProviderError):
    """Raised when a request to Gnani API exceeds timeout threshold."""

    pass


class OllamaError(ProviderError):
    """Base exception for Ollama conversational brain errors."""

    pass


class OllamaConnectionError(OllamaError):
    """Raised when unable to reach the local Ollama service."""

    pass


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an invalid or unexpected response."""

    pass
