"""LLM client for translation service."""

import os
from dataclasses import dataclass
from typing import Optional

import httpx
from loguru import logger


@dataclass
class LLMResponse:
    """Response from LLM API."""
    content: str
    model: str
    usage: Optional[dict] = None


class LLMClient:
    """OpenAI-compatible LLM client for translation.

    Supports Anthropic Claude and other OpenAI-compatible APIs.
    """

    # Default headers for Anthropic Claude
    ANTHROPIC_DEFAULT_HEADERS = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    def __init__(
        self,
        base_url: str = "https://api.anthropic.com",
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        """Initialize LLM client.

        Args:
            base_url: API base URL (e.g., https://api.anthropic.com)
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=60.0)
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def _is_anthropic(self) -> bool:
        """Check if the API is Anthropic."""
        return "anthropic.com" in self.base_url

    def _build_anthropic_request(self, prompt: str) -> dict:
        """Build request body for Anthropic API."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

    def _build_openai_request(self, prompt: str) -> dict:
        """Build request body for OpenAI-compatible API."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }

    def translate(self, prompt: str) -> LLMResponse:
        """Send translation prompt to LLM and get the response.

        Args:
            prompt: The translation prompt

        Returns:
            LLMResponse object containing the translation

        Raises:
            httpx.HTTPStatusError: If the API request fails
        """
        client = self._get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        if self._is_anthropic():
            headers.update(self.ANTHROPIC_DEFAULT_HEADERS)

        if self._is_anthropic():
            request_body = self._build_anthropic_request(prompt)
            url = f"{self.base_url}/v1/messages"
        else:
            request_body = self._build_openai_request(prompt)
            url = f"{self.base_url}/v1/chat/completions"

        logger.debug(f"Sending request to {url} with model {self.model}")

        response = client.post(url, json=request_body, headers=headers)
        response.raise_for_status()

        data = response.json()

        # Parse response based on API type
        if self._is_anthropic():
            content = data["content"][0]["text"]
        else:
            content = data["choices"][0]["message"]["content"]

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            usage=data.get("usage"),
        )

    def __enter__(self) -> "LLMClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()