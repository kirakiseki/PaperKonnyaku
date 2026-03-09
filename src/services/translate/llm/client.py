"""LLM client for translation service."""

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

import httpx
from loguru import logger

from services.translate.llm.rate_limiter import RateLimiter


@dataclass
class LLMResponse:
    """Response from LLM API."""

    content: str
    model: str
    usage: Optional[dict] = None


class LLMClient:
    """OpenAI-compatible LLM client for translation.

    Supports Anthropic Claude and other OpenAI-compatible APIs.
    Includes built-in rate limiting with token bucket algorithm.
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
        rpm: int = 60,
        tpm: int = 100000,
        max_concurrent: int = 10,
    ):
        """Initialize LLM client.

        Args:
            base_url: API base URL (e.g., https://api.anthropic.com)
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            rpm: Requests per minute limit (0 to disable)
            tpm: Tokens per minute limit (0 to disable)
            max_concurrent: Maximum concurrent requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.max_concurrent = max_concurrent

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(rpm=rpm, tpm=tpm)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for concurrency control."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    async def aclose(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
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

    async def _do_request(
        self, client: httpx.AsyncClient, url: str, request_body: dict, headers: dict
    ) -> tuple[dict, httpx.Response]:
        """Execute a single HTTP request.

        Args:
            client: HTTP client
            url: API URL
            request_body: Request body
            headers: Request headers

        Returns:
            Tuple of (response_data, raw_response)
        """
        response = await client.post(url, json=request_body, headers=headers)
        data = response.json()
        return data, response

    async def chat(
        self,
        prompt: str,
        max_retries: int = 3,
    ) -> LLMResponse:
        """Send a chat prompt to LLM with rate limiting and retry logic.

        This method includes:
        - Rate limiting (RPM/TPM) via token bucket algorithm
        - Concurrency control via semaphore
        - Exponential backoff retry on rate limit errors (429)
        - Real-time adjustment based on actual token usage

        Args:
            prompt: The chat prompt
            max_retries: Maximum retry attempts on rate limit errors

        Returns:
            LLMResponse object containing the response

        Raises:
            httpx.HTTPStatusError: If the API request fails (after retries)
            RateLimitError: If max retries exceeded
        """
        # Get concurrency control semaphore
        semaphore = await self._get_semaphore()

        async with semaphore:
            # Estimate tokens and acquire rate limit permission
            estimated_tokens = self.rate_limiter.estimate_request_tokens(
                prompt, self.max_tokens
            )
            await self.rate_limiter.acquire(estimated_tokens)

            # Retry loop with exponential backoff
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    client = await self._get_client()
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

                    logger.debug(
                        f"Sending request to {url} with model {self.model} "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )

                    data, response = await self._do_request(
                        client, url, request_body, headers
                    )

                    # Check for HTTP errors
                    response.raise_for_status()

                    # Parse response based on API type
                    if self._is_anthropic():
                        content = data["content"][0]["text"]
                    else:
                        content = data["choices"][0]["message"]["content"]

                    # Get usage info for real-time adjustment
                    usage = data.get("usage")
                    if usage:
                        self.rate_limiter.update_from_usage(usage)

                    return LLMResponse(
                        content=content,
                        model=data.get("model", self.model),
                        usage=usage,
                    )

                except httpx.HTTPStatusError as e:
                    last_exception = e

                    # Handle rate limit (429) specifically
                    if e.response.status_code == 429:
                        retry_after: Optional[float] = None

                        # Try to get Retry-After header
                        retry_after_str = e.response.headers.get("Retry-After")
                        if retry_after_str:
                            try:
                                retry_after = float(retry_after_str)
                            except ValueError:
                                pass

                        if attempt < max_retries:
                            # Calculate delay with exponential backoff
                            delay = await self.rate_limiter.handle_rate_limit_error(
                                attempt, retry_after
                            )
                            logger.warning(
                                f"Rate limited (429), retrying in {delay:.2f}s"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.error(
                                f"Rate limit exceeded after {max_retries} retries"
                            )
                            raise RateLimitError(
                                f"Rate limit exceeded after {max_retries} retries"
                            ) from e

                    # For other HTTP errors, don't retry
                    raise

                except httpx.TimeoutException as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = self.rate_limiter.get_retry_delay(attempt)
                        logger.warning(
                            f"Request timeout, retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise

                except httpx.ConnectError as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = self.rate_limiter.get_retry_delay(attempt)
                        logger.warning(
                            f"Connection error, retrying in {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise

            # If we get here, all retries failed
            raise last_exception or RuntimeError("Unknown error in chat")

    @property
    def stats(self) -> dict:
        """Get rate limiter statistics."""
        return self.rate_limiter.stats

    async def __aenter__(self) -> "LLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.aclose()


class RateLimitError(Exception):
    """Raised when rate limit is exceeded after max retries."""
    pass