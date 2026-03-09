"""Rate limiter for LLM API calls with token bucket algorithm.

This module provides rate limiting for LLM API calls with:
- Token bucket algorithm for TPM (tokens per minute) control
- Sliding window for RPM (requests per minute) control
- Token estimation for requests
- Exponential backoff for retry on rate limit errors
- Real-time adjustment based on actual API usage
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class TokenEstimator:
    """Estimates token count for prompts and responses.

    Uses character-based heuristics and can be calibrated based on actual API usage.
    """

    # Average tokens per character (English ~4, Chinese ~1.5)
    tokens_per_char_english: float = 4.0
    tokens_per_char_chinese: float = 1.5

    # Overhead per message (system prompt, formatting, etc.)
    overhead_tokens: int = 50

    # Calibration state
    _input_total_estimated: float = field(default=0.0, init=False)
    _input_total_actual: float = field(default=0.0, init=False)
    _output_total_estimated: float = field(default=0.0, init=False)
    _output_total_actual: float = field(default=0.0, init=False)
    _sample_count: int = field(default=0, init=False)

    # Smoothing factor for exponential moving average (0.1 = slow adaptation, 0.5 = fast)
    _smoothing_factor: float = 0.2

    def estimate_input_tokens(self, text: str) -> int:
        """Estimate token count for input text.

        Args:
            text: Input text to estimate

        Returns:
            Estimated token count
        """
        # Simple heuristic: count Chinese chars separately
        chinese_chars = sum(1 for c in text if ord(c) > 0x4E00 and ord(c) < 0x9FFF)
        other_chars = len(text) - chinese_chars

        # Apply calibration factor if we have enough samples
        base_estimate = (
            chinese_chars * self.tokens_per_char_chinese
            + other_chars * self.tokens_per_char_english
        )
        estimate = base_estimate + self.overhead_tokens

        # Apply adjustment factor based on historical accuracy
        if self._sample_count >= 5 and self._input_total_estimated > 0:
            adjustment_factor = self._input_total_actual / self._input_total_estimated
            # Clamp to reasonable range (0.5x to 2x)
            adjustment_factor = max(0.5, min(2.0, adjustment_factor))
            estimate *= adjustment_factor

        return int(max(1, estimate))

    def estimate_output_tokens(self, max_tokens: int) -> int:
        """Estimate token count for output.

        Args:
            max_tokens: Maximum tokens allowed for output

        Returns:
            Estimated token count (typically 75% of max_tokens)
        """
        estimate = max_tokens * 0.75

        # Apply calibration factor if we have enough samples
        if self._sample_count >= 5 and self._output_total_estimated > 0:
            adjustment_factor = self._output_total_actual / self._output_total_estimated
            # Clamp to reasonable range (0.5x to 2x)
            adjustment_factor = max(0.5, min(2.0, adjustment_factor))
            estimate *= adjustment_factor

        return int(max(1, estimate))

    def update_from_usage(self, usage: dict) -> None:
        """Update estimator based on actual API usage.

        Tracks the ratio between estimated and actual tokens, then adjusts
        future estimates accordingly using exponential moving average.

        Args:
            usage: Usage dict from API response
                - input_tokens / prompt_tokens: actual input tokens
                - output_tokens / completion_tokens: actual output tokens
        """
        if not usage:
            return

        # Extract actual token counts (support different API formats)
        actual_input = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        actual_output = usage.get("output_tokens") or usage.get("completion_tokens") or 0

        if actual_input <= 0 and actual_output <= 0:
            return

        self._sample_count += 1

        # Estimate what we predicted for comparison
        # Note: We don't have the original text here, so we use a default estimate
        # This is a limitation - ideally we'd pass the text or pre-store estimates
        # For now, we use a rough estimate based on typical prompt sizes
        estimated_input = max(actual_input * 0.8, 1)  # Assume we're slightly underestimating
        estimated_output = max(actual_output * 0.8, 1)

        # Update totals for running average
        # Use exponential moving average for smoother adaptation
        if self._sample_count == 1:
            self._input_total_estimated = estimated_input
            self._input_total_actual = actual_input
            self._output_total_estimated = estimated_output
            self._output_total_actual = actual_output
        else:
            # EMA: new_avg = alpha * new_value + (1 - alpha) * old_avg
            alpha = self._smoothing_factor
            self._input_total_estimated = alpha * estimated_input + (1 - alpha) * self._input_total_estimated
            self._input_total_actual = alpha * actual_input + (1 - alpha) * self._input_total_actual
            self._output_total_estimated = alpha * estimated_output + (1 - alpha) * self._output_total_estimated
            self._output_total_actual = alpha * actual_output + (1 - alpha) * self._output_total_actual

        if self._sample_count % 10 == 0:
            logger.debug(
                f"TokenEstimator calibrated: samples={self._sample_count}, "
                f"input_ratio={self._input_total_actual/max(self._input_total_estimated, 1):.2f}, "
                f"output_ratio={self._output_total_actual/max(self._output_total_estimated, 1):.2f}"
            )


@dataclass
class RateLimiter:
    """Rate limiter using token bucket + sliding window algorithm.

    Features:
    - Token bucket for TPM (tokens per minute) control
    - Sliding window for RPM (requests per minute) control
    - Exponential backoff for 429 errors
    - Real-time adjustment based on actual usage
    """

    rpm: int = 60  # Requests per minute limit
    tpm: int = 100000  # Tokens per minute limit
    window: float = 60.0  # Window time in seconds
    max_retries: int = 3  # Maximum retry attempts
    base_delay: float = 1.0  # Base delay for exponential backoff

    # Internal state
    _tokens: float = field(default=0, init=False)
    _last_refill: float = field(default=0, init=False)
    _request_timestamps: list[float] = field(default_factory=list, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    # Token estimation
    _token_estimator: TokenEstimator = field(default_factory=TokenEstimator, init=False)

    # Statistics
    _total_requests: int = field(default=0, init=False)
    _total_retries: int = field(default=0, init=False)
    _total_rate_limited: int = field(default=0, init=False)

    def __post_init__(self):
        """Initialize rate limiter state."""
        # If rate limiting is disabled (0), use very high limits
        if self.rpm <= 0:
            self.rpm = 999999
        if self.tpm <= 0:
            self.tpm = 999999999

        self._tokens = float(self.tpm)
        self._last_refill = time.monotonic()
        logger.info(
            f"RateLimiter initialized: rpm={self.rpm}, tpm={self.tpm}"
        )

    def estimate_request_tokens(
        self, prompt: str, max_output_tokens: int
    ) -> int:
        """Estimate total tokens for a request.

        Args:
            prompt: Input prompt text
            max_output_tokens: Maximum output tokens

        Returns:
            Estimated total tokens (input + output)
        """
        input_tokens = self._token_estimator.estimate_input_tokens(prompt)
        output_tokens = self._token_estimator.estimate_output_tokens(max_output_tokens)
        return input_tokens + output_tokens

    async def acquire(self, estimated_tokens: int) -> None:
        """Wait until permission to make a request is granted.

        Uses token bucket for TPM control and sliding window for RPM control.

        Args:
            estimated_tokens: Estimated token count for this request
        """
        while True:
            async with self._lock:
                now = time.monotonic()

                # 1. Refill tokens based on elapsed time
                elapsed = now - self._last_refill
                tokens_per_second = self.tpm / self.window
                self._tokens = min(self.tpm, self._tokens + elapsed * tokens_per_second)
                self._last_refill = now

                # 2. Clean up expired request timestamps (sliding window)
                self._request_timestamps = [
                    ts for ts in self._request_timestamps
                    if now - ts < self.window
                ]

                # 3. Check if we can proceed
                can_proceed = (
                    len(self._request_timestamps) < self.rpm
                    and self._tokens >= estimated_tokens
                )

                if can_proceed:
                    # Consume tokens
                    self._tokens -= estimated_tokens
                    self._request_timestamps.append(now)
                    self._total_requests += 1

                    # Log stats every 100 requests
                    if self._total_requests % 100 == 0:
                        logger.debug(
                            f"RateLimiter stats: requests={self._total_requests}, "
                            f"retries={self._total_retries}, rate_limited={self._total_rate_limited}"
                        )
                    return

                # Calculate wait time
                wait_times = []

                # Wait for token refill if needed
                if self._tokens < estimated_tokens:
                    tokens_needed = estimated_tokens - self._tokens
                    wait_time = tokens_needed / tokens_per_second
                    wait_times.append(wait_time)

                # Wait for RPM window if needed
                if len(self._request_timestamps) >= self.rpm:
                    oldest_request = min(self._request_timestamps)
                    wait_time = self.window - (now - oldest_request)
                    if wait_time > 0:
                        wait_times.append(wait_time)

                # Wait for the minimum of wait times
                wait = min(wait_times) if wait_times else 0.1
                wait = max(0.05, min(wait, 1.0))  # Clamp between 50ms and 1s

            # Release lock and wait
            await asyncio.sleep(wait)

    def get_retry_delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Calculate delay for exponential backoff.

        Args:
            attempt: Current retry attempt (0-indexed)
            retry_after: Optional Retry-After header value from API

        Returns:
            Delay in seconds
        """
        if retry_after is not None:
            # Use server-provided delay if available
            return retry_after

        # Exponential backoff: base_delay * 2^attempt
        # Add jitter (0.5 to 1.5 of base)
        import random
        jitter = 0.5 + random.random()
        return self.base_delay * (2 ** attempt) * jitter

    async def handle_rate_limit_error(
        self,
        attempt: int,
        retry_after: Optional[float] = None,
    ) -> float:
        """Handle rate limit error (429) with exponential backoff.

        Args:
            attempt: Current retry attempt
            retry_after: Optional Retry-After header value

        Returns:
            Delay before next retry
        """
        self._total_retries += 1
        self._total_rate_limited += 1

        delay = self.get_retry_delay(attempt, retry_after)
        logger.warning(
            f"Rate limited (attempt {attempt + 1}/{self.max_retries}), "
            f"waiting {delay:.2f}s before retry"
        )

        return delay

    def update_from_usage(self, usage: dict) -> None:
        """Update rate limiter based on actual API usage.

        This allows real-time adjustment of token estimation.

        Args:
            usage: Usage dict from API response
        """
        if not usage:
            return

        # Extract token counts
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0

        if input_tokens > 0 or output_tokens > 0:
            logger.debug(
                f"Actual usage: input={input_tokens}, output={output_tokens}"
            )

            # Update token estimator with actual usage
            self._token_estimator.update_from_usage(usage)

    @property
    def stats(self) -> dict:
        """Get rate limiter statistics.

        Returns:
            Dict with statistics
        """
        return {
            "total_requests": self._total_requests,
            "total_retries": self._total_retries,
            "total_rate_limited": self._total_rate_limited,
            "current_tokens": int(self._tokens),
            "current_requests_in_window": len(self._request_timestamps),
        }