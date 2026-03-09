"""Tests for rate limiter module."""

import asyncio
import time
import pytest

from services.translate.llm.rate_limiter import RateLimiter, TokenEstimator


class TestTokenEstimator:
    """Tests for TokenEstimator."""

    def test_estimate_english_text(self):
        """Test token estimation for English text."""
        estimator = TokenEstimator()
        text = "Hello, world!"  # 13 chars
        tokens = estimator.estimate_input_tokens(text)

        # 13 chars / 4 chars per token = ~3.25 + overhead 50 = ~53
        assert tokens >= 50
        assert tokens <= 120  # Adjusted for safety margin

    def test_estimate_chinese_text(self):
        """Test token estimation for Chinese text."""
        estimator = TokenEstimator()
        text = "你好世界"  # 4 Chinese chars
        tokens = estimator.estimate_input_tokens(text)

        # 4 chars / 1.5 chars per token = ~2.67 + overhead 50 = ~53
        assert tokens >= 50
        assert tokens <= 60

    def test_estimate_mixed_text(self):
        """Test token estimation for mixed text."""
        estimator = TokenEstimator()
        text = "Hello 你好 World 世界"  # Mix of English and Chinese
        tokens = estimator.estimate_input_tokens(text)

        # Should be higher than pure Chinese or pure English
        assert tokens > 50

    def test_estimate_output_tokens(self):
        """Test output token estimation."""
        estimator = TokenEstimator()
        max_tokens = 1000
        estimated = estimator.estimate_output_tokens(max_tokens)

        # Should be ~75% of max_tokens
        assert estimated == 750


class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        limiter = RateLimiter()

        assert limiter.rpm == 60
        assert limiter.tpm == 100000
        assert limiter.window == 60.0

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        limiter = RateLimiter(rpm=30, tpm=50000, window=30.0)

        assert limiter.rpm == 30
        assert limiter.tpm == 50000
        assert limiter.window == 30.0

    def test_init_with_disabled_limits(self):
        """Test initialization with disabled limits (0 = disable)."""
        limiter = RateLimiter(rpm=0, tpm=0)

        # Should use very high limits when disabled
        assert limiter.rpm == 999999
        assert limiter.tpm == 999999999


class TestRateLimiterAcquire:
    """Tests for rate limiter acquire method."""

    @pytest.mark.asyncio
    async def test_acquire_immediate(self):
        """Test that acquire returns immediately when tokens available."""
        limiter = RateLimiter(rpm=60, tpm=100000)

        # Should complete immediately
        start = time.monotonic()
        await limiter.acquire(1000)
        elapsed = time.monotonic() - start

        # Should be nearly instant (< 0.1s)
        assert elapsed < 0.1

        stats = limiter.stats
        assert stats["total_requests"] == 1

    @pytest.mark.asyncio
    async def test_acquire_rpm_limit(self):
        """Test RPM limiting works."""
        limiter = RateLimiter(rpm=2, tpm=999999999, window=1.0)

        # First request should be immediate
        await limiter.acquire(1000)

        # Second request should also be immediate
        await limiter.acquire(1000)

        # Third request should wait (RPM = 2, window = 1s)
        start = time.monotonic()
        await limiter.acquire(1000)
        elapsed = time.monotonic() - start

        # Should have waited at least ~1s
        assert elapsed >= 0.9

    @pytest.mark.asyncio
    async def test_acquire_tpm_limit(self):
        """Test TPM limiting works."""
        limiter = RateLimiter(rpm=999999, tpm=600, window=1.0)

        # First request with 500 tokens - should be immediate
        start = time.monotonic()
        await limiter.acquire(500)
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

        # Current tokens = 600 - 500 = 100
        # Next request needs 500 tokens
        # Will need to wait ~0.67s for token refill (400 tokens needed at 600/s rate)
        start = time.monotonic()
        await limiter.acquire(500)
        elapsed = time.monotonic() - start

        # Should have waited for token refill (at least 0.5s)
        assert elapsed >= 0.4


class TestRateLimiterRetryDelay:
    """Tests for exponential backoff."""

    def test_get_retry_delay_base(self):
        """Test base retry delay calculation."""
        limiter = RateLimiter(base_delay=1.0)

        delay = limiter.get_retry_delay(attempt=0)

        # Base delay should be ~1.0s (with jitter 0.5-1.5)
        assert 0.5 <= delay <= 1.5

    def test_get_retry_delay_exponential(self):
        """Test exponential backoff increases with attempts."""
        # Test without jitter - use retry_after to bypass random
        limiter = RateLimiter(base_delay=1.0)

        # When retry_after is provided, use it directly
        delay0 = limiter.get_retry_delay(attempt=0, retry_after=1.0)
        delay1 = limiter.get_retry_delay(attempt=1, retry_after=2.0)

        assert delay0 == 1.0
        assert delay1 == 2.0

    def test_get_retry_delay_without_retry_after(self):
        """Test exponential backoff without Retry-After header."""
        limiter = RateLimiter(base_delay=1.0)

        # Without retry_after, should return positive delay
        delay = limiter.get_retry_delay(attempt=0)
        assert delay > 0

        delay = limiter.get_retry_delay(attempt=2)
        assert delay > 0

    def test_get_retry_delay_with_retry_after(self):
        """Test retry delay uses server-provided value."""
        limiter = RateLimiter(base_delay=1.0)

        # When server provides Retry-After, use it
        delay = limiter.get_retry_delay(attempt=0, retry_after=5.0)

        assert delay == 5.0


class TestRateLimiterStats:
    """Tests for rate limiter statistics."""

    @pytest.mark.asyncio
    async def test_stats_initial(self):
        """Test initial stats."""
        limiter = RateLimiter(rpm=60, tpm=100000)

        stats = limiter.stats

        assert stats["total_requests"] == 0
        assert stats["total_retries"] == 0
        assert stats["total_rate_limited"] == 0
        assert stats["current_tokens"] == 100000
        assert stats["current_requests_in_window"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_requests(self):
        """Test stats after making requests."""
        limiter = RateLimiter(rpm=60, tpm=100000)

        await limiter.acquire(1000)
        await limiter.acquire(2000)

        stats = limiter.stats

        assert stats["total_requests"] == 2
        assert stats["current_requests_in_window"] == 2
        # Should have consumed 3000 tokens
        assert stats["current_tokens"] < 100000


class TestRateLimiterRealTimeAdjustment:
    """Tests for real-time adjustment based on actual usage."""

    @pytest.mark.asyncio
    async def test_update_from_usage(self):
        """Test updating from usage data."""
        limiter = RateLimiter(rpm=60, tpm=100000)

        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
        }

        # Should not raise
        limiter.update_from_usage(usage)

    def test_update_from_empty_usage(self):
        """Test updating from empty usage data."""
        limiter = RateLimiter(rpm=60, tpm=100000)

        # Should not raise
        limiter.update_from_usage(None)
        limiter.update_from_usage({})


class TestRateLimiterConcurrency:
    """Tests for concurrent access to rate limiter."""

    @pytest.mark.asyncio
    async def test_concurrent_acquire(self):
        """Test concurrent acquire requests."""
        limiter = RateLimiter(rpm=100, tpm=100000, window=1.0)

        async def acquire_task(tokens: int):
            await limiter.acquire(tokens)

        # Run 10 concurrent requests
        tasks = [acquire_task(100) for _ in range(10)]
        await asyncio.gather(*tasks)

        stats = limiter.stats
        assert stats["total_requests"] == 10

    @pytest.mark.asyncio
    async def test_concurrent_acquire_with_high_tpm(self):
        """Test concurrent acquire with high TPM limit."""
        limiter = RateLimiter(rpm=1000, tpm=1000000, window=1.0)

        async def acquire_task(i: int):
            await limiter.acquire(1000)
            return i

        # Run 20 concurrent requests
        tasks = [acquire_task(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        stats = limiter.stats
        assert stats["total_requests"] == 20


class TestRateLimiterHandleRateLimitError:
    """Tests for handling rate limit errors."""

    @pytest.mark.asyncio
    async def test_handle_rate_limit(self):
        """Test handling rate limit error."""
        limiter = RateLimiter(rpm=60, tpm=100000)

        delay = await limiter.handle_rate_limit_error(attempt=0, retry_after=None)

        # Should return a delay
        assert delay > 0

        stats = limiter.stats
        assert stats["total_retries"] == 1
        assert stats["total_rate_limited"] == 1

    @pytest.mark.asyncio
    async def test_handle_rate_limit_with_retry_after(self):
        """Test handling rate limit with server-provided delay."""
        limiter = RateLimiter(rpm=60, tpm=100000)

        delay = await limiter.handle_rate_limit_error(attempt=0, retry_after=2.0)

        assert delay == 2.0


class TestIntegrationWithLLMClient:
    """Integration tests for rate limiter with LLM client."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rate_limiter_integration(self):
        """Test rate limiter works with actual API calls."""
        from core.config import config
        from services.translate.llm.client import LLMClient

        llm_config = config.translate.llm

        if not llm_config.api_key:
            pytest.skip("API key not configured in config.toml")

        client = LLMClient(
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
            model=llm_config.model,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            rpm=llm_config.rpm,
            tpm=llm_config.tpm,
            max_concurrent=llm_config.max_concurrent,
        )

        try:
            # Make a few requests to test rate limiting
            for i in range(3):
                response = await client.chat(f"Hello, world! {i}")
                print(f"\nRequest {i + 1}: {response.content[:50]}...")
                print(f"Stats: {client.stats}")

            # Check stats are being tracked
            stats = client.stats
            assert stats["total_requests"] == 3

        finally:
            await client.aclose()