"""Tests for LLM client."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from services.translate.llm.client import LLMClient, LLMResponse


class TestLLMClientInit:
    """Tests for LLMClient initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        client = LLMClient()
        assert client.base_url == "https://api.anthropic.com"
        assert client.model == "claude-sonnet-4-20250514"
        assert client.max_tokens == 4096
        assert client.temperature == 0.0

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        client = LLMClient(
            base_url="https://api.openai.com",
            api_key="test-key",
            model="gpt-4o",
            max_tokens=2048,
            temperature=0.5,
        )
        assert client.base_url == "https://api.openai.com"
        assert client.api_key == "test-key"
        assert client.model == "gpt-4o"
        assert client.max_tokens == 2048
        assert client.temperature == 0.5

    def test_init_from_env_var(self):
        """Test initialization from environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
            client = LLMClient()
            assert client.api_key == "env-key"


class TestLLMClientIsAnthropic:
    """Tests for Anthropic API detection."""

    def test_is_anthropic_true(self):
        """Test that Anthropic URL is detected correctly."""
        client = LLMClient(base_url="https://api.anthropic.com")
        assert client._is_anthropic() is True

    def test_is_anthropic_false(self):
        """Test that non-Anthropic URL is detected correctly."""
        client = LLMClient(base_url="https://api.openai.com")
        assert client._is_anthropic() is False

    def test_is_anthropic_with_path(self):
        """Test detection with URL path."""
        client = LLMClient(base_url="https://api.anthropic.com/v1/")
        assert client._is_anthropic() is True


class TestLLMClientBuildRequest:
    """Tests for request building."""

    def test_build_anthropic_request(self):
        """Test building Anthropic API request."""
        client = LLMClient(model="claude-sonnet-4-20250514", max_tokens=1000, temperature=0.3)
        request = client._build_anthropic_request("Translate this")

        assert request["model"] == "claude-sonnet-4-20250514"
        assert request["max_tokens"] == 1000
        assert request["temperature"] == 0.3
        assert len(request["messages"]) == 1
        assert request["messages"][0]["role"] == "user"
        assert request["messages"][0]["content"] == "Translate this"

    def test_build_openai_request(self):
        """Test building OpenAI API request."""
        client = LLMClient(model="gpt-4o", max_tokens=2000, temperature=0.7)
        request = client._build_openai_request("Translate this")

        assert request["model"] == "gpt-4o"
        assert request["max_tokens"] == 2000
        assert request["temperature"] == 0.7
        assert len(request["messages"]) == 1
        assert request["messages"][0]["role"] == "user"
        assert request["messages"][0]["content"] == "Translate this"


class TestLLMClientChat:
    """Tests for chat method (renamed from translate)."""

    @pytest.mark.asyncio
    @patch("services.translate.llm.client.httpx.AsyncClient")
    async def test_chat_anthropic_success(self, mock_client_class):
        """Test successful chat with Anthropic API."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "content": [{"text": "Translated text"}],
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = LLMClient(api_key="test-key", rpm=0, tpm=0)  # Disable rate limiting for test
        result = await client.chat("Translate this")

        assert isinstance(result, LLMResponse)
        assert result.content == "Translated text"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.usage == {"input_tokens": 100, "output_tokens": 50}

    @pytest.mark.asyncio
    @patch("services.translate.llm.client.httpx.AsyncClient")
    async def test_chat_openai_success(self, mock_client_class):
        """Test successful chat with OpenAI-compatible API."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "翻译文本"}}],
            "model": "gpt-4o",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = LLMClient(base_url="https://api.openai.com", api_key="test-key", model="gpt-4o", rpm=0, tpm=0)
        result = await client.chat("Translate this")

        assert isinstance(result, LLMResponse)
        assert result.content == "翻译文本"
        assert result.model == "gpt-4o"

    @pytest.mark.asyncio
    @patch("services.translate.llm.client.httpx.AsyncClient")
    async def test_chat_request_url_anthropic(self, mock_client_class):
        """Test that correct URL is used for Anthropic."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "content": [{"text": "Translated"}],
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = LLMClient(api_key="test-key", rpm=0, tpm=0)
        await client.chat("Test")

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.anthropic.com/v1/messages"

    @pytest.mark.asyncio
    @patch("services.translate.llm.client.httpx.AsyncClient")
    async def test_chat_request_url_openai(self, mock_client_class):
        """Test that correct URL is used for OpenAI."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Translated"}}],
        }
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = LLMClient(base_url="https://api.openai.com", api_key="test-key", rpm=0, tpm=0)
        await client.chat("Test")

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"


class TestLLMClientContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test that context manager properly closes client."""
        client = LLMClient(api_key="test-key")
        # Get the client to create the internal httpx.AsyncClient
        _ = await client._get_client()

        async with client:
            pass

        # After context manager exits, client should be closed
        assert client._client is None


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_response_creation(self):
        """Test creating LLMResponse."""
        response = LLMResponse(
            content="Translated text",
            model="claude-sonnet-4-20250514",
            usage={"input_tokens": 100},
        )

        assert response.content == "Translated text"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.usage == {"input_tokens": 100}

    def test_response_optional_usage(self):
        """Test LLMResponse with optional usage."""
        response = LLMResponse(content="Text", model="gpt-4o")

        assert response.content == "Text"
        assert response.model == "gpt-4o"
        assert response.usage is None


class TestLLMClientIntegration:
    """Integration tests that make actual API calls.

    These tests require valid API keys in config.toml and will skip if not available.
    Use: pytest tests/test_llm_client.py::TestLLMClientIntegration -v -s to run.
    """

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_chat_with_config(self):
        """Test actual chat using config from config.toml."""
        from core.config import config

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

        prompt = "Translate the following English to Chinese: Hello, world!"

        try:
            response = await client.chat(prompt)
            print(f"\nResponse: {response.content}")
            print(f"Model: {response.model}")
            print(f"Stats: {client.stats}")

            assert response.content
            assert len(response.content) > 0
            assert response.model == llm_config.model
        finally:
            await client.aclose()