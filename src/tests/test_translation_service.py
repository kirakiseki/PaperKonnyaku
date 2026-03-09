"""Tests for translation service."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from core.config import config
from services.translate.llm.service import TranslationService, TranslationResult


# Test asset paths
TEST_ASSETS_DIR = config.test.test_assets_dir
LAYOUT_JSON_PATH = TEST_ASSETS_DIR / "extracted" / "layout.json"


@pytest.fixture
def layout_json_path():
    """Path to test layout.json file."""
    return LAYOUT_JSON_PATH


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = Mock()
    client.translate = AsyncMock(
        return_value=Mock(
            content="Translated: Hello",
            model="test-model",
            usage={"input_tokens": 100, "output_tokens": 50},
        )
    )
    return client


class TestTranslationServiceInit:
    """Tests for TranslationService initialization."""

    def test_init_with_path(self, layout_json_path):
        """Test initialization with path."""
        service = TranslationService(layout_json_path)
        assert service.layout_path == Path(layout_json_path)
        assert service.llm_client is None

    def test_init_with_client(self, layout_json_path, mock_llm_client):
        """Test initialization with client."""
        service = TranslationService(layout_json_path, mock_llm_client)
        assert service.llm_client == mock_llm_client

    def test_init_with_string_path(self, layout_json_path):
        """Test initialization with string path."""
        service = TranslationService(str(layout_json_path))
        assert service.layout_path == Path(layout_json_path)


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""

    def test_creation_success(self):
        """Test creating successful translation result."""
        result = TranslationResult(
            original="Hello",
            translated="你好",
            para_index=1,
            line_index=0,
            success=True,
        )

        assert result.original == "Hello"
        assert result.translated == "你好"
        assert result.para_index == 1
        assert result.line_index == 0
        assert result.success is True
        assert result.error is None

    def test_creation_failure(self):
        """Test creating failed translation result."""
        result = TranslationResult(
            original="Hello",
            translated="Hello",
            para_index=1,
            line_index=0,
            success=False,
            error="API error",
        )

        assert result.success is False
        assert result.error == "API error"
        assert result.translated == "Hello"  # Fallback to original


class TestTranslationServiceTranslate:
    """Tests for translate method."""

    @pytest.mark.asyncio
    async def test_translate_with_mock_client(self, layout_json_path, mock_llm_client):
        """Test translation with mock client."""
        service = TranslationService(layout_json_path, mock_llm_client)

        results = await service.translate(target_lang="zh-CN")

        assert len(results) > 0
        assert all(isinstance(r, TranslationResult) for r in results)
        # Check mock was called
        assert mock_llm_client.translate.called

    @pytest.mark.asyncio
    async def test_translate_progress_callback(self, layout_json_path, mock_llm_client):
        """Test translation with progress callback."""
        service = TranslationService(layout_json_path, mock_llm_client)

        progress_calls = []
        results = await service.translate(
            target_lang="zh-CN",
            progress_callback=lambda current, total: progress_calls.append((current, total)),
        )

        assert len(progress_calls) == len(results)
        assert progress_calls[-1] == (len(results), len(results))

    @pytest.mark.asyncio
    async def test_translate_fallback_on_error(self, layout_json_path):
        """Test that translation falls back to original on error."""
        mock_client = Mock()
        mock_client.translate = AsyncMock(side_effect=Exception("API error"))

        service = TranslationService(layout_json_path, mock_client)
        results = await service.translate(target_lang="zh-CN")

        # All should have fallback to original
        assert all(r.success is False for r in results)
        assert all(r.translated == r.original for r in results)


class TestTranslationServiceTranslateAndSave:
    """Tests for translate_and_save method."""

    @pytest.mark.asyncio
    async def test_translate_and_save(self, layout_json_path, mock_llm_client, tmp_path):
        """Test translating and saving to file."""
        service = TranslationService(layout_json_path, mock_llm_client)

        output_path = tmp_path / "translated_layout.json"

        # Mock extract_all_text_lines to return controlled data
        with patch.object(service.prompt_generator, "extract_all_text_lines", return_value=[]):
            result_path = await service.translate_and_save(
                output_path=output_path,
                target_lang="zh-CN",
            )

        assert result_path == output_path
        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "pdf_info" in data


class TestApplyTranslations:
    """Tests for _apply_translations method."""

    def test_apply_translations(self, layout_json_path):
        """Test applying translations to layout data."""
        service = TranslationService(layout_json_path)

        # Load layout
        with open(layout_json_path, "r", encoding="utf-8") as f:
            layout_data = json.load(f)

        # Create translation map
        translation_map = {
            (1, 0): "Translated Title",
            (6, 0): "Translated first line",
        }

        # Apply translations
        service._apply_translations(layout_data, translation_map)

        # Verify translations were applied
        found_translation = False
        for page_info in layout_data.get("pdf_info", []):
            for para in page_info.get("para_blocks", []):
                if para.get("index") == 6:
                    lines = para.get("lines", [])
                    if lines and lines[0].get("spans"):
                        content = lines[0]["spans"][0].get("content", "")
                        if "Translated first line" in content:
                            found_translation = True

        assert found_translation


class TestTranslationServiceIntegration:
    """Integration tests for translation service."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_translate_with_config(self):
        """Test actual translation using config.toml with random samples."""
        import random

        from core.config import config

        llm_config = config.translate.llm
        if not llm_config.api_key:
            pytest.skip("API key not configured")

        service = TranslationService(LAYOUT_JSON_PATH)

        # Get all lines and randomly sample 5 for quick test
        all_lines = service.prompt_generator.extract_all_text_lines()
        sample_size = min(5, len(all_lines))
        lines = random.sample(all_lines, sample_size)

        results = []
        client = await service._get_llm_client()

        try:
            for line in lines:
                prompt = service.prompt_generator.build_translation_prompt(line, "zh-CN")
                response = await client.translate(prompt)

                from services.translate.llm.service import _parse_xml_response
                translated = _parse_xml_response(response.content)

                results.append(TranslationResult(
                    original=line.content,
                    translated=translated,
                    para_index=line.para_index,
                    line_index=line.line_index,
                    success=True,
                ))
        finally:
            await client.aclose()

        print(f"\nTranslated {len(results)} lines (random sample):")
        for r in results:
            print(f"  Original: {r.original}")
            print(f"  Translated: {r.translated}")
            print()

        assert len(results) == sample_size
        assert all(r.success for r in results)