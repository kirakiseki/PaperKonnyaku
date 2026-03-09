"""Tests for LLM translation prompt generator."""

import pytest
from pathlib import Path

from core.config import config
from services.translate.llm.prompt import (
    TranslatePromptGenerator,
    TextLine,
    TranslationContext,
)


# Test asset paths
TEST_ASSETS_DIR = config.test.test_assets_dir
LAYOUT_JSON_PATH = TEST_ASSETS_DIR / "extracted" / "layout.json"


@pytest.fixture
def layout_json_path():
    """Path to test layout.json file."""
    return LAYOUT_JSON_PATH


@pytest.fixture
def generator(layout_json_path):
    """Create a TranslatePromptGenerator instance."""
    return TranslatePromptGenerator(layout_json_path)


@pytest.fixture
def generator_with_loaded_layout(generator):
    """Create a generator with layout already loaded."""
    generator.load_layout()
    return generator


class TestTranslatePromptGeneratorInit:
    """Tests for TranslatePromptGenerator initialization."""

    def test_init_with_string_path(self, layout_json_path):
        """Test initialization with string path."""
        generator = TranslatePromptGenerator(str(layout_json_path))
        assert generator.layout_path == Path(layout_json_path)

    def test_init_with_path_object(self, layout_json_path):
        """Test initialization with Path object."""
        generator = TranslatePromptGenerator(layout_json_path)
        assert generator.layout_path == layout_json_path

    def test_init_layout_data_empty(self, generator):
        """Test that layout_data is empty before loading."""
        assert generator.layout_data == {}


class TestLoadLayout:
    """Tests for loading layout.json."""

    def test_load_layout(self, generator):
        """Test loading layout.json file."""
        generator.load_layout()
        assert "pdf_info" in generator.layout_data
        assert len(generator.layout_data["pdf_info"]) > 0


class TestExtractTextLines:
    """Tests for extracting text lines from layout."""

    def test_extract_all_text_lines(self, generator):
        """Test extracting all text lines."""
        lines = generator.extract_all_text_lines()
        assert len(lines) > 0
        assert all(isinstance(line, TextLine) for line in lines)

    def test_text_lines_have_required_fields(self, generator):
        """Test that extracted lines have all required fields."""
        lines = generator.extract_all_text_lines()
        first_line = lines[0]
        assert hasattr(first_line, "content")
        assert hasattr(first_line, "bbox")
        assert hasattr(first_line, "para_index")
        assert hasattr(first_line, "para_type")
        assert hasattr(first_line, "line_index")
        assert first_line.content  # content should not be empty

    def test_extract_abstract(self, generator):
        """Test extracting abstract content."""
        abstract = generator.get_abstract()
        assert abstract is not None
        assert len(abstract) > 0
        # Abstract should contain some expected content
        assert "search" in abstract.lower() or "retrieval" in abstract.lower()


class TestGetParaContent:
    """Tests for getting paragraph content."""

    def test_get_para_content(self, generator_with_loaded_layout):
        """Test getting content of a specific paragraph."""
        # Get content for a paragraph
        para_content = generator_with_loaded_layout.get_para_content(6)
        assert para_content is not None
        assert len(para_content) > 0
        assert isinstance(para_content, str)

    def test_get_para_content_not_found(self, generator_with_loaded_layout):
        """Test getting content for non-existent paragraph."""
        para_content = generator_with_loaded_layout.get_para_content(9999)
        assert para_content is None


class TestBuildTranslationPrompt:
    """Tests for building translation prompts."""

    def test_build_prompt_structure(self, generator):
        """Test that generated prompt has correct structure."""
        lines = generator.extract_all_text_lines()
        first_line = lines[0]

        prompt = generator.build_translation_prompt(first_line)

        assert "You are a professional academic paper translator" in prompt
        assert first_line.content in prompt
        assert "Translation:" in prompt

    def test_build_prompt_with_abstract_context(self, generator):
        """Test that abstract is included in prompt when available."""
        lines = generator.extract_all_text_lines()

        # Find a line that should have abstract context
        prompt = generator.build_translation_prompt(lines[0])

        # Abstract should be in the prompt
        assert "Paper Abstract:" in prompt

    def test_build_prompt_with_para_context(self, generator):
        """Test that paragraph context is included in prompt."""
        lines = generator.extract_all_text_lines()
        first_line = lines[0]

        prompt = generator.build_translation_prompt(first_line)

        assert "Current paragraph context:" in prompt
        assert first_line.content in prompt

    def test_build_prompt_target_lang(self, generator):
        """Test that target language is correctly set in prompt."""
        lines = generator.extract_all_text_lines()
        first_line = lines[0]

        # Test default (zh-CN)
        prompt_zh = generator.build_translation_prompt(first_line, "zh-CN")
        assert "zh-CN" in prompt_zh

        # Test Japanese
        prompt_ja = generator.build_translation_prompt(first_line, "ja-JP")
        assert "ja-JP" in prompt_ja


class TestGenerateAllPrompts:
    """Tests for generating all prompts."""

    def test_generate_all_prompts(self, generator):
        """Test generating prompts for all text lines."""
        prompts = generator.generate_all_prompts()

        assert len(prompts) > 0
        assert all("line" in p for p in prompts)
        assert all("prompt" in p for p in prompts)

    def test_generate_all_prompts_count(self, generator):
        """Test that prompt count matches line count."""
        lines = generator.extract_all_text_lines()
        prompts = generator.generate_all_prompts()

        assert len(prompts) == len(lines)


class TestTextLine:
    """Tests for TextLine dataclass."""

    def test_text_line_creation(self):
        """Test creating a TextLine instance."""
        line = TextLine(
            content="Test content",
            bbox=[0, 0, 100, 20],
            para_index=1,
            para_type="text",
            line_index=0,
        )

        assert line.content == "Test content"
        assert line.bbox == [0, 0, 100, 20]
        assert line.para_index == 1
        assert line.para_type == "text"
        assert line.line_index == 0


class TestTranslationContext:
    """Tests for TranslationContext dataclass."""

    def test_translation_context_creation(self):
        """Test creating a TranslationContext instance."""
        context = TranslationContext(
            abstract="This is an abstract",
            para_content="This is a paragraph",
        )

        assert context.abstract == "This is an abstract"
        assert context.para_content == "This is a paragraph"

    def test_translation_context_optional_fields(self):
        """Test TranslationContext with optional fields."""
        context = TranslationContext()

        assert context.abstract is None
        assert context.para_content is None


class TestPrintPrompt:
    """Test to print prompt for debugging."""

    def test_print_prompt_for_first_line(self, generator):
        """Print prompt for first few lines to debug."""
        lines = generator.extract_all_text_lines()

        # Print prompt for first 3 lines
        for i in range(min(3, len(lines))):
            print(f"\n{'='*60}")
            print(f"Line {i}: {lines[i].content}")
            print(f"{'='*60}")

            prev = lines[i-1].content if i > 0 else None
            prompt = generator.build_translation_prompt(lines[i], "zh-CN", prev)
            print(prompt)
            print(f"{'='*60}\n")