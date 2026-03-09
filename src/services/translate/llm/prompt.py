"""Translation prompt generator based on layout.json."""

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Optional

import yaml


def _load_template() -> str:
    """Load translation template from YAML file.

    Returns:
        The translation template string
    """
    template_file = files("services.translate.llm").joinpath("templates.yaml")
    with open(template_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["translation_template"]


@dataclass
class TextLine:
    """Represents a single line of text in the layout."""
    content: str
    bbox: list[float]
    para_index: int
    para_type: str
    line_index: int


@dataclass
class TranslationContext:
    """Context information for translation."""
    abstract: Optional[str] = None
    para_content: Optional[str] = None
    prev_translated: Optional[str] = None  # Previous line's translation result


@dataclass
class TranslationPromptRequest:
    """Request to generate translation prompt for a text line."""
    line: TextLine
    context: TranslationContext
    target_lang: str = "zh-CN"


class TranslatePromptGenerator:
    """Generator for LLM translation prompts based on layout.json."""

    # Load translation template from YAML file
    _TRANSLATION_TEMPLATE: str | None = None

    @classmethod
    def _get_template(cls) -> str:
        """Get the translation template, loading it if necessary."""
        if cls._TRANSLATION_TEMPLATE is None:
            cls._TRANSLATION_TEMPLATE = _load_template()
        return cls._TRANSLATION_TEMPLATE

    def __init__(self, layout_path: str | Path):
        """Initialize with layout.json file path.

        Args:
            layout_path: Path to the layout.json file
        """
        self.layout_path = Path(layout_path)
        self.layout_data: dict = {}
        self._text_lines: list[TextLine] = []
        self._abstract: Optional[str] = None

    def load_layout(self) -> None:
        """Load and parse the layout.json file."""
        with open(self.layout_path, "r", encoding="utf-8") as f:
            self.layout_data = json.load(f)

    def extract_all_text_lines(self) -> list[TextLine]:
        """Extract all text lines from layout.json.

        Returns:
            List of TextLine objects
        """
        if not self.layout_data:
            self.load_layout()

        self._text_lines = []
        self._abstract = None
        found_abstract_title = False

        for page_info in self.layout_data.get("pdf_info", []):
            for para in page_info.get("para_blocks", []):
                para_type = para.get("type", "text")
                para_index = para.get("index", 0)

                # Extract abstract content - abstract is the paragraph after "Abstract" title
                if found_abstract_title and self._abstract is None:
                    self._abstract = self._extract_para_text(para)
                    found_abstract_title = False  # Reset after getting abstract content

                if para_type == "title":
                    lines = para.get("lines", [])
                    if lines:
                        first_line_content = self._get_first_line_content(lines)
                        if first_line_content and "abstract" in first_line_content.lower():
                            found_abstract_title = True

                for line_idx, line in enumerate(para.get("lines", [])):
                    for span in line.get("spans", []):
                        if span.get("type") == "text" and span.get("content"):
                            text_line = TextLine(
                                content=span["content"],
                                bbox=span.get("bbox", []),
                                para_index=para_index,
                                para_type=para_type,
                                line_index=line_idx,
                            )
                            self._text_lines.append(text_line)

        return self._text_lines

    def _get_first_line_content(self, lines: list) -> str:
        """Get the content of the first line in a list of lines."""
        if not lines:
            return ""
        first_line = lines[0]
        spans = first_line.get("spans", [])
        if spans:
            return spans[0].get("content", "")
        return ""

    def _extract_para_text(self, para: dict) -> str:
        """Extract all text content from a paragraph."""
        texts = []
        for line in para.get("lines", []):
            for span in line.get("spans", []):
                if span.get("type") == "text" and span.get("content"):
                    texts.append(span["content"])
        return " ".join(texts)

    def get_para_content(self, para_index: int) -> Optional[str]:
        """Get the full content of a paragraph by its index.

        Args:
            para_index: The paragraph index

        Returns:
            The paragraph text content, or None if not found
        """
        if not self.layout_data:
            self.load_layout()

        for page_info in self.layout_data.get("pdf_info", []):
            for para in page_info.get("para_blocks", []):
                if para.get("index") == para_index:
                    return self._extract_para_text(para)
        return None

    def get_abstract(self) -> Optional[str]:
        """Get the abstract content from the layout.

        Returns:
            The abstract text, or None if not found
        """
        if self._abstract is None:
            self.extract_all_text_lines()
        return self._abstract

    def build_translation_prompt(
        self,
        line: TextLine,
        target_lang: str = "zh-CN",
        prev_translated: Optional[str] = None,
    ) -> str:
        """Build translation prompt for a single text line.

        Args:
            line: The text line to translate
            target_lang: Target language code
            prev_translated: Previous line's translation result for context

        Returns:
            The formatted prompt for LLM translation
        """
        # Get paragraph content
        para_content = self.get_para_content(line.para_index)

        context = TranslationContext(
            abstract=self.get_abstract(),
            para_content=para_content,
            prev_translated=prev_translated,
        )

        # Build full context string
        full_context = ""
        if context.abstract:
            full_context = f"Paper Abstract:\n{context.abstract}\n"

        # Build para context string
        para_context = context.para_content or "No paragraph context available"

        # Build previous translation context (for continuity)
        if prev_translated:
            prev_context = f"\n\nPrevious sentence translation (for continuity):\n{prev_translated}"
        else:
            prev_context = ""

        return self._get_template().format(
            target_lang=target_lang,
            full_context=full_context,
            para_context=para_context,
            text=line.content,
            prev_context=prev_context,
        )

    def generate_all_prompts(self, target_lang: str = "zh-CN") -> list[dict]:
        """Generate translation prompts for all text lines.

        Args:
            target_lang: Target language code

        Returns:
            List of dicts containing line info and prompt
        """
        if not self._text_lines:
            self.extract_all_text_lines()

        results = []
        for line in self._text_lines:
            prompt = self.build_translation_prompt(line, target_lang)
            results.append({
                "line": line,
                "prompt": prompt,
            })

        return results