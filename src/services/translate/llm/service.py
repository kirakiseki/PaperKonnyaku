"""Translation service that combines prompt generation and LLM client."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from services.translate.llm.client import LLMClient, LLMResponse
from services.translate.llm.prompt import TranslatePromptGenerator


@dataclass
class TranslationResult:
    """Result of translating a text line."""
    original: str
    translated: str
    para_index: int
    line_index: int
    success: bool
    error: Optional[str] = None


def _parse_xml_response(content: str) -> Optional[str]:
    """Parse XML response from LLM to extract translation.

    Args:
        content: Raw response content from LLM

    Returns:
        Extracted translation string, or None if parsing fails
    """
    # Try to find content between <translation> tags
    match = re.search(r"<translation>(.*?)</translation>", content, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If parsing fails, return the raw content
    logger.warning(f"Failed to parse XML response, using raw content: {content[:100]}...")
    return content.strip()


class TranslationService:
    """Service for translating PDF layout content using LLM."""

    def __init__(
        self,
        layout_path: str | Path,
        llm_client: Optional[LLMClient] = None,
    ):
        """Initialize translation service.

        Args:
            layout_path: Path to the layout.json file
            llm_client: Optional LLM client (will create from config if not provided)
        """
        self.layout_path = Path(layout_path)
        self.prompt_generator = TranslatePromptGenerator(layout_path)
        self.llm_client = llm_client

    async def _get_llm_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self.llm_client is None:
            from core.config import config

            llm_config = config.translate.llm
            self.llm_client = LLMClient(
                base_url=llm_config.base_url,
                api_key=llm_config.api_key,
                model=llm_config.model,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                rpm=llm_config.rpm,
                tpm=llm_config.tpm,
                max_concurrent=llm_config.max_concurrent,
            )
        return self.llm_client

    async def translate(
        self,
        target_lang: str = "zh-CN",
        progress_callback: Optional[callable] = None,
    ) -> list[TranslationResult]:
        """Translate all text lines in the layout.

        Args:
            target_lang: Target language code
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            List of TranslationResult objects
        """
        # Extract all text lines
        lines = self.prompt_generator.extract_all_text_lines()
        total = len(lines)
        results: list[TranslationResult] = []

        logger.info(f"Starting translation of {total} lines")

        client = await self._get_llm_client()

        prev_translated: Optional[str] = None

        try:
            for idx, line in enumerate(lines):
                # Build prompt with previous translation context
                prompt = self.prompt_generator.build_translation_prompt(
                    line, target_lang, prev_translated
                )

                try:
                    # Send to LLM
                    response = await client.chat(prompt)

                    # Parse JSON response to extract translation
                    translated = _parse_xml_response(response.content)

                    result = TranslationResult(
                        original=line.content,
                        translated=translated,
                        para_index=line.para_index,
                        line_index=line.line_index,
                        success=True,
                    )
                except Exception as e:
                    logger.error(f"Translation failed for line {idx}: {e}")
                    result = TranslationResult(
                        original=line.content,
                        translated=line.content,  # Fallback to original
                        para_index=line.para_index,
                        line_index=line.line_index,
                        success=False,
                        error=str(e),
                    )

                results.append(result)

                # Update previous translation for next line continuity
                if result.success:
                    prev_translated = result.translated
                else:
                    # Use original if translation failed
                    prev_translated = line.content

                if progress_callback:
                    progress_callback(idx + 1, total)

                # Log progress every 50 lines
                if (idx + 1) % 50 == 0:
                    logger.info(f"Translated {idx + 1}/{total} lines")

        finally:
            if self.llm_client is None:
                await client.aclose()

        success_count = sum(1 for r in results if r.success)
        logger.info(f"Translation completed: {success_count}/{total} successful")

        return results

    async def translate_and_save(
        self,
        output_path: Optional[str | Path] = None,
        target_lang: str = "zh-CN",
        progress_callback: Optional[callable] = None,
    ) -> Path:
        """Translate layout and save the result to a new JSON file.

        Args:
            output_path: Optional output path (defaults to adding "_translated" suffix)
            target_lang: Target language code
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = self.layout_path.parent / f"{self.layout_path.stem}_translated.json"
        else:
            output_path = Path(output_path)

        # Load original layout
        with open(self.layout_path, "r", encoding="utf-8") as f:
            layout_data = json.load(f)

        # Translate
        results = await self.translate(target_lang, progress_callback)

        # Create a lookup dict for translations
        translation_map: dict[tuple[int, int], str] = {}
        for result in results:
            if result.success:
                translation_map[(result.para_index, result.line_index)] = result.translated

        # Apply translations to layout
        self._apply_translations(layout_data, translation_map)

        # Save translated layout
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(layout_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Translated layout saved to: {output_path}")
        return output_path

    def _apply_translations(
        self,
        layout_data: dict,
        translation_map: dict[tuple[int, int], str],
    ) -> None:
        """Apply translations to layout data in place.

        Args:
            layout_data: The layout data to modify
            translation_map: Dict mapping (para_index, line_index) -> translated text
        """
        for page_info in layout_data.get("pdf_info", []):
            for para in page_info.get("para_blocks", []):
                para_index = para.get("index", 0)

                for line in para.get("lines", []):
                    line_idx = para.get("lines", []).index(line)

                    for span in line.get("spans", []):
                        if span.get("type") == "text" and span.get("content"):
                            key = (para_index, line_idx)
                            if key in translation_map:
                                span["translated"] = translation_map[key]