"""PDF Translation rendering service.

This module provides functionality to replace PDF text content with translations
based on MinerU layout analysis results that contain "translated" fields.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
from loguru import logger
from pypdf import PdfReader, PdfWriter

from services.render.font import FontManager


@dataclass
class TranslationItem:
    """A translation item with position and content."""

    bbox: List[float]  # [x0, y0, x1, y1] in MinerU coordinates
    original: str
    translated: str
    page_index: int
    para_index: int
    line_index: int
    span_index: int


class TranslationRenderer:
    """PDF translation renderer for replacing text with translations.

    This class provides methods to render translated text on PDF pages
    based on MinerU layout analysis data with "translated" fields.

    Usage:
        from services.render.translation import TranslationRenderer

        # Load layout data (with translated fields)
        with open("layout_translated.json") as f:
            layout_data = json.load(f)

        # Create renderer and render translations
        renderer = TranslationRenderer()
        output_path = await renderer.render_translation(
            layout_data=layout_data,
            pdf_path=Path("input.pdf"),
            output_path=Path("output.pdf"),
        )
    """

    def __init__(
        self,
        font_name: Optional[str] = None,
        font_size: Optional[float] = None,
        text_color: tuple = (0.0, 0.0, 0.0),  # Black
        fill_color: tuple = (1.0, 1.0, 1.0),  # White (for covering original text)
        line_spacing: float = 1.2,
        margin: float = 2.0,
    ):
        """Initialize the TranslationRenderer.

        Args:
            font_name: Font name to use for translated text. Defaults to config value.
            font_size: Base font size for translated text. Defaults to config value.
            text_color: RGB tuple for text color (0-1 range).
            fill_color: RGB tuple for background fill color (0-1 range).
            line_spacing: Line spacing multiplier.
            margin: Margin around text in points.
        """
        self.text_color = text_color
        self.fill_color = fill_color
        self.line_spacing = line_spacing
        self.margin = margin

        # Use FontManager for font operations
        self.font_manager = FontManager(font_name=font_name, font_size=font_size)

    @property
    def font_name(self) -> str:
        """Return the registered font name."""
        return self.font_manager.font_name

    @property
    def font_size(self) -> float:
        """Return the base font size."""
        return self.font_manager.font_size

    def _convert_bbox_to_pdf_coords(
        self,
        bbox: List[float],
        page_height: float,
    ) -> Tuple[float, float, float, float]:
        """Convert bbox coordinates from MinerU to PDF coordinate system.

        MinerU uses top-left origin (y increases downward),
        PDF uses bottom-left origin (y increases upward).

        Args:
            bbox: The bounding box in MinerU coordinates [x0, y0, x1, y1].
            page_height: The height of the PDF page.

        Returns:
            Tuple of (x0, y0, x1, y1) in PDF coordinates.
        """
        x0 = bbox[0]
        x1 = bbox[2]
        y0 = page_height - bbox[3]  # top -> bottom
        y1 = page_height - bbox[1]  # bottom -> top
        return (x0, y0, x1, y1)

    def _extract_translation_items(
        self,
        layout_data: dict,
    ) -> Dict[int, List[TranslationItem]]:
        """Extract translation items from layout data.

        Args:
            layout_data: The parsed JSON data from MinerU's layout.json.

        Returns:
            Dictionary mapping page index to list of TranslationItems.
        """
        result: Dict[int, List[TranslationItem]] = {}

        pdf_info = layout_data.get("pdf_info", [])
        for page_idx, page_data in enumerate(pdf_info):
            page_index = page_data.get("page_idx", page_idx)
            para_blocks = page_data.get("para_blocks", [])

            items = []
            for para_idx, para in enumerate(para_blocks):
                para_index = para.get("index", para_idx)

                for line_idx, line in enumerate(para.get("lines", [])):
                    for span_idx, span in enumerate(line.get("spans", [])):
                        # Only process text spans with translated content
                        if span.get("type") == "text":
                            original = span.get("content", "")
                            translated = span.get("translated")

                            if translated:
                                item = TranslationItem(
                                    bbox=span.get("bbox", [0, 0, 0, 0]),
                                    original=original,
                                    translated=translated,
                                    page_index=page_index,
                                    para_index=para_index,
                                    line_index=line_idx,
                                    span_index=span_idx,
                                )
                                items.append(item)

            if items:
                result[page_index] = items

        total = sum(len(v) for v in result.values())
        logger.info(f"Extracted {total} translation items from {len(result)} pages")
        return result

    def _estimate_font_size(
        self,
        bbox_width: float,
        text: str,
        font_name: str,
        base_size: float,
    ) -> float:
        """Estimate appropriate font size based on bbox width and text length.

        Args:
            bbox_width: Width of the bounding box in points.
            text: The text to fit.
            font_name: Font name to use (kept for compatibility, ignored).
            base_size: Base font size to start from.

        Returns:
            Estimated font size that fits the bbox.
        """
        return self.font_manager.estimate_font_size(bbox_width, text, base_size)

    def _wrap_text(
        self,
        text: str,
        bbox_width: float,
        font_name: str,
        font_size: float,
    ) -> List[str]:
        """Wrap text to fit within the bounding box width.

        Args:
            text: The text to wrap.
            bbox_width: Width of the bounding box in points.
            font_name: Font name to use (kept for compatibility, ignored).
            font_size: Current font size.

        Returns:
            List of wrapped text lines.
        """
        return self.font_manager.wrap_text(text, bbox_width, font_size)

    async def render_translation(
        self,
        layout_data: dict,
        pdf_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Path:
        """Render translated text on PDF based on layout data.

        Args:
            layout_data: The parsed JSON data from MinerU's layout.json (with translated fields).
            pdf_path: Path to the source PDF file.
            output_path: Path to save the output PDF with translations.

        Returns:
            Path to the output PDF file.

        Raises:
            FileNotFoundError: If the source PDF file does not exist.
            ValueError: If the layout data is invalid.
        """
        pdf_path = Path(pdf_path)
        output_path = Path(output_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"Source PDF not found: {pdf_path}")

        logger.info(f"Loading PDF: {pdf_path}")
        reader = PdfReader(pdf_path)

        # Extract translation items
        items_by_page = self._extract_translation_items(layout_data)

        if not items_by_page:
            logger.warning("No translation items found in layout data")
            # Just copy the original PDF
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                writer.write(f)
            return output_path

        # Use PyMuPDF to draw translations directly on pages
        src_doc = fitz.open(str(pdf_path))

        # Process each page
        for page_num in range(len(src_doc)):
            page = src_doc[page_num]
            page_width = page.rect.width
            page_height = page.rect.height

            # Get translation items for this page
            items = items_by_page.get(page_num, [])

            if items:
                # Get font for PyMuPDF
                font_name = self.font_manager.get_font_name_for_pymupdf()
                font_file = self.font_manager.get_font_for_pymupdf()

                for item in items:
                    # PyMuPDF uses PDF coordinate system (origin at bottom-left)
                    # MinerU also uses bottom-left origin with y increasing upward
                    # So we can use the bbox coordinates directly
                    bbox = item.bbox
                    x0 = bbox[0]
                    x1 = bbox[2]
                    y0 = bbox[1]  # bottom in PDF coordinates
                    y1 = bbox[3]  # top in PDF coordinates

                    bbox_width = x1 - x0
                    bbox_height = y1 - y0

                    if bbox_width <= 0 or bbox_height <= 0:
                        logger.warning(f"Invalid bbox for translation: {item.bbox}")
                        continue

                    # Calculate effective font size based on bbox
                    font_size = self._estimate_font_size(
                        bbox_width - 2 * self.margin,
                        item.translated,
                        self.font_name,
                        self.font_size,
                    )

                    # Wrap text to fit bbox width
                    lines = self._wrap_text(
                        item.translated,
                        bbox_width - 2 * self.margin,
                        self.font_name,
                        font_size,
                    )

                    # Calculate total text height
                    line_height = font_size * self.line_spacing
                    total_text_height = len(lines) * line_height

                    # Adjust starting Y to center vertically or align to bottom
                    start_y = y0 + (bbox_height - total_text_height) / 2 + font_size

                    # Draw white background to cover original text
                    page.draw_rect(
                        fitz.Rect(x0, y0, x1, y1),
                        color=None,
                        fill=self.fill_color,
                        overlay=True,
                    )

                    # Draw translated text
                    for i, line in enumerate(lines):
                        text_y = start_y + i * line_height
                        if text_y > y1:
                            break  # Don't draw outside bbox

                        # Insert text using PyMuPDF
                        if font_file:
                            page.insert_text(
                                fitz.Point(x0 + self.margin, text_y),
                                line,
                                fontsize=font_size,
                                fontfile=font_file,
                                color=self.text_color,
                                overlay=True,
                            )
                        else:
                            page.insert_text(
                                fitz.Point(x0 + self.margin, text_y),
                                line,
                                fontsize=font_size,
                                fontname=font_name,
                                color=self.text_color,
                                overlay=True,
                            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save output
        logger.info(f"Saving translated PDF: {output_path}")
        src_doc.save(str(output_path))
        src_doc.close()

        logger.info(f"Translated PDF saved to: {output_path}")
        return output_path


class TranslationManager:
    """High-level manager for PDF translation rendering operations.

    This class provides a simplified interface for rendering translations
    and includes utility methods for loading layout data.

    Usage:
        from services.render.translation import TranslationManager

        manager = TranslationManager()
        output_path = await manager.render_from_files(
            layout_json_path=Path("layout_translated.json"),
            pdf_path=Path("input.pdf"),
            output_path=Path("output.pdf"),
        )
    """

    def __init__(self, **kwargs):
        """Initialize the TranslationManager.

        Args:
            **kwargs: Additional arguments passed to TranslationRenderer.
        """
        self.renderer = TranslationRenderer(**kwargs)

    @classmethod
    async def from_files(
        cls,
        layout_json_path: Union[str, Path],
        pdf_path: Union[str, Path],
        output_path: Union[str, Path],
        **kwargs,
    ) -> Path:
        """Create translated PDF from files.

        This is a convenience class method that loads the layout JSON
        and renders the translations in one call.

        Args:
            layout_json_path: Path to the layout.json file (with translated fields).
            pdf_path: Path to the source PDF file.
            output_path: Path to save the output PDF.
            **kwargs: Additional arguments passed to TranslationManager.

        Returns:
            Path to the output PDF file.
        """
        layout_json_path = Path(layout_json_path)
        pdf_path = Path(pdf_path)
        output_path = Path(output_path)

        if not layout_json_path.exists():
            raise FileNotFoundError(f"Layout JSON not found: {layout_json_path}")

        logger.info(f"Loading layout data from: {layout_json_path}")
        with open(layout_json_path) as f:
            layout_data = json.load(f)

        manager = cls(**kwargs)
        return await manager.render_translation(
            layout_data=layout_data,
            pdf_path=pdf_path,
            output_path=output_path,
        )

    async def render_translation(
        self,
        layout_data: dict,
        pdf_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Path:
        """Render translated text on PDF based on layout data.

        Args:
            layout_data: The parsed JSON data from MinerU's layout.json (with translated fields).
            pdf_path: Path to the source PDF file.
            output_path: Path to save the output PDF with translations.

        Returns:
            Path to the output PDF file.
        """
        return await self.renderer.render_translation(
            layout_data=layout_data,
            pdf_path=pdf_path,
            output_path=output_path,
        )