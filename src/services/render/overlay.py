"""PDF Overlay rendering service.

This module provides functionality to overlay colored bounding boxes on PDF pages
based on MinerU layout analysis results.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from loguru import logger
from pypdf import PdfReader, PdfWriter

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import Color
import io


class BlockType(str, Enum):
    """MinerU block types."""

    TITLE = "title"
    TEXT = "text"
    IMAGE = "image"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    INTERLINE_EQUATION = "interline_equation"
    OTHER = "other"


# Color mapping for different block types (R, G, B)
# Each color is a tuple of (r, g, b) values from 0 to 1
BLOCK_TYPE_COLORS: Dict[BlockType, Tuple[float, float, float]] = {
    BlockType.TITLE: (1.0, 0.0, 0.0),  # Red
    BlockType.TEXT: (0.0, 0.0, 1.0),  # Blue
    BlockType.IMAGE: (0.0, 1.0, 0.0),  # Green
    BlockType.CODE: (1.0, 1.0, 0.0),  # Yellow
    BlockType.TABLE: (1.0, 0.0, 1.0),  # Magenta
    BlockType.LIST: (0.0, 1.0, 1.0),  # Cyan
    BlockType.INTERLINE_EQUATION: (1.0, 0.5, 0.0),  # Orange
    BlockType.OTHER: (0.5, 0.5, 0.5),  # Gray
}

# Color mapping for span types
SPAN_TYPE_COLORS: Dict[str, Tuple[float, float, float]] = {
    "text": (0.0, 0.0, 0.7),  # Dark blue
    "inline_equation": (0.8, 0.2, 0.2),  # Dark red
    "equation": (0.8, 0.4, 0.0),  # Dark orange
}


@dataclass
class BoundingBox:
    """Bounding box representation."""

    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        """Create BoundingBox from list [x0, y0, x1, y1]."""
        return cls(x0=coords[0], y0=coords[1], x1=coords[2], y1=coords[3])

    def to_list(self) -> List[float]:
        """Convert to list [x0, y0, x1, y1]."""
        return [self.x0, self.y0, self.x1, self.y1]


@dataclass
class LayoutBlock:
    """A layout block from MinerU analysis."""

    bbox: BoundingBox
    block_type: BlockType
    page_index: int
    index: int
    lines: List["LayoutLine"] = field(default_factory=list)
    json_path: str = ""  # JSONPath to locate this block in layout data


@dataclass
class LayoutLine:
    """A line within a layout block."""

    bbox: BoundingBox
    spans: List["LayoutSpan"] = field(default_factory=list)
    json_path: str = ""  # JSONPath to locate this line in layout data


@dataclass
class LayoutSpan:
    """A span within a line."""

    bbox: BoundingBox
    span_type: str
    content: str
    json_path: str = ""  # JSONPath to locate this span in layout data


class OverlayRenderer:
    """PDF overlay renderer for visualizing MinerU layout analysis results.

    This class provides methods to overlay colored bounding boxes on PDF pages
    based on MinerU layout analysis data.

    Usage:
        from services.render.overlay import OverlayRenderer

        # Load layout data
        with open("layout.json") as f:
            layout_data = json.load(f)

        # Create renderer and render overlay
        renderer = OverlayRenderer()
        output_path = await renderer.render_overlay(
            layout_data=layout_data,
            pdf_path=Path("input.pdf"),
            output_path=Path("output.pdf"),
        )
    """

    def __init__(
        self,
        border_width: float = 1.0,
        alpha: float = 0.3,
        color_map: Optional[Dict[BlockType, Tuple[float, float, float]]] = None,
        show_json_path: bool = True,
        font_size: float = 6.0,
    ):
        """Initialize the OverlayRenderer.

        Args:
            border_width: Width of the border around each bounding box.
            alpha: Alpha transparency for the fill color (0-1).
            color_map: Optional custom color mapping for block types.
            show_json_path: Whether to show JSONPath text on the overlay.
            font_size: Font size for JSONPath text.
        """
        self.border_width = border_width
        self.alpha = alpha
        self.color_map = color_map or BLOCK_TYPE_COLORS
        self.show_json_path = show_json_path
        self.font_size = font_size

    def _get_color(self, block_type: str) -> Tuple[float, float, float]:
        """Get color for a block type.

        Args:
            block_type: The block type string from MinerU.

        Returns:
            Tuple of (r, g, b) color values.
        """
        try:
            bt = BlockType(block_type)
        except ValueError:
            bt = BlockType.OTHER
        return self.color_map.get(bt, self.color_map[BlockType.OTHER])

    def _get_span_color(self, span_type: str) -> Tuple[float, float, float]:
        """Get color for a span type.

        Args:
            span_type: The span type string from MinerU (e.g., "text", "inline_equation").

        Returns:
            Tuple of (r, g, b) color values.
        """
        return SPAN_TYPE_COLORS.get(span_type, SPAN_TYPE_COLORS["text"])

    def parse_layout_data(
        self,
        layout_data: dict,
        include_lines: bool = True,
        include_spans: bool = False,
    ) -> Dict[int, List[LayoutBlock]]:
        """Parse MinerU layout data into LayoutBlock objects.

        Args:
            layout_data: The parsed JSON data from MinerU's layout.json.
            include_lines: Whether to include line-level bounding boxes.
            include_spans: Whether to include span-level bounding boxes.

        Returns:
            Dictionary mapping page index to list of LayoutBlocks.
        """
        result: Dict[int, List[LayoutBlock]] = {}

        pdf_info = layout_data.get("pdf_info", [])
        for page_idx, page_data in enumerate(pdf_info):
            page_idx = page_data.get("page_idx", page_idx)
            para_blocks = page_data.get("para_blocks", [])

            blocks = []
            for pb_idx, pb in enumerate(para_blocks):
                bbox = BoundingBox.from_list(pb.get("bbox", [0, 0, 0, 0]))
                block_type = pb.get("type", "other")
                index = pb.get("index", 0)

                # Build JSONPath for this block
                block_json_path = f"$.pdf_info[{page_idx}].para_blocks[{pb_idx}]"

                lines = []
                if include_lines:
                    for line_idx, line_data in enumerate(pb.get("lines", [])):
                        line_bbox = BoundingBox.from_list(line_data.get("bbox", [0, 0, 0, 0]))
                        line_json_path = f"{block_json_path}.lines[{line_idx}]"

                        spans = []
                        if include_spans:
                            for span_idx, span_data in enumerate(line_data.get("spans", [])):
                                span_bbox = BoundingBox.from_list(span_data.get("bbox", [0, 0, 0, 0]))
                                span = LayoutSpan(
                                    bbox=span_bbox,
                                    span_type=span_data.get("type", "text"),
                                    content=span_data.get("content", ""),
                                    json_path=f"{line_json_path}.spans[{span_idx}]",
                                )
                                spans.append(span)
                        line = LayoutLine(bbox=line_bbox, spans=spans, json_path=line_json_path)
                        lines.append(line)

                block = LayoutBlock(
                    bbox=bbox,
                    block_type=BlockType(block_type) if block_type in [e.value for e in BlockType] else BlockType.OTHER,
                    page_index=page_idx,
                    index=index,
                    lines=lines,
                    json_path=block_json_path,
                )
                blocks.append(block)

            result[page_idx] = blocks

        logger.info(f"Parsed {len(result)} pages with {sum(len(v) for v in result.values())} blocks")
        return result

    def _convert_bbox_to_pdf_coords(
        self,
        bbox: BoundingBox,
        page_height: float,
    ) -> Tuple[float, float, float, float]:
        """Convert bbox coordinates from MinerU to PDF coordinate system.

        MinerU uses top-left origin (y increases downward),
        PDF uses bottom-left origin (y increases upward).

        Args:
            bbox: The bounding box in MinerU coordinates.
            page_height: The height of the PDF page.

        Returns:
            Tuple of (x0, y0, x1, y1) in PDF coordinates.
        """
        # x coordinates remain the same
        x0 = bbox.x0
        x1 = bbox.x1

        # y coordinates need to be flipped
        # MinerU: y0 is top, y1 is bottom
        # PDF: y0 is bottom, y1 is top
        y0 = page_height - bbox.y1
        y1 = page_height - bbox.y0

        return (x0, y0, x1, y1)

    def _create_overlay_page(
        self,
        page_width: float,
        page_height: float,
        blocks: List[LayoutBlock],
        include_spans: bool = False,
    ) -> io.BytesIO:
        """Create an overlay page with bounding boxes.

        Args:
            page_width: Width of the page in points.
            page_height: Height of the page in points.
            blocks: List of LayoutBlocks to render.
            include_spans: Whether to include span-level bounding boxes.

        Returns:
            BytesIO containing the PDF overlay page.
        """
        # Colors for lines (different from block colors to differentiate)
        LINE_COLORS = [
            (0.0, 0.5, 1.0),   # Light blue
            (0.0, 0.7, 0.7),   # Teal
            (0.5, 0.5, 1.0),   # Purple-blue
            (0.3, 0.6, 0.3),   # Light green
            (0.6, 0.4, 0.2),   # Brown
            (0.5, 0.3, 0.5),   # Plum
            (0.4, 0.6, 0.6),   # Steel blue
            (0.7, 0.5, 0.3),  # Tan
        ]

        packet = io.BytesIO()
        c = canvas.Canvas(packet, pagesize=(page_width, page_height))

        for block_idx, block in enumerate(blocks):
            # Get color for block type
            color = self._get_color(block.block_type.value)
            r, g, b = color

            # Convert bbox to PDF coordinates
            x0, y0, x1, y1 = self._convert_bbox_to_pdf_coords(block.bbox, page_height)

            # Draw rectangle with border and transparent fill
            c.setStrokeColorRGB(r, g, b)
            c.setFillColorRGB(r, g, b, alpha=self.alpha)
            c.setLineWidth(self.border_width)

            # Draw rectangle (x, y, width, height) - note: rect uses bottom-left corner
            c.rect(x0, y0, x1 - x0, y1 - y0, fill=True, stroke=True)

            # Draw JSONPath text next to the block if enabled
            if self.show_json_path and block.json_path:
                c.setFillColorRGB(0, 0, 0)  # Black text
                c.setFont("Helvetica", self.font_size)
                # Place text at top-left corner of the bbox
                c.drawString(x0, y1 + 2, block.json_path)

            # Optionally draw line-level bounding boxes with different colors
            for line_idx, line in enumerate(block.lines):
                line_x0, line_y0, line_x1, line_y1 = self._convert_bbox_to_pdf_coords(
                    line.bbox, page_height
                )
                # Use a different color for each line based on index
                line_color = LINE_COLORS[line_idx % len(LINE_COLORS)]
                lr, lg, lb = line_color

                # Draw line rectangle with its own color
                c.setStrokeColorRGB(lr, lg, lb)
                c.setFillColorRGB(lr, lg, lb, alpha=self.alpha * 0.5)
                c.setLineWidth(0.5)
                c.rect(line_x0, line_y0, line_x1 - line_x0, line_y1 - line_y0, fill=True, stroke=True)

                # Draw line JSONPath if enabled
                if self.show_json_path and line.json_path:
                    c.setFillColorRGB(0, 0, 0)  # Black text
                    c.setFont("Helvetica", self.font_size - 1)  # Slightly smaller for lines
                    c.drawString(line_x0, line_y1 + 1, line.json_path)

                # Draw span-level bounding boxes if enabled
                if include_spans:
                    for span in line.spans:
                        span_x0, span_y0, span_x1, span_y1 = self._convert_bbox_to_pdf_coords(
                            span.bbox, page_height
                        )

                        # Get color for span type
                        span_color = self._get_span_color(span.span_type)
                        sr, sg, sb = span_color

                        # Draw span rectangle with its own color
                        c.setStrokeColorRGB(sr, sg, sb)
                        c.setFillColorRGB(sr, sg, sb, alpha=self.alpha * 0.7)
                        c.setLineWidth(0.3)
                        c.rect(span_x0, span_y0, span_x1 - span_x0, span_y1 - span_y0, fill=True, stroke=True)

                        # Draw span JSONPath if enabled
                        if self.show_json_path and span.json_path:
                            c.setFillColorRGB(0, 0, 0)  # Black text
                            c.setFont("Helvetica", self.font_size - 2)  # Even smaller for spans
                            c.drawString(span_x0, span_y1 + 0.5, span.json_path)

        c.save()
        packet.seek(0)
        return packet

    async def render_overlay(
        self,
        layout_data: dict,
        pdf_path: Union[str, Path],
        output_path: Union[str, Path],
        include_lines: bool = True,
        include_spans: bool = False,
    ) -> Path:
        """Render overlay on PDF based on MinerU layout data.

        Args:
            layout_data: The parsed JSON data from MinerU's layout.json.
            pdf_path: Path to the source PDF file.
            output_path: Path to save the output PDF with overlay.
            include_lines: Whether to include line-level bounding boxes.
            include_spans: Whether to include span-level bounding boxes (e.g., inline_equation).

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

        # Parse layout data
        blocks_by_page = self.parse_layout_data(layout_data, include_lines=include_lines, include_spans=include_spans)

        # Create writer with clone_from to properly attach pages
        writer = PdfWriter(clone_from=reader)

        # Process each page
        for page_num, page in enumerate(writer.pages):
            page_width = float(page.mediabox.width)
            page_height = float(page.mediabox.height)

            # Get blocks for this page
            blocks = blocks_by_page.get(page_num, [])

            if blocks:
                # Create overlay page
                overlay_packet = self._create_overlay_page(page_width, page_height, blocks, include_spans=include_spans)
                overlay_pdf = PdfReader(overlay_packet)
                overlay_page = overlay_pdf.pages[0]

                # Merge overlay with original page
                page.merge_page(overlay_page)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save output
        logger.info(f"Saving overlay PDF: {output_path}")
        with open(output_path, "wb") as f:
            writer.write(f)

        logger.info(f"Overlay PDF saved to: {output_path}")
        return output_path


class OverlayManager:
    """High-level manager for PDF overlay operations.

    This class provides a simplified interface for rendering overlays
    and includes utility methods for loading layout data.

    Usage:
        from services.render.overlay import OverlayManager

        manager = OverlayManager()
        output_path = await manager.render_from_files(
            layout_json_path=Path("layout.json"),
            pdf_path=Path("input.pdf"),
            output_path=Path("output.pdf"),
        )
    """

    def __init__(self, **kwargs):
        """Initialize the OverlayManager.

        Args:
            **kwargs: Additional arguments passed to OverlayRenderer.
        """
        self.renderer = OverlayRenderer(**kwargs)

    @classmethod
    async def from_files(
        cls,
        layout_json_path: Union[str, Path],
        pdf_path: Union[str, Path],
        output_path: Union[str, Path],
        **kwargs,
    ) -> Path:
        """Create overlay from files.

        This is a convenience class method that loads the layout JSON
        and renders the overlay in one call.

        Args:
            layout_json_path: Path to the layout.json file.
            pdf_path: Path to the source PDF file.
            output_path: Path to save the output PDF.
            **kwargs: Additional arguments passed to OverlayManager.

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
        return await manager.render_overlay(
            layout_data=layout_data,
            pdf_path=pdf_path,
            output_path=output_path,
        )

    async def render_overlay(
        self,
        layout_data: dict,
        pdf_path: Union[str, Path],
        output_path: Union[str, Path],
        include_lines: bool = True,
        include_spans: bool = False,
    ) -> Path:
        """Render overlay on PDF based on layout data.

        Args:
            layout_data: The parsed JSON data from MinerU's layout.json.
            pdf_path: Path to the source PDF file.
            output_path: Path to save the output PDF with overlay.
            include_lines: Whether to include line-level bounding boxes.
            include_spans: Whether to include span-level bounding boxes (e.g., inline_equation).

        Returns:
            Path to the output PDF file.
        """
        return await self.renderer.render_overlay(
            layout_data=layout_data,
            pdf_path=pdf_path,
            output_path=output_path,
            include_lines=include_lines,
            include_spans=include_spans,
        )

    async def render_from_dict(
        self,
        layout_data: dict,
        pdf_path: Union[str, Path],
        output_dir: Union[str, Path],
        include_lines: bool = True,
        include_spans: bool = False,
    ) -> Path:
        """Render overlay and save to output directory.

        Args:
            layout_data: The parsed JSON data from MinerU's layout.json.
            pdf_path: Path to the source PDF file.
            output_dir: Directory to save the output PDF.
            include_lines: Whether to include line-level bounding boxes.
            include_spans: Whether to include span-level bounding boxes (e.g., inline_equation).

        Returns:
            Path to the output PDF file.
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        output_path = output_dir / f"{pdf_path.stem}_overlay.pdf"
        return await self.render_overlay(
            layout_data=layout_data,
            pdf_path=pdf_path,
            output_path=output_path,
            include_lines=include_lines,
            include_spans=include_spans,
        )