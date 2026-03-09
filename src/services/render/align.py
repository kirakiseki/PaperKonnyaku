"""BBox alignment service for PDF layout correction.

This module provides functionality to align text bounding boxes in PDF layouts,
ensuring proper text alignment within paragraphs and between columns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from loguru import logger

from core.config import config


@dataclass
class ColumnInfo:
    """Information about a text column."""

    x_start: float  # Left edge of the column
    x_end: float    # Right edge of the column
    blocks: List["LayoutBlock"] = field(default_factory=list)


@dataclass
class LayoutBlock:
    """A layout block from MinerU analysis."""

    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float
    block_type: str
    page_index: int
    index: int
    lines: List["LayoutLine"] = field(default_factory=list)
    json_path: str = ""


@dataclass
class LayoutLine:
    """A line within a layout block."""

    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float
    spans: List["LayoutSpan"] = field(default_factory=list)
    json_path: str = ""


@dataclass
class LayoutSpan:
    """A span within a line."""

    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float
    span_type: str
    content: str
    json_path: str = ""


class BBoxAligner:
    """Aligns text bounding boxes in PDF layouts.

    This class provides methods to correct bbox positions for:
    1. Intra-paragraph alignment: All lines in a paragraph align their
       left and right edges to the paragraph's bbox edges.
    2. Inter-paragraph alignment: Vertically stacked paragraphs in the
       same column align their edges together.

    Usage:
        from services.render.align import BBoxAligner

        # Parse layout data
        aligner = BBoxAligner()
        blocks = aligner.parse_layout_data(layout_data)

        # Align bboxes
        aligned_blocks = aligner.align(blocks)

        # Export aligned layout data
        aligned_data = aligner.export_layout_data(aligned_blocks, original_layout_data)
    """

    def __init__(
        self,
        intra_paragraph_align: bool = True,
        inter_paragraph_align: bool = True,
        outlier_width_threshold: float = None,
    ):
        """Initialize the BBoxAligner.

        Args:
            intra_paragraph_align: Whether to align text within paragraphs.
            inter_paragraph_align: Whether to align paragraphs in the same column.
            outlier_width_threshold: Threshold for detecting outlier blocks.
                Blocks with width difference ratio >= this threshold will be kept as-is.
                Defaults to config value.
        """
        self.intra_paragraph_align = intra_paragraph_align
        self.inter_paragraph_align = inter_paragraph_align
        self.outlier_width_threshold = outlier_width_threshold or config.render.outlier_width_threshold

    def parse_layout_data(
        self,
        layout_data: dict,
        include_spans: bool = True,
    ) -> Dict[int, List[LayoutBlock]]:
        """Parse MinerU layout data into LayoutBlock objects.

        Args:
            layout_data: The parsed JSON data from MinerU's layout.json.
            include_spans: Whether to include span-level data.

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
                bbox = pb.get("bbox", [0, 0, 0, 0])
                block_type = pb.get("type", "other")
                index = pb.get("index", 0)

                block_json_path = f"$.pdf_info[{page_idx}].para_blocks[{pb_idx}]"

                lines = []
                for line_idx, line_data in enumerate(pb.get("lines", [])):
                    line_bbox = line_data.get("bbox", [0, 0, 0, 0])
                    line_json_path = f"{block_json_path}.lines[{line_idx}]"

                    spans = []
                    if include_spans:
                        for span_idx, span_data in enumerate(line_data.get("spans", [])):
                            span_bbox = span_data.get("bbox", [0, 0, 0, 0])
                            span = LayoutSpan(
                                bbox_x0=span_bbox[0],
                                bbox_y0=span_bbox[1],
                                bbox_x1=span_bbox[2],
                                bbox_y1=span_bbox[3],
                                span_type=span_data.get("type", "text"),
                                content=span_data.get("content", ""),
                                json_path=f"{line_json_path}.spans[{span_idx}]",
                            )
                            spans.append(span)

                    line = LayoutLine(
                        bbox_x0=line_bbox[0],
                        bbox_y0=line_bbox[1],
                        bbox_x1=line_bbox[2],
                        bbox_y1=line_bbox[3],
                        spans=spans,
                        json_path=line_json_path,
                    )
                    lines.append(line)

                block = LayoutBlock(
                    bbox_x0=bbox[0],
                    bbox_y0=bbox[1],
                    bbox_x1=bbox[2],
                    bbox_y1=bbox[3],
                    block_type=block_type,
                    page_index=page_idx,
                    index=index,
                    lines=lines,
                    json_path=block_json_path,
                )
                blocks.append(block)

            result[page_idx] = blocks

        logger.info(f"Parsed {len(result)} pages with {sum(len(v) for v in result.values())} blocks")
        return result

    def _group_blocks_by_column(
        self,
        blocks: List[LayoutBlock],
        page_width: float,
    ) -> List[List[LayoutBlock]]:
        """Group blocks into columns using center-point clustering.

        Algorithm:
        1. Filter text blocks (exclude title)
        2. Calculate center x-coordinate for each block
        3. Cluster blocks by center x-coordinate
        4. Each cluster represents one column

        Args:
            blocks: List of layout blocks.
            page_width: Total width of the page.

        Returns:
            List of column groups (each group is a list of blocks).
        """
        if not blocks:
            return []

        # Filter text blocks only (exclude title)
        text_blocks = [b for b in blocks if b.block_type == "text"]
        if not text_blocks:
            return []

        # Calculate center x-coordinate for each block
        block_centers = []
        for block in text_blocks:
            center_x = (block.bbox_x0 + block.bbox_x1) / 2
            block_centers.append((block, center_x))

        # Sort by center x
        block_centers.sort(key=lambda x: x[1])

        # Cluster by center x-coordinate using simple distance-based clustering
        # Threshold: 80 points difference means different columns
        # This helps handle MinerU detection variations within same column
        cluster_threshold = 80.0
        clusters: List[List[Tuple[LayoutBlock, float]]] = []
        current_cluster: List[Tuple[LayoutBlock, float]] = []

        for block, center_x in block_centers:
            if not current_cluster:
                current_cluster.append((block, center_x))
                continue

            rep_block, rep_center = current_cluster[-1]
            center_diff = abs(center_x - rep_center)

            if center_diff < cluster_threshold:
                # Same column
                current_cluster.append((block, center_x))
            else:
                # New column
                clusters.append(current_cluster)
                current_cluster = [(block, center_x)]

        if current_cluster:
            clusters.append(current_cluster)

        # Convert back to block lists
        columns: List[List[LayoutBlock]] = []
        for cluster in clusters:
            column_blocks = [block for block, _ in cluster]
            columns.append(column_blocks)

        logger.debug(f"Grouped {len(text_blocks)} text blocks into {len(columns)} columns")
        return columns

    def _compute_column_reference_bounds(
        self,
        column_blocks: List[LayoutBlock],
    ) -> Tuple[float, float]:
        """Compute reference x boundaries for a column.

        Excludes outlier blocks (e.g., abstract with different width).

        Args:
            column_blocks: List of text blocks in the column.

        Returns:
            Tuple of (min_x0, max_x1) for the column.
        """
        if not column_blocks:
            return 0, 0

        # Get all widths
        widths = [b.bbox_x1 - b.bbox_x0 for b in column_blocks]
        widths.sort()

        # Use median width as reference to identify outliers
        median_width = widths[len(widths) // 2] if widths else 0

        # Filter out blocks with significantly different width (> threshold difference from median)
        reference_blocks = []
        for block in column_blocks:
            width = block.bbox_x1 - block.bbox_x0
            if median_width > 0 and abs(width - median_width) / median_width < self.outlier_width_threshold:
                reference_blocks.append(block)

        # If we have too few reference blocks, use all
        if len(reference_blocks) < 2:
            reference_blocks = column_blocks

        min_x0 = min(b.bbox_x0 for b in reference_blocks)
        max_x1 = max(b.bbox_x1 for b in reference_blocks)

        return min_x0, max_x1

    def _align_intra_paragraph(self, block: LayoutBlock) -> LayoutBlock:
        """Align all lines within a paragraph to the paragraph's bbox edges.

        First and last lines preserve their original indentation:
        - First line: preserves left indentation (space before first span)
        - Last line: preserves right indentation (space after last span)
        - Middle lines: aligned to paragraph edges

        Args:
            block: The layout block to align.

        Returns:
            Aligned layout block.
        """
        if not block.lines or block.block_type != "text":
            return block

        para_x0 = block.bbox_x0
        para_x1 = block.bbox_x1
        para_width = para_x1 - para_x0

        num_lines = len(block.lines)
        new_lines = []

        for line_idx, line in enumerate(block.lines):
            line_old_width = line.bbox_x1 - line.bbox_x0
            if line_old_width <= 0:
                # Invalid line, keep as is
                new_lines.append(line)
                continue

            is_first_line = (line_idx == 0)
            is_last_line = (line_idx == num_lines - 1)

            # Calculate original left/right spacing based on spans
            # First span's left edge relative to line start
            first_span_x0 = line.spans[0].bbox_x0 if line.spans else line.bbox_x0
            # Last span's right edge relative to line end
            last_span_x1 = line.spans[-1].bbox_x1 if line.spans else line.bbox_x1

            left_space = first_span_x0 - line.bbox_x0  # Space before first span
            right_space = line.bbox_x1 - last_span_x1  # Space after last span

            # For first line, preserve left spacing
            # For last line, preserve right spacing
            # For middle lines, use 0 spacing
            if is_first_line:
                new_x0 = line.bbox_x0  # Preserve original left position
                new_x1 = para_x1  # Align right edge to paragraph
            elif is_last_line:
                new_x0 = para_x0  # Align left edge to paragraph
                new_x1 = line.bbox_x1  # Preserve original right position
            else:
                new_x0 = para_x0
                new_x1 = para_x1

            new_line_width = new_x1 - new_x0

            # Adjust spans proportionally
            new_spans = []
            for span in line.spans:
                span_old_width = span.bbox_x1 - span.bbox_x0
                if span_old_width <= 0:
                    new_spans.append(span)
                    continue

                # Calculate relative position ratio within original line
                span_ratio_start = (span.bbox_x0 - line.bbox_x0) / line_old_width
                span_ratio_end = (span.bbox_x1 - line.bbox_x0) / line_old_width

                # Map to new line positions
                new_span_x0 = new_x0 + span_ratio_start * new_line_width
                new_span_x1 = new_x0 + span_ratio_end * new_line_width

                new_spans.append(LayoutSpan(
                    bbox_x0=new_span_x0,
                    bbox_y0=span.bbox_y0,
                    bbox_x1=new_span_x1,
                    bbox_y1=span.bbox_y1,
                    span_type=span.span_type,
                    content=span.content,
                    json_path=span.json_path,
                ))

            new_lines.append(LayoutLine(
                bbox_x0=new_x0,
                bbox_y0=line.bbox_y0,
                bbox_x1=new_x1,
                bbox_y1=line.bbox_y1,
                spans=new_spans,
                json_path=line.json_path,
            ))

        return LayoutBlock(
            bbox_x0=block.bbox_x0,
            bbox_y0=block.bbox_y0,
            bbox_x1=block.bbox_x1,
            bbox_y1=block.bbox_y1,
            block_type=block.block_type,
            page_index=block.page_index,
            index=block.index,
            lines=new_lines,
            json_path=block.json_path,
        )

    def _align_inter_paragraph(self, column_blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """Align paragraphs within the same column.

        All paragraphs in a column will have their bbox x0 and x1 aligned
        to the reference boundaries (computed excluding outliers like abstract).

        First and last lines preserve their original indentation.

        Args:
            column_blocks: List of layout blocks in the same column.

        Returns:
            List of aligned layout blocks.
        """
        if not column_blocks:
            return column_blocks

        # Only process text blocks (exclude title)
        text_blocks = [b for b in column_blocks if b.block_type == "text"]
        if not text_blocks:
            return column_blocks

        # Compute reference boundaries (excluding outliers)
        min_x0, max_x1 = self._compute_column_reference_bounds(text_blocks)
        common_width = max_x1 - min_x0

        logger.debug(f"Aligning column: x0={min_x0}, x1={max_x1}, count={len(text_blocks)}")

        new_blocks = []
        for block in column_blocks:
            if block.block_type != "text":
                new_blocks.append(block)
                continue

            old_width = block.bbox_x1 - block.bbox_x0
            if old_width <= 0:
                new_blocks.append(block)
                continue

            # Check if this block is an outlier (width significantly different)
            block_width = block.bbox_x1 - block.bbox_x0
            median_width = common_width
            if median_width > 0:
                width_diff_ratio = abs(block_width - median_width) / median_width
                if width_diff_ratio >= self.outlier_width_threshold:
                    # Outlier block - don't align, keep original width
                    # But still apply intra-paragraph alignment
                    aligned_block = self._align_intra_paragraph(block)
                    new_blocks.append(aligned_block)
                    logger.debug(f"Keeping outlier block at y={block.bbox_y0} with width={block_width}")
                    continue

            # Scale width to common column width
            width_ratio = common_width / old_width

            # Compute reference left/right positions
            ref_x0 = min_x0
            ref_x1 = max_x1

            num_lines = len(block.lines)
            new_lines = []
            for line_idx, line in enumerate(block.lines):
                line_old_width = line.bbox_x1 - line.bbox_x0
                if line_old_width <= 0:
                    new_lines.append(line)
                    continue

                is_first_line = (line_idx == 0)
                is_last_line = (line_idx == num_lines - 1)

                # Calculate original left/right spacing based on spans
                # First span's left edge relative to line start
                first_span_x0 = line.spans[0].bbox_x0 if line.spans else line.bbox_x0
                # Last span's right edge relative to line end
                last_span_x1 = line.spans[-1].bbox_x1 if line.spans else line.bbox_x1

                # For first line, preserve left spacing (space before first span)
                # For last line, preserve right spacing (space after last span)
                # For middle lines, stretch to column boundaries
                if is_first_line:
                    # First line: preserve left position, stretch right to column
                    new_line_x0 = line.bbox_x0
                    new_line_x1 = ref_x1
                elif is_last_line:
                    # Last line: stretch left to column, preserve right position
                    new_line_x0 = ref_x0
                    new_line_x1 = line.bbox_x1
                else:
                    # Middle lines: full stretch to column boundaries
                    new_line_x0 = ref_x0 + (line.bbox_x0 - block.bbox_x0) * width_ratio
                    new_line_x1 = new_line_x0 + line_old_width * width_ratio

                new_line_width = new_line_x1 - new_line_x0

                # Adjust spans proportionally
                new_spans = []
                for span in line.spans:
                    span_old_width = span.bbox_x1 - span.bbox_x0
                    if span_old_width <= 0:
                        new_spans.append(span)
                        continue

                    span_ratio_start = (span.bbox_x0 - line.bbox_x0) / line_old_width
                    span_ratio_end = (span.bbox_x1 - line.bbox_x0) / line_old_width

                    new_span_x0 = new_line_x0 + span_ratio_start * new_line_width
                    new_span_x1 = new_line_x0 + span_ratio_end * new_line_width

                    new_spans.append(LayoutSpan(
                        bbox_x0=new_span_x0,
                        bbox_y0=span.bbox_y0,
                        bbox_x1=new_span_x1,
                        bbox_y1=span.bbox_y1,
                        span_type=span.span_type,
                        content=span.content,
                        json_path=span.json_path,
                    ))

                new_lines.append(LayoutLine(
                    bbox_x0=new_line_x0,
                    bbox_y0=line.bbox_y0,
                    bbox_x1=new_line_x1,
                    bbox_y1=line.bbox_y1,
                    spans=new_spans,
                    json_path=line.json_path,
                ))

            new_blocks.append(LayoutBlock(
                bbox_x0=ref_x0,
                bbox_y0=block.bbox_y0,
                bbox_x1=ref_x1,
                bbox_y1=block.bbox_y1,
                block_type=block.block_type,
                page_index=block.page_index,
                index=block.index,
                lines=new_lines,
                json_path=block.json_path,
            ))

        return new_blocks

    def align(
        self,
        blocks_by_page: Dict[int, List[LayoutBlock]],
        page_width: float = 595.0,
    ) -> Dict[int, List[LayoutBlock]]:
        """Align bboxes for all pages.

        Args:
            blocks_by_page: Dictionary mapping page index to list of LayoutBlocks.
            page_width: Total width of the page (for column detection).

        Returns:
            Dictionary with aligned LayoutBlocks.
        """
        aligned_pages: Dict[int, List[LayoutBlock]] = {}

        for page_idx, blocks in blocks_by_page.items():
            logger.debug(f"Aligning page {page_idx} with {len(blocks)} blocks")

            # Step 1: Group blocks into columns
            columns = self._group_blocks_by_column(blocks, page_width)
            logger.debug(f"Detected {len(columns)} columns on page {page_idx}")

            # Step 2: Process each column
            all_aligned_blocks = []
            for col_idx, column_blocks in enumerate(columns):
                logger.debug(f"Processing column {col_idx} with {len(column_blocks)} blocks")

                # First: Intra-paragraph alignment (align lines within each paragraph)
                if self.intra_paragraph_align:
                    aligned_col_blocks = [
                        self._align_intra_paragraph(b) for b in column_blocks
                    ]
                else:
                    aligned_col_blocks = column_blocks

                # Second: Inter-paragraph alignment (align paragraphs within the column)
                if self.inter_paragraph_align and len(aligned_col_blocks) > 1:
                    aligned_col_blocks = self._align_inter_paragraph(aligned_col_blocks)

                all_aligned_blocks.extend(aligned_col_blocks)

            # Add non-text blocks (keep original) - title and other types
            non_text_blocks = [
                b for b in blocks if b.block_type != "text"
            ]
            all_aligned_blocks.extend(non_text_blocks)

            # Sort by y position (top to bottom)
            all_aligned_blocks.sort(key=lambda b: b.bbox_y0)

            aligned_pages[page_idx] = all_aligned_blocks

        return aligned_pages

    def export_layout_data(
        self,
        aligned_blocks: Dict[int, List[LayoutBlock]],
        original_layout_data: dict,
    ) -> dict:
        """Export aligned layout data back to original format.

        Args:
            aligned_blocks: Dictionary of aligned LayoutBlocks by page.
            original_layout_data: Original layout data structure.

        Returns:
            Layout data with updated bbox values.
        """
        import copy
        result = copy.deepcopy(original_layout_data)

        for page_idx, page_data in enumerate(result.get("pdf_info", [])):
            aligned_page_blocks = aligned_blocks.get(page_idx, [])

            # Build a map of json_path to aligned block
            block_map = {}
            for block in aligned_page_blocks:
                block_map[block.json_path] = block

            # Update para_blocks
            for pb_idx, pb in enumerate(page_data.get("para_blocks", [])):
                json_path = f"$.pdf_info[{page_idx}].para_blocks[{pb_idx}]"

                if json_path in block_map:
                    aligned_block = block_map[json_path]

                    # Update block bbox
                    pb["bbox"] = [
                        aligned_block.bbox_x0,
                        aligned_block.bbox_y0,
                        aligned_block.bbox_x1,
                        aligned_block.bbox_y1,
                    ]

                    # Update lines
                    for line_idx, line in enumerate(pb.get("lines", [])):
                        line_json_path = f"{json_path}.lines[{line_idx}]"

                        # Find matching line in aligned block
                        aligned_line = None
                        for al in aligned_block.lines:
                            if al.json_path == line_json_path:
                                aligned_line = al
                                break

                        if aligned_line:
                            line["bbox"] = [
                                aligned_line.bbox_x0,
                                aligned_line.bbox_y0,
                                aligned_line.bbox_x1,
                                aligned_line.bbox_y1,
                            ]

                            # Update spans
                            for span_idx, span in enumerate(line.get("spans", [])):
                                span_json_path = f"{line_json_path}.spans[{span_idx}]"

                                aligned_span = None
                                for als in aligned_line.spans:
                                    if als.json_path == span_json_path:
                                        aligned_span = als
                                        break

                                if aligned_span:
                                    span["bbox"] = [
                                        aligned_span.bbox_x0,
                                        aligned_span.bbox_y0,
                                        aligned_span.bbox_x1,
                                        aligned_span.bbox_y1,
                                    ]

        return result


class BBoxAlignerManager:
    """High-level manager for bbox alignment operations.

    Provides simplified interface for aligning layout data from files.
    """

    def __init__(
        self,
        intra_paragraph_align: bool = True,
        inter_paragraph_align: bool = True,
    ):
        """Initialize the BBoxAlignerManager.

        Args:
            intra_paragraph_align: Whether to align text within paragraphs.
            inter_paragraph_align: Whether to align paragraphs in the same column.
        """
        self.aligner = BBoxAligner(
            intra_paragraph_align=intra_paragraph_align,
            inter_paragraph_align=inter_paragraph_align,
        )

    def align_from_data(
        self,
        layout_data: dict,
        page_width: float = 595.0,
    ) -> dict:
        """Align bboxes from layout data.

        Args:
            layout_data: The parsed JSON data from MinerU's layout.json.
            page_width: Total width of the page.

        Returns:
            Aligned layout data.
        """
        # Parse
        blocks = self.aligner.parse_layout_data(layout_data)

        # Align
        aligned_blocks = self.aligner.align(blocks, page_width=page_width)

        # Export
        return self.aligner.export_layout_data(aligned_blocks, layout_data)