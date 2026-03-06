"""Tests for bbox alignment service."""

import json
from pathlib import Path

import pytest
from pypdf import PdfReader
from loguru import logger

from core.config import config
from services.render.align import (
    BBoxAligner,
    BBoxAlignerManager,
    LayoutBlock,
    LayoutLine,
    LayoutSpan,
)
from services.render.overlay import OverlayManager


# Test asset paths
TEST_ASSETS_DIR = config.test.test_assets_dir
LAYOUT_JSON_PATH = TEST_ASSETS_DIR / "extracted" / "layout.json"
SOURCE_PDF_PATH = TEST_ASSETS_DIR / "source" / "test_example.pdf"
OUTPUT_DIR = Path(__file__).parent / "output"


# Test fixtures
@pytest.fixture
def layout_data():
    """Load real layout data from JSON file."""
    with open(LAYOUT_JSON_PATH) as f:
        return json.load(f)


@pytest.fixture
def pdf_path():
    """Path to test PDF file."""
    return SOURCE_PDF_PATH


@pytest.fixture
def output_dir():
    """Create and return output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
@pytest.fixture
def sample_layout_data():
    """Sample layout data for testing."""
    return {
        "pdf_info": [
            {
                "page_idx": 0,
                "width": 595.0,
                "height": 842.0,
                "para_blocks": [
                    {
                        "bbox": [50, 100, 280, 150],
                        "type": "text",
                        "index": 0,
                        "lines": [
                            {
                                "bbox": [50, 100, 200, 115],
                                "spans": [
                                    {
                                        "bbox": [50, 100, 150, 115],
                                        "type": "text",
                                        "content": "This is a test paragraph.",
                                    },
                                ],
                            },
                            {
                                "bbox": [55, 115, 240, 130],
                                "spans": [
                                    {
                                        "bbox": [55, 115, 240, 130],
                                        "type": "text",
                                        "content": "The second line is slightly shorter.",
                                    },
                                ],
                            },
                            {
                                "bbox": [60, 130, 280, 150],
                                "spans": [
                                    {
                                        "bbox": [60, 130, 180, 150],
                                        "type": "text",
                                        "content": "And this is the last line.",
                                    },
                                ],
                            },
                        ],
                    },
                    {
                        "bbox": [50, 160, 280, 220],
                        "type": "text",
                        "index": 1,
                        "lines": [
                            {
                                "bbox": [50, 160, 180, 175],
                                "spans": [
                                    {
                                        "bbox": [50, 160, 180, 175],
                                        "type": "text",
                                        "content": "Another paragraph here.",
                                    },
                                ],
                            },
                            {
                                "bbox": [52, 175, 250, 190],
                                "spans": [
                                    {
                                        "bbox": [52, 175, 250, 190],
                                        "type": "text",
                                        "content": "With two lines of text.",
                                    },
                                ],
                            },
                        ],
                    },
                ],
            }
        ]
    }


class TestBBoxAligner:
    """Tests for BBoxAligner class."""

    def test_init(self):
        """Test aligner initialization."""
        aligner = BBoxAligner(
            intra_paragraph_align=True,
            inter_paragraph_align=True,
        )
        assert aligner.intra_paragraph_align is True
        assert aligner.inter_paragraph_align is True

    def test_parse_layout_data(self, sample_layout_data):
        """Test parsing layout data."""
        aligner = BBoxAligner()
        blocks = aligner.parse_layout_data(sample_layout_data)

        assert 0 in blocks
        assert len(blocks[0]) == 2

        # Check first block
        first_block = blocks[0][0]
        assert first_block.bbox_x0 == 50
        assert first_block.bbox_x1 == 280
        assert first_block.block_type == "text"
        assert len(first_block.lines) == 3

    def test_align_intra_paragraph(self, sample_layout_data):
        """Test intra-paragraph alignment."""
        aligner = BBoxAligner(intra_paragraph_align=True, inter_paragraph_align=False)
        blocks = aligner.parse_layout_data(sample_layout_data)

        # First block
        first_block = blocks[0][0]
        aligned_block = aligner._align_intra_paragraph(first_block)

        # First line should start at paragraph left edge
        assert aligned_block.lines[0].bbox_x0 == first_block.bbox_x0

        # Last line should end at paragraph right edge
        assert aligned_block.lines[-1].bbox_x1 == first_block.bbox_x1

    def test_align_inter_paragraph(self, sample_layout_data):
        """Test inter-paragraph alignment."""
        aligner = BBoxAligner(intra_paragraph_align=False, inter_paragraph_align=True)
        blocks = aligner.parse_layout_data(sample_layout_data)

        # Get text blocks only
        text_blocks = [b for b in blocks[0] if b.block_type == "text"]

        # Align inter-paragraph
        aligned_blocks = aligner._align_inter_paragraph(text_blocks)

        # All blocks should have same x boundaries
        min_x0 = min(b.bbox_x0 for b in aligned_blocks)
        max_x1 = max(b.bbox_x1 for b in aligned_blocks)

        for block in aligned_blocks:
            assert block.bbox_x0 == min_x0
            assert block.bbox_x1 == max_x1

    def test_full_align(self, sample_layout_data):
        """Test full alignment pipeline."""
        aligner = BBoxAligner(
            intra_paragraph_align=True,
            inter_paragraph_align=True,
        )
        blocks = aligner.parse_layout_data(sample_layout_data)
        aligned_blocks = aligner.align(blocks, page_width=595.0)

        assert 0 in aligned_blocks
        assert len(aligned_blocks[0]) > 0

    def test_export_layout_data(self, sample_layout_data):
        """Test exporting aligned data back to original format."""
        aligner = BBoxAligner(
            intra_paragraph_align=True,
            inter_paragraph_align=True,
        )

        blocks = aligner.parse_layout_data(sample_layout_data)
        aligned_blocks = aligner.align(blocks, page_width=595.0)
        result = aligner.export_layout_data(aligned_blocks, sample_layout_data)

        # Check structure is preserved
        assert "pdf_info" in result
        assert len(result["pdf_info"]) == 1

        page_data = result["pdf_info"][0]
        assert "para_blocks" in page_data
        assert len(page_data["para_blocks"]) == 2


class TestBBoxAlignerManager:
    """Tests for BBoxAlignerManager class."""

    def test_init(self):
        """Test manager initialization."""
        manager = BBoxAlignerManager(
            intra_paragraph_align=True,
            inter_paragraph_align=True,
        )
        assert manager.aligner.intra_paragraph_align is True
        assert manager.aligner.inter_paragraph_align is True

    def test_align_from_data(self, sample_layout_data):
        """Test align_from_data method."""
        manager = BBoxAlignerManager()
        result = manager.align_from_data(sample_layout_data, page_width=595.0)

        # Check result is valid
        assert "pdf_info" in result
        assert len(result["pdf_info"]) == 1

        # Check bboxes were modified (x values should be aligned)
        para_blocks = result["pdf_info"][0]["para_blocks"]
        # All text blocks should have same x boundaries after alignment
        x0_values = [pb["bbox"][0] for pb in para_blocks]
        x1_values = [pb["bbox"][2] for pb in para_blocks]

        # All start positions should be equal
        assert len(set(x0_values)) == 1
        # All end positions should be equal
        assert len(set(x1_values)) == 1


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_blocks(self):
        """Test with empty block list."""
        aligner = BBoxAligner()
        result = aligner.align({}, page_width=595.0)
        assert result == {}

    def test_non_text_blocks(self):
        """Test that non-text blocks are preserved."""
        data = {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "width": 595.0,
                    "height": 842.0,
                    "para_blocks": [
                        {
                            "bbox": [50, 100, 200, 150],
                            "type": "image",  # Non-text block
                            "index": 0,
                            "lines": [],
                        },
                        {
                            "bbox": [50, 160, 280, 200],
                            "type": "text",
                            "index": 1,
                            "lines": [
                                {
                                    "bbox": [50, 160, 200, 175],
                                    "spans": [
                                        {
                                            "bbox": [50, 160, 200, 175],
                                            "type": "text",
                                            "content": "Text block",
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                }
            ]
        }

        aligner = BBoxAlignerManager()
        result = aligner.align_from_data(data, page_width=595.0)

        # Image block should be preserved
        assert result["pdf_info"][0]["para_blocks"][0]["type"] == "image"


class TestBBoxAlignerVisualization:
    """Tests for visualizing aligned layout with real data."""

    @pytest.fixture
    def sample_layout_data(self):
        """Sample layout data for testing."""
        return {
            "pdf_info": [
                {
                    "page_idx": 0,
                    "width": 595.0,
                    "height": 842.0,
                    "para_blocks": [
                        {
                            "bbox": [50, 100, 280, 150],
                            "type": "text",
                            "index": 0,
                            "lines": [
                                {
                                    "bbox": [50, 100, 200, 115],
                                    "spans": [
                                        {
                                            "bbox": [50, 100, 150, 115],
                                            "type": "text",
                                            "content": "This is a test paragraph.",
                                        },
                                    ],
                                },
                                {
                                    "bbox": [55, 115, 240, 130],
                                    "spans": [
                                        {
                                            "bbox": [55, 115, 240, 130],
                                            "type": "text",
                                            "content": "The second line is slightly shorter.",
                                        },
                                    ],
                                },
                                {
                                    "bbox": [60, 130, 280, 150],
                                    "spans": [
                                        {
                                            "bbox": [60, 130, 180, 150],
                                            "type": "text",
                                            "content": "And this is the last line.",
                                        },
                                    ],
                                },
                            ],
                        },
                        {
                            "bbox": [50, 160, 280, 220],
                            "type": "text",
                            "index": 1,
                            "lines": [
                                {
                                    "bbox": [50, 160, 180, 175],
                                    "spans": [
                                        {
                                            "bbox": [50, 160, 180, 175],
                                            "type": "text",
                                            "content": "Another paragraph here.",
                                        },
                                    ],
                                },
                                {
                                    "bbox": [52, 175, 250, 190],
                                    "spans": [
                                        {
                                            "bbox": [52, 175, 250, 190],
                                            "type": "text",
                                            "content": "With two lines of text.",
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                }
            ]
        }

    @pytest.mark.asyncio
    async def test_align_and_overlay_visualization(self, layout_data, pdf_path, output_dir):
        """Test align and overlay visualization with real layout data.

        This test demonstrates align functionality by:
        1. Loading real layout data from tests/assets
        2. Running align to correct bboxes
        3. Rendering overlay PDF to visualize the aligned result
        """
        # Step 1: Align the layout data
        aligner = BBoxAlignerManager()
        aligned_data = aligner.align_from_data(layout_data, page_width=595.0)

        # Step 2: Generate overlay PDF with aligned data
        output_path = output_dir / "test_align_visualization.pdf"

        overlay_manager = OverlayManager()
        result_path = await overlay_manager.render_overlay(
            layout_data=aligned_data,
            pdf_path=pdf_path,
            output_path=output_path,
            include_lines=True,
        )

        # Verify output
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify PDF is valid
        reader = PdfReader(output_path)
        assert len(reader.pages) > 0

        logger.info(f"Aligned visualization saved to: {output_path}")