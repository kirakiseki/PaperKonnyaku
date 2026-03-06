"""Tests for overlay rendering service."""

import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from core.config import config
from services.render.overlay import (
    OverlayManager,
    OverlayRenderer,
    BoundingBox,
    LayoutBlock,
    LayoutLine,
    LayoutSpan,
    BlockType,
)


# Test asset paths
TEST_ASSETS_DIR = config.test.test_assets_dir
LAYOUT_JSON_PATH = TEST_ASSETS_DIR / "extracted" / "layout.json"
SOURCE_PDF_PATH = TEST_ASSETS_DIR / "source" / "test_example.pdf"
OUTPUT_DIR = Path(__file__).parent / "output"


@pytest.fixture
def layout_data():
    """Load layout data from JSON file."""
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
def overlay_renderer():
    """Create an OverlayRenderer instance."""
    return OverlayRenderer()


@pytest.fixture
def overlay_manager():
    """Create an OverlayManager instance."""
    return OverlayManager()


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_from_list(self):
        """Test creating BoundingBox from list."""
        coords = [100, 200, 300, 400]
        bbox = BoundingBox.from_list(coords)
        assert bbox.x0 == 100
        assert bbox.y0 == 200
        assert bbox.x1 == 300
        assert bbox.y1 == 400

    def test_to_list(self):
        """Test converting BoundingBox to list."""
        bbox = BoundingBox(x0=100, y0=200, x1=300, y1=400)
        result = bbox.to_list()
        assert result == [100, 200, 300, 400]


class TestOverlayRenderer:
    """Tests for OverlayRenderer class."""

    def test_init(self):
        """Test renderer initialization."""
        renderer = OverlayRenderer(border_width=2.0, alpha=0.5)
        assert renderer.border_width == 2.0
        assert renderer.alpha == 0.5

    def test_get_color(self):
        """Test color retrieval for block types."""
        renderer = OverlayRenderer()
        assert renderer._get_color("title") == (1.0, 0.0, 0.0)
        assert renderer._get_color("text") == (0.0, 0.0, 1.0)
        assert renderer._get_color("image") == (0.0, 1.0, 0.0)
        assert renderer._get_color("unknown_type") == (0.5, 0.5, 0.5)

    def test_parse_layout_data(self, layout_data):
        """Test parsing layout data into LayoutBlock objects."""
        renderer = OverlayRenderer()
        blocks_by_page = renderer.parse_layout_data(layout_data)

        # Check we have pages
        assert len(blocks_by_page) > 0

        # Check first page has blocks
        page_0_blocks = blocks_by_page.get(0, [])
        assert len(page_0_blocks) > 0

        # Check block structure
        first_block = page_0_blocks[0]
        assert isinstance(first_block, LayoutBlock)
        assert isinstance(first_block.bbox, BoundingBox)
        assert first_block.block_type in BlockType

    def test_convert_bbox_to_pdf_coords(self):
        """Test coordinate conversion from MinerU to PDF."""
        renderer = OverlayRenderer()
        bbox = BoundingBox(x0=100, y0=100, x1=200, y1=200)
        page_height = 841  # A4 page height in points

        x0, y0, x1, y1 = renderer._convert_bbox_to_pdf_coords(bbox, page_height)

        # x coordinates should remain the same
        assert x0 == 100
        assert x1 == 200
        # y coordinates should be flipped
        # y0 in PDF = page_height - bbox.y1 = 841 - 200 = 641
        # y1 in PDF = page_height - bbox.y0 = 841 - 100 = 741
        assert y0 == 641
        assert y1 == 741

    @pytest.mark.asyncio
    async def test_render_overlay(self, layout_data, pdf_path, output_dir):
        """Test rendering overlay on PDF."""
        renderer = OverlayRenderer()
        output_path = output_dir / "test_overlay_output.pdf"

        result = await renderer.render_overlay(
            layout_data=layout_data,
            pdf_path=pdf_path,
            output_path=output_path,
            include_lines=True,
        )

        # Check output file was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Check PDF is valid
        reader = PdfReader(output_path)
        assert len(reader.pages) > 0

    @pytest.mark.asyncio
    async def test_render_overlay_without_lines(self, layout_data, pdf_path, output_dir):
        """Test rendering overlay without line-level boxes."""
        renderer = OverlayRenderer()
        output_path = output_dir / "test_overlay_no_lines.pdf"

        result = await renderer.render_overlay(
            layout_data=layout_data,
            pdf_path=pdf_path,
            output_path=output_path,
            include_lines=False,
        )

        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_render_overlay_pdf_not_found(self, layout_data, output_dir):
        """Test error handling when PDF not found."""
        renderer = OverlayRenderer()
        output_path = output_dir / "test_nonexistent.pdf"

        with pytest.raises(FileNotFoundError):
            await renderer.render_overlay(
                layout_data=layout_data,
                pdf_path="/nonexistent/path.pdf",
                output_path=output_path,
            )


class TestOverlayManager:
    """Tests for OverlayManager class."""

    def test_init(self):
        """Test manager initialization."""
        manager = OverlayManager(border_width=3.0)
        assert manager.renderer.border_width == 3.0

    @pytest.mark.asyncio
    async def test_from_files(self, output_dir):
        """Test creating overlay from files."""
        output_path = output_dir / "test_manager_output.pdf"

        result = await OverlayManager.from_files(
            layout_json_path=LAYOUT_JSON_PATH,
            pdf_path=SOURCE_PDF_PATH,
            output_path=output_path,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_from_files_json_not_found(self, output_dir):
        """Test error when layout JSON not found."""
        output_path = output_dir / "test_error.pdf"

        with pytest.raises(FileNotFoundError):
            await OverlayManager.from_files(
                layout_json_path="/nonexistent/layout.json",
                pdf_path=SOURCE_PDF_PATH,
                output_path=output_path,
            )

    @pytest.mark.asyncio
    async def test_render_overlay(self, layout_data, pdf_path, output_dir):
        """Test render_overlay method."""
        manager = OverlayManager()
        output_path = output_dir / "test_manager_render.pdf"

        result = await manager.render_overlay(
            layout_data=layout_data,
            pdf_path=pdf_path,
            output_path=output_path,
        )

        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_render_from_dict(self, layout_data, pdf_path, output_dir):
        """Test render_from_dict method."""
        manager = OverlayManager()
        output_dir_path = output_dir / "dict_output"

        result = await manager.render_from_dict(
            layout_data=layout_data,
            pdf_path=pdf_path,
            output_dir=output_dir_path,
        )

        assert result.exists()
        assert result.name == "test_example_overlay.pdf"