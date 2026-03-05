"""Tests for MinerU service."""

import pytest

from core.config import config
from services.mineru import MinerUService, TaskHistory


@pytest.fixture
def test_file():
    """Path to test PDF file from config."""
    return config.test.test_file


@pytest.fixture
async def mineru_service():
    """Create a MinerU service instance."""
    async with MinerUService() as mineru:
        yield mineru


@pytest.mark.asyncio
async def test_upload(mineru_service, test_file):
    """Test file upload returns batch_id."""
    batch_id = await mineru_service.upload(test_file)
    assert batch_id is not None
    assert isinstance(batch_id, str)


@pytest.mark.asyncio
async def test_get_result(mineru_service, test_file):
    """Test getting result from a batch."""
    # Upload first
    batch_id = await mineru_service.upload(test_file)

    # Get result
    result = await mineru_service.get_result(batch_id)

    assert result is not None
    assert result.file_name == "test_example.pdf"
    assert result.state in ["pending", "running", "done", "failed", "waiting-file", "converting"]


@pytest.mark.asyncio
async def test_upload_and_wait(mineru_service, test_file):
    """Test upload and wait for completion."""
    result = await mineru_service.upload_and_wait(test_file, timeout=600)

    assert result is not None
    assert result.file_name == "test_example.pdf"
    assert result.state == "done"
    assert result.full_zip_url is not None


@pytest.mark.asyncio
async def test_process_full_pipeline(mineru_service, test_file):
    """Test complete pipeline: upload, wait, download, extract."""
    result = await mineru_service.process(test_file, timeout=600)

    # Check basic result
    assert result is not None
    assert result.file_name == "test_example.pdf"
    assert result.state == "done"
    assert result.task_id is not None

    # Check source file was copied
    assert result.source_file.exists()
    assert result.source_file.name == "test_example.pdf"

    # Check zip file was downloaded
    assert result.zip_file is not None
    assert result.zip_file.exists()
    assert result.zip_file.suffix == ".zip"

    # Check extraction
    assert result.extracted_dir is not None
    assert result.extracted_dir.exists()

    # Check extracted files
    extracted_files = list(result.extracted_dir.iterdir())
    assert len(extracted_files) > 0

    print(f"Task ID: {result.task_id}")
    print(f"Source file: {result.source_file}")
    print(f"Zip file: {result.zip_file}")
    print(f"Extracted dir: {result.extracted_dir}")
    print(f"Extracted files: {extracted_files}")


@pytest.mark.asyncio
async def test_file_not_found():
    """Test error when file does not exist."""
    async with MinerUService() as mineru:
        with pytest.raises(ValueError, match="File does not exist"):
            await mineru.upload("/nonexistent/file.pdf")


@pytest.mark.asyncio
async def test_service_with_custom_token():
    """Test service initialization with custom token."""
    async with MinerUService(api_token="test_token") as mineru:
        assert mineru.api_token == "test_token"


@pytest.mark.asyncio
async def test_task_history():
    """Test task history saving and loading."""
    history = TaskHistory()

    # Save a test task
    task_info = {
        "task_id": "test_task_001",
        "source_file": "/path/to/test.pdf",
        "status": "completed",
        "full_zip_url": "https://example.com/result.zip",
    }

    task_file = history.save_task(task_info)
    assert task_file.exists()

    # Load the task
    loaded = history.load_task("test_task_001")
    assert loaded is not None
    assert loaded["task_id"] == "test_task_001"
    assert loaded["source_file"] == "/path/to/test.pdf"

    # List all tasks
    tasks = history.list_tasks()
    assert len(tasks) > 0