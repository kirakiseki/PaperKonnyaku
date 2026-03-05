"""MinerU service module.

This module provides a client for interacting with the MinerU document extraction API.
"""

import asyncio
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import httpx
from loguru import logger

from core.config import config, TaskHistory


@dataclass
class MinerUFileResult:
    """Result of a MinerU extraction task for a single file."""

    file_name: str
    state: str  # done, waiting-file, pending, running, failed, converting
    data_id: Optional[str] = None
    full_zip_url: Optional[str] = None
    err_msg: Optional[str] = None
    extract_progress: Optional[dict] = None


@dataclass
class MinerUExtractResult:
    """Complete extraction result including downloaded and extracted files."""

    file_name: str
    state: str
    source_file: Path  # Path to the uploaded source file
    zip_file: Optional[Path] = None  # Path to downloaded zip file
    extracted_dir: Optional[Path] = None  # Path to extracted directory
    full_zip_url: Optional[str] = None
    data_id: Optional[str] = None
    err_msg: Optional[str] = None
    task_id: Optional[str] = None  # Task ID for tracing


class MinerUService:
    """MinerU document extraction service client.

    Provides methods for uploading local files and retrieving extraction results
    from the MinerU API.

    Usage:
        from services.mineru import MinerUService

        async with MinerUService() as mineru:
            # Upload and wait for completion (polls until done)
            result = await mineru.upload_and_wait("/path/to/file.pdf")

            # Or upload, wait, download and extract
            extract_result = await mineru.process("/path/to/file.pdf")
    """

    def __init__(self, api_token: Optional[str] = None, task_history: Optional[TaskHistory] = None):
        """Initialize the MinerU service client.

        Args:
            api_token: API token for MinerU. If not provided, uses token from config.
            task_history: TaskHistory instance for recording task information.
        """
        mineru_config = config.extract.mineru
        self.api_token = api_token or mineru_config.api_token
        self.base_url = mineru_config.url
        self.poll_interval = mineru_config.poll_interval
        self.timeout = mineru_config.timeout
        self.output_dir = Path(mineru_config.output_dir)
        self.task_history = task_history or TaskHistory()
        self._client = httpx.AsyncClient(timeout=120.0)
        logger.debug("MinerUService initialized with API token")

    def _get_headers(self) -> dict:
        """Get request headers with authorization."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }

    def _generate_task_id(self, source_file: Path) -> str:
        """Generate a unique task ID based on source file name and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{source_file.stem}_{timestamp}"

    def _get_task_output_dir(self, source_file: Path) -> Path:
        """Get the output directory for a task."""
        task_id = self._generate_task_id(source_file)
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir

    def _save_task_info(self, task_id: str, task_info: dict) -> None:
        """Save task information to history."""
        full_info = {
            "task_id": task_id,
            "created_at": datetime.now().isoformat(),
            **task_info,
        }
        self.task_history.save_task(full_info)
        logger.info(f"Task info saved: {task_id}")

    async def upload(
        self,
        file_path: Union[str, Path],
        model_version: str = "vlm",
        is_ocr: Optional[bool] = None,
        page_ranges: Optional[str] = None,
        enable_formula: Optional[bool] = None,
        enable_table: Optional[bool] = None,
        language: Optional[str] = None,
        callback: Optional[str] = None,
        seed: Optional[str] = None,
        extra_formats: Optional[list] = None,
    ) -> str:
        """Upload a single local file and create an extraction task.

        Args:
            file_path: Path to the local file to upload.
            model_version: Model version to use (pipeline, vlm, MinerU-HTML). Default: vlm.
            is_ocr: Whether to enable OCR. Default: false.
            page_ranges: Page range to extract (e.g., "1-10" or "1,3-5").
            enable_formula: Whether to enable formula recognition. Default: true.
            enable_table: Whether to enable table recognition. Default: true.
            language: Document language code. Default: ch.
            callback: Callback URL for async notification.
            seed: Random string for callback signature.
            extra_formats: Additional export formats (docx, html, latex).

        Returns:
            str: The batch_id for the created extraction task.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")

        logger.info(f"Uploading file: {file_path.name}")

        # Generate task ID
        task_id = self._generate_task_id(file_path)

        # Prepare file data
        files_data = [{"name": file_path.name}]
        if is_ocr is not None:
            files_data[0]["is_ocr"] = is_ocr
        if page_ranges is not None:
            files_data[0]["page_ranges"] = page_ranges

        request_data = {
            "files": files_data,
            "model_version": model_version,
        }

        if enable_formula is not None:
            request_data["enable_formula"] = enable_formula
        if enable_table is not None:
            request_data["enable_table"] = enable_table
        if language is not None:
            request_data["language"] = language
        if callback is not None:
            request_data["callback"] = callback
        if seed is not None:
            request_data["seed"] = seed
        if extra_formats is not None:
            request_data["extra_formats"] = extra_formats

        logger.debug(f"Applying for upload URL with model_version: {model_version}")

        # Apply for upload URL
        response = await self._client.post(
            f"{self.base_url}/file-urls/batch",
            headers=self._get_headers(),
            json=request_data,
        )
        response.raise_for_status()

        result = response.json()
        if result.get("code") != 0:
            logger.error(f"MinerU API error: {result.get('msg', 'Unknown error')}")
            raise Exception(f"MinerU API error: {result.get('msg', 'Unknown error')}")

        batch_id = result["data"]["batch_id"]
        file_urls = result["data"]["file_urls"]

        if not file_urls:
            raise Exception("No upload URL returned from MinerU API")

        upload_url = file_urls[0]

        # Upload file
        logger.info(f"Uploading to: {upload_url}")
        with open(file_path, "rb") as f:
            upload_response = await self._client.put(upload_url, content=f.read())

        if upload_response.status_code != 200:
            logger.error(f"Failed to upload file: {upload_response.status_code}")
            raise Exception(f"Failed to upload file: {upload_response.status_code}")

        logger.info(f"File uploaded successfully, batch_id: {batch_id}")

        # Save task info
        self._save_task_info(task_id, {
            "batch_id": batch_id,
            "source_file": str(file_path),
            "source_file_name": file_path.name,
            "model_version": model_version,
            "status": "uploaded",
        })

        return batch_id

    async def get_result(self, batch_id: str) -> MinerUFileResult:
        """Get the result of a batch extraction task.

        Args:
            batch_id: The batch_id returned from upload.

        Returns:
            MinerUFileResult: The file extraction result.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        logger.debug(f"Fetching result for batch_id: {batch_id}")

        response = await self._client.get(
            f"{self.base_url}/extract-results/batch/{batch_id}",
            headers=self._get_headers(),
        )
        response.raise_for_status()

        result = response.json()
        if result.get("code") != 0:
            logger.error(f"MinerU API error: {result.get('msg', 'Unknown error')}")
            raise Exception(f"MinerU API error: {result.get('msg', 'Unknown error')}")

        data = result["data"]
        extract_result = data.get("extract_result", [])

        if not extract_result:
            raise Exception("No result found for batch_id")

        item = extract_result[0]
        file_result = MinerUFileResult(
            file_name=item.get("file_name"),
            state=item.get("state"),
            data_id=item.get("data_id"),
            full_zip_url=item.get("full_zip_url"),
            err_msg=item.get("err_msg"),
            extract_progress=item.get("extract_progress"),
        )

        logger.info(f"File: {file_result.file_name}, State: {file_result.state}")
        if file_result.state == "done":
            logger.info(f"Extraction complete, result URL: {file_result.full_zip_url}")
        elif file_result.state == "failed":
            logger.error(f"Extraction failed: {file_result.err_msg}")

        return file_result

    async def upload_and_wait(
        self,
        file_path: Union[str, Path],
        poll_interval: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> MinerUFileResult:
        """Upload a file and wait for extraction to complete.

        This is a convenience method that uploads the file and polls
        the result until the task is done or failed.

        Args:
            file_path: Path to the local file to upload.
            poll_interval: Interval between polls in seconds. Default: from config.
            timeout: Maximum time to wait in seconds. Default: from config.
            **kwargs: Additional arguments passed to upload().

        Returns:
            MinerUFileResult: The completed extraction result.

        Raises:
            TimeoutError: If the task does not complete within timeout.
        """
        poll_interval = poll_interval or self.poll_interval
        timeout = timeout or self.timeout

        batch_id = await self.upload(file_path, **kwargs)

        start_time = asyncio.get_event_loop().time()
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Extraction timed out after {timeout} seconds")

            result = await self.get_result(batch_id)

            if result.state == "done":
                return result
            elif result.state == "failed":
                logger.error(f"Extraction failed: {result.err_msg}")
                return result
            else:
                # pending, running, waiting-file, converting
                logger.debug(f"Task state: {result.state}, waiting...")
                await asyncio.sleep(poll_interval)

    async def download_zip(self, url: str, output_path: Path) -> Path:
        """Download the zip file from the given URL.

        Args:
            url: The URL to download from.
            output_path: The path to save the downloaded file.

        Returns:
            Path: The path to the downloaded file.
        """
        logger.info(f"Downloading zip from: {url}")

        response = await self._client.get(url)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded zip to: {output_path}")
        return output_path

    def extract_zip(self, zip_path: Path, output_dir: Path) -> Path:
        """Extract the zip file to the output directory.

        Args:
            zip_path: Path to the zip file.
            output_dir: Directory to extract to.

        Returns:
            Path: The path to the extracted directory.
        """
        logger.info(f"Extracting zip to: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        logger.info(f"Extracted to: {output_dir}")
        return output_dir

    async def process(
        self,
        file_path: Union[str, Path],
        poll_interval: Optional[float] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> MinerUExtractResult:
        """Upload, wait for extraction, download and extract the result.

        This is the complete pipeline that:
        1. Uploads the file to MinerU
        2. Waits for extraction to complete
        3. Downloads the result zip file
        4. Extracts the zip file

        Args:
            file_path: Path to the local file to process.
            poll_interval: Interval between polls in seconds. Default: from config.
            timeout: Maximum time to wait in seconds. Default: from config.
            **kwargs: Additional arguments passed to upload().

        Returns:
            MinerUExtractResult: Complete extraction result with file paths.
        """
        file_path = Path(file_path)

        # Generate task ID
        task_id = self._generate_task_id(file_path)

        # Create task output directory
        task_dir = self._get_task_output_dir(file_path)
        source_dir = task_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        # Copy source file to output directory
        source_file = source_dir / file_path.name
        shutil.copy2(file_path, source_file)
        logger.info(f"Copied source file to: {source_file}")

        # Upload and wait for extraction
        result = await self.upload_and_wait(
            file_path,
            poll_interval=poll_interval,
            timeout=timeout,
            **kwargs,
        )

        # If failed, return early with error info
        if result.state == "failed":
            # Save failed task info
            self._save_task_info(task_id, {
                "source_file": str(file_path),
                "source_file_name": file_path.name,
                "status": "failed",
                "error": result.err_msg,
                "output_dir": str(task_dir),
            })

            return MinerUExtractResult(
                file_name=result.file_name,
                state=result.state,
                source_file=source_file,
                err_msg=result.err_msg,
                task_id=task_id,
            )

        # Download zip file
        if result.full_zip_url:
            zip_path = task_dir / "result.zip"
            await self.download_zip(result.full_zip_url, zip_path)

            # Extract zip file
            extracted_dir = task_dir / "extracted"
            self.extract_zip(zip_path, extracted_dir)

            # Save completed task info
            self._save_task_info(task_id, {
                "source_file": str(file_path),
                "source_file_name": file_path.name,
                "status": "completed",
                "full_zip_url": result.full_zip_url,
                "output_dir": str(task_dir),
                "extracted_dir": str(extracted_dir),
            })

            return MinerUExtractResult(
                file_name=result.file_name,
                state=result.state,
                source_file=source_file,
                zip_file=zip_path,
                extracted_dir=extracted_dir,
                full_zip_url=result.full_zip_url,
                task_id=task_id,
            )

        # No zip URL, return with what's available
        return MinerUExtractResult(
            file_name=result.file_name,
            state=result.state,
            source_file=source_file,
            full_zip_url=result.full_zip_url,
            task_id=task_id,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        logger.debug("Closing MinerUService HTTP client")
        await self._client.aclose()

    async def __aenter__(self) -> "MinerUService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()