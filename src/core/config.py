"""Configuration management module.

This module provides a configuration class that reads and validates
settings from config.toml in the project root directory.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import toml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field


class MinerUConfig(BaseModel):
    """MinerU extraction service configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_token: str = Field(default="", description="API token for MinerU service")
    url: str = Field(default="https://mineru.net/api/v4", description="MinerU API base URL")
    poll_interval: float = Field(default=2.0, description="Poll interval in seconds")
    timeout: float = Field(default=300.0, description="Timeout in seconds")
    output_dir: Path = Field(default=Path("./output"), description="Output directory for extracted files")
    task_history_dir: Path = Field(default=Path("./output/tasks"), description="Task history directory")


class TestConfig(BaseModel):
    """Test configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    test_file: str = Field(default="tests/assets/test_example.pdf", description="Test file path")


class ExtractConfig(BaseModel):
    """Extract module configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mineru: MinerUConfig = Field(default=MinerUConfig(), description="MinerU configuration")


class AppConfig(BaseModel):
    """Application configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    extract: ExtractConfig = Field(default=ExtractConfig(), description="Extract configuration")
    test: TestConfig = Field(default=TestConfig(), description="Test configuration")


class Config:
    """Configuration manager singleton class.

    This class provides a singleton instance that loads configuration
    from config.toml in the project root directory.

    Usage:
        from core.config import config

        # Access configuration values
        mineru_token = config.extract.mineru.api_token
        mineru_url = config.extract.mineru.url
        output_dir = config.extract.mineru.output_dir
        test_file = config.test.test_file

        # Or access the full config object
        full_config = config.settings
    """

    _instance: Optional["Config"] = None
    _config_path: Optional[Path] = None

    def __new__(cls) -> "Config":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load and validate configuration from config.toml."""
        # Determine config.toml path (project root)
        config_dir = Path(__file__).parent.parent.parent
        self._config_path = config_dir / "config.toml"

        logger.info(f"Loading config from: {self._config_path}")

        if not self._config_path.exists():
            logger.warning("config.toml not found, creating default config")
            # Create default config if not exists
            self._create_default_config()
        else:
            logger.debug("config.toml found, loading settings")

        # Read and parse TOML file
        toml_data = toml.load(self._config_path)

        # Merge with defaults and validate
        self._settings = AppConfig(**toml_data)

        logger.info("Configuration loaded successfully")

    def _create_default_config(self) -> None:
        """Create a default config.toml file."""
        default_config = {
            "extract": {
                "mineru": {
                    "api_token": "",
                    "url": "https://mineru.net/api/v4",
                    "poll_interval": 2.0,
                    "timeout": 300.0,
                    "output_dir": "./output",
                    "task_history_dir": "./output/tasks",
                }
            },
            "test": {
                "test_file": "tests/assets/test_example.pdf",
            }
        }

        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w", encoding="utf-8") as f:
            toml.dump(default_config, f)

    @property
    def settings(self) -> AppConfig:
        """Get the validated configuration object."""
        return self._settings

    # Expose commonly used settings as properties for convenience
    @property
    def extract(self) -> ExtractConfig:
        return self._settings.extract

    @property
    def test(self) -> TestConfig:
        return self._settings.test


# Singleton instance that can be imported directly
config = Config()


class TaskHistory:
    """Task history manager for tracking extraction tasks."""

    def __init__(self, history_dir: Optional[Path] = None):
        """Initialize task history manager.

        Args:
            history_dir: Directory to store task history. Defaults to config setting.
        """
        self.history_dir = history_dir or config.extract.mineru.task_history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"TaskHistory initialized with directory: {self.history_dir}")

    def save_task(self, task_info: dict) -> Path:
        """Save task information to history.

        Args:
            task_info: Dictionary containing task information.

        Returns:
            Path to the saved task file.
        """
        task_id = task_info.get("task_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        task_file = self.history_dir / f"{task_id}.json"

        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task_info, f, indent=2, ensure_ascii=False)

        logger.info(f"Task saved: {task_id} -> {task_file}")
        return task_file

    def load_task(self, task_id: str) -> Optional[dict]:
        """Load task information from history.

        Args:
            task_id: Task ID to load.

        Returns:
            Task information dictionary or None if not found.
        """
        task_file = self.history_dir / f"{task_id}.json"
        if not task_file.exists():
            logger.warning(f"Task not found: {task_id}")
            return None

        with open(task_file, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        logger.debug(f"Task loaded: {task_id}")
        return task_data

    def list_tasks(self) -> list[dict]:
        """List all tasks in history.

        Returns:
            List of task information dictionaries.
        """
        tasks = []
        for task_file in sorted(self.history_dir.glob("*.json")):
            with open(task_file, "r", encoding="utf-8") as f:
                tasks.append(json.load(f))
        logger.debug(f"Listed {len(tasks)} tasks from history")
        return tasks