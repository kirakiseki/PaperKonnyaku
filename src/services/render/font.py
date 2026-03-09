"""Font management module for PDF rendering.

This module provides font registration, font size estimation, and text wrapping
functionality for PDF translation rendering using PyMuPDF.
"""

import os
import platform
from typing import List, Optional, Tuple, Union

from loguru import logger


# Font extensions supported by PyMuPDF
PYMUPDF_SUPPORTED_EXTENSIONS = {".ttf", ".otf", ".ttc", ".otc", ".woff", ".woff2"}


class FontManager:
    """Manager for PDF font operations.

    This class provides methods for registering fonts, estimating appropriate
    font sizes, and wrapping text to fit within bounding boxes.

    Usage:
        from services.render.font import FontManager

        font_manager = FontManager(font_name="Helvetica", font_size=10.0)
        font_manager.register_font()
    """

    def __init__(
        self,
        font_name: Optional[str] = None,
        font_size: Optional[float] = None,
    ):
        """Initialize the FontManager.

        Args:
            font_name: Font name to use. Defaults to config value.
            font_size: Base font size. Defaults to config value.
        """
        from core.config import config

        self.font_name = font_name or config.render.font_name
        self.font_size = font_size or config.render.font_size
        self._font_file_path: Optional[str] = None

        # Try to register the font
        self._register_font(self.font_name)

    @property
    def registered_font_name(self) -> str:
        """Return the actual registered font name.

        Returns:
            The font name that is currently registered and available.
        """
        return self.font_name

    def _get_font_file_path(self, font_name: str) -> Optional[str]:
        """Find font file path from font name.

        Args:
            font_name: The font name to search for.

        Returns:
            Path to font file if found, None otherwise.
        """
        system = platform.system()

        # Define search paths based on OS
        if system == "Darwin":
            base_paths = [
                "/System/Library/Fonts",
                "/System/Library/Fonts/Supplemental",
                "/Library/Fonts",
                "/Library/Fonts/Supplemental",
            ]
        elif system == "Linux":
            base_paths = [
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.fonts"),
                os.path.expanduser("~/.local/share/fonts"),
            ]
        elif system == "Windows":
            base_paths = [
                "C:\\Windows\\Fonts",
                os.path.expanduser("~\\AppData\\Local\\Microsoft\\Windows\\Fonts"),
            ]
        else:
            return None

        # Try different extensions
        extensions = list(PYMUPDF_SUPPORTED_EXTENSIONS)

        for base_path in base_paths:
            if not os.path.exists(base_path):
                continue
            for ext in extensions:
                font_path = os.path.join(base_path, f"{font_name}{ext}")
                if os.path.isfile(font_path):
                    return font_path
                # Also try case-insensitive match
                try:
                    for f in os.listdir(base_path):
                        if f.lower().startswith(font_name.lower()) and f.lower().endswith(ext):
                            return os.path.join(base_path, f)
                except OSError:
                    continue

        return None

    def _register_font(self, font_name: str) -> None:
        """Try to register a font, fall back to Helvetica if not available.

        Args:
            font_name: The font name or path to register.
        """
        # Check if font_name is a valid file path
        if os.path.isfile(font_name):
            # It's a file path, try to use it directly
            if os.path.splitext(font_name)[1].lower() in PYMUPDF_SUPPORTED_EXTENSIONS:
                self._font_file_path = font_name
                # Extract font name from file path
                self.font_name = os.path.splitext(os.path.basename(font_path))[0] if (font_path := font_name) else font_name
                logger.info(f"Using font from path: {font_name}")
                return
            else:
                logger.warning(f"Unsupported font extension for {font_name}")

        # Try to find system font
        font_path = self._get_font_file_path(font_name)
        if font_path:
            ext = os.path.splitext(font_path)[1].lower()
            if ext in PYMUPDF_SUPPORTED_EXTENSIONS:
                self._font_file_path = font_path
                logger.info(f"Found system font: {font_name} at {font_path}")
                return

        # Fall back to Helvetica if font not found
        if font_name != "Helvetica":
            logger.warning(f"Font '{font_name}' not available, falling back to Helvetica")
            self.font_name = "Helvetica"
        self._font_file_path = None

    def register_font_from_path(self, font_path: str) -> str:
        """Register a font from a file path.

        Args:
            font_path: Path to the font file.

        Returns:
            The registered font name.

        Raises:
            ValueError: If the font file doesn't exist or can't be registered.
        """
        if not os.path.exists(font_path):
            raise ValueError(f"Font file not found: {font_path}")

        ext = os.path.splitext(font_path)[1].lower()
        if ext not in PYMUPDF_SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported font extension: {ext}")

        # Extract font name from file path for registration
        registered_name = os.path.splitext(os.path.basename(font_path))[0]

        self._font_file_path = font_path
        self.font_name = registered_name
        logger.info(f"Registered font from path: {font_path}")
        return registered_name

    def get_font_for_pymupdf(self) -> Optional[str]:
        """Get font path for PyMuPDF.

        Returns:
            Font file path for PyMuPDF, or None if using built-in PDF font.
        """
        # Check if font_name is a file path
        if os.path.isfile(self.font_name):
            return self.font_name

        # Use registered font file path if available
        if self._font_file_path:
            return self._font_file_path

        # Try to find system font
        font_path = self._get_font_file_path(self.font_name)
        if font_path:
            return font_path

        # For built-in PDF fonts, return None (PyMuPDF will use default)
        if self.font_name in ("Helvetica", "Times-Roman", "Courier"):
            return None

        logger.warning(f"Could not find font file for {self.font_name}")
        return None

    def get_font_name_for_pymupdf(self) -> str:
        """Get font name for PyMuPDF's insert_text function.

        PyMuPDF uses specific font names for built-in fonts:
        - "helv" for Helvetica
        - "tiro" for Times-Roman
        - "cour" for Courier
        - Or a custom font file path

        Returns:
            Font name or path for PyMuPDF.
        """
        # Map common font names to PyMuPDF built-in fonts
        font_mapping = {
            "Helvetica": "helv",
            "Helvetica-Bold": "helvb",
            "Times-Roman": "tiro",
            "Times-Roman-Bold": "tirob",
            "Courier": "cour",
            "Courier-Bold": "courb",
        }

        # Check if it's a built-in font
        if self.font_name in font_mapping:
            return font_mapping[self.font_name]

        # Check if it's a file path or return the name
        if self._font_file_path:
            return self._font_file_path

        # Try to find system font
        font_path = self._get_font_file_path(self.font_name)
        if font_path:
            return font_path

        # Fallback to Helvetica
        return "helv"

    @staticmethod
    def find_system_cjk_fonts() -> dict:
        """Find available CJK (Chinese/Japanese/Korean) fonts on the system.

        Returns:
            Dictionary mapping font names to file paths.
        """
        cjk_font_patterns = {
            "Darwin": {
                "Noto Sans CJK": ["/System/Library/Fonts/Supplemental/NotoSansCJK-Regular.ttc"],
                "Noto Sans JP": ["/System/Library/Fonts/Supplemental/NotoSansJP-Regular.otf"],
                "Noto Sans SC": ["/System/Library/Fonts/Supplemental/NotoSansSC-Regular.otf"],
                "Noto Sans TC": ["/System/Library/Fonts/Supplemental/NotoSansTC-Regular.otf"],
                "Noto Sans KR": ["/System/Library/Fonts/Supplemental/NotoSansKR-Regular.otf"],
                "Hiragino Sans": ["/System/Library/Fonts/Hiragino Sans GB W3.otf",
                                  "/System/Library/Fonts/HiraginoSans-W3.ttc"],
                "PingFang": ["/System/Library/Fonts/PingFang.ttc"],
                "STHeiti": ["/System/Library/Fonts/STHeiti Light.ttc",
                            "/System/Library/Fonts/STHeiti Medium.ttc"],
                "SimHei": ["/Library/Fonts/SimHei.ttf"],
                "SimSun": ["/Library/Fonts/SimSun.ttf"],
            },
            "Linux": {
                "Noto Sans CJK": ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"],
                "Noto Sans SC": ["/usr/share/fonts/opentype/noto/NotoSansSC-Regular.otf"],
                "WenQuanYi": ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"],
            },
            "Windows": {
                "Microsoft YaHei": ["C:\\Windows\\Fonts\\msyh.ttc"],
                "SimSun": ["C:\\Windows\\Fonts\\simsun.ttc"],
                "SimHei": ["C:\\Windows\\Fonts\\simhei.ttf"],
            }
        }

        system = platform.system()
        patterns = cjk_font_patterns.get(system, {})

        found_fonts = {}
        for font_name, paths in patterns.items():
            for path in paths:
                if os.path.exists(path):
                    found_fonts[font_name] = path
                    logger.info(f"Found CJK font: {font_name} at {path}")
                    break

        return found_fonts

    def estimate_font_size(
        self,
        bbox_width: float,
        text: str,
        base_size: Optional[float] = None,
    ) -> float:
        """Estimate appropriate font size based on bbox width and text length.

        Args:
            bbox_width: Width of the bounding box in points.
            text: The text to fit.
            base_size: Base font size to start from. Defaults to self.font_size.

        Returns:
            Estimated font size that fits the bbox.
        """
        if base_size is None:
            base_size = self.font_size

        if not text:
            return base_size

        # Simple estimation: average character width is roughly 0.6 * font_size for Helvetica
        avg_char_width = 0.6 * base_size
        required_width = len(text) * avg_char_width

        if required_width > bbox_width:
            # Scale down to fit
            scale = bbox_width / required_width
            return max(base_size * scale, 4.0)  # Minimum font size of 4
        return base_size

    def wrap_text(
        self,
        text: str,
        bbox_width: float,
        font_size: Optional[float] = None,
    ) -> List[str]:
        """Wrap text to fit within the bounding box width.

        Args:
            text: The text to wrap.
            bbox_width: Width of the bounding box in points.
            font_size: Current font size. Defaults to self.font_size.

        Returns:
            List of wrapped text lines.
        """
        if font_size is None:
            font_size = self.font_size

        if not text:
            return []

        # Estimate average character width
        avg_char_width = 0.6 * font_size
        chars_per_line = int(bbox_width / avg_char_width)

        if chars_per_line <= 0:
            return [text]

        lines = []
        words = text.split()
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                # Check if single word is longer than line
                if len(word) > chars_per_line:
                    # Break long word
                    while len(word) > chars_per_line:
                        lines.append(word[:chars_per_line])
                        word = word[chars_per_line:]
                    current_line = word
                else:
                    current_line = word

        if current_line:
            lines.append(current_line)

        return lines