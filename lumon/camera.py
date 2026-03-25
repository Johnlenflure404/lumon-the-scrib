"""
Camera capture utilities for live translation workflow.

Uses streamlit-webrtc for live camera preview and frame capture.
Supports manual capture and automatic capture at configurable intervals.
"""

import io
import time
from dataclasses import dataclass, field

from PIL import Image


@dataclass
class CapturedPage:
    """A single captured page with its OCR result."""
    image_bytes: bytes
    ocr_text: str = ""
    translated_text: str = ""
    timestamp: float = 0.0
    page_number: int = 0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class CaptureSession:
    """Manages a camera capture session with accumulated pages."""
    pages: list[CapturedPage] = field(default_factory=list)
    auto_capture_interval: float = 0.0  # 0 = disabled
    last_capture_time: float = 0.0

    def add_page(self, image_bytes: bytes) -> CapturedPage:
        """Add a new captured page to the session."""
        page = CapturedPage(
            image_bytes=image_bytes,
            page_number=len(self.pages) + 1,
        )
        self.pages.append(page)
        self.last_capture_time = time.time()
        return page

    def should_auto_capture(self) -> bool:
        """Check if it's time for an automatic capture."""
        if self.auto_capture_interval <= 0:
            return False
        return (time.time() - self.last_capture_time) >= self.auto_capture_interval

    def get_all_ocr_text(self) -> str:
        """Get combined OCR text from all pages."""
        texts = []
        for page in self.pages:
            if page.ocr_text:
                texts.append(f"--- Page {page.page_number} ---\n{page.ocr_text}")
        return "\n\n".join(texts)

    def get_all_translated_text(self) -> str:
        """Get combined translated text from all pages."""
        texts = []
        for page in self.pages:
            if page.translated_text:
                texts.append(page.translated_text)
        return "\n\n---\n\n".join(texts)

    def remove_page(self, index: int):
        """Remove a page by index and renumber remaining pages."""
        if 0 <= index < len(self.pages):
            self.pages.pop(index)
            for i, page in enumerate(self.pages):
                page.page_number = i + 1

    def clear(self):
        """Clear all captured pages."""
        self.pages.clear()
        self.last_capture_time = 0.0


def frame_to_bytes(frame, format: str = "PNG") -> bytes:
    """
    Convert a video frame (numpy array) to image bytes.

    Args:
        frame: numpy array (H, W, 3) in RGB or BGR format
        format: Output image format (PNG, JPEG)

    Returns:
        Image bytes
    """
    import numpy as np

    if isinstance(frame, np.ndarray):
        # WebRTC frames are typically BGR, convert to RGB
        if frame.shape[2] == 3:
            img = Image.fromarray(frame)
        else:
            img = Image.fromarray(frame[:, :, :3])
    else:
        img = frame

    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def create_thumbnail(image_bytes: bytes, max_size: tuple[int, int] = (200, 200)) -> bytes:
    """Create a thumbnail from image bytes."""
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
