"""
GLM-OCR client supporting LM Studio and Ollama backends.

Supports the three official GLM-OCR prompt modes:
  - "Text Recognition:"    — general text OCR
  - "Formula Recognition:"  — mathematical formulas
  - "Table Recognition:"    — table structure

LM Studio: uses /v1/chat/completions with OpenAI vision format (image_url)
Ollama:    uses /api/generate with native images field
"""

import base64
import json
import requests
import time

from .config import (
    OCR_PROMPTS,
    DEFAULT_OCR_MODEL,
    DEFAULT_OCR_TEMPERATURE,
    DEFAULT_OCR_MAX_TOKENS,
    DEFAULT_OCR_TIMEOUT,
    MAX_RETRIES,
)


def _encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def ocr_image_lmstudio(
    image_bytes: bytes,
    prompt: str,
    base_url: str,
    model: str,
    temperature: float = DEFAULT_OCR_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
    timeout: int = DEFAULT_OCR_TIMEOUT,
) -> str:
    """
    Perform OCR via LM Studio's OpenAI-compatible vision API.

    Uses /v1/chat/completions with image_url content type.
    """
    b64 = _encode_image_base64(image_bytes)
    data_uri = f"data:image/png;base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json; charset=utf-8",
    }

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"]


def ocr_image_ollama(
    image_bytes: bytes,
    prompt: str,
    base_url: str,
    model: str,
    temperature: float = DEFAULT_OCR_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
    timeout: int = DEFAULT_OCR_TIMEOUT,
) -> str:
    """
    Perform OCR via Ollama's native /api/generate endpoint.

    Per official GLM-OCR documentation, Ollama uses the native endpoint
    with 'images' field containing base64-encoded images.
    """
    b64 = _encode_image_base64(image_bytes)

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    resp = requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    return data.get("response", "")


def ocr_image(
    image_bytes: bytes,
    prompt_mode: str = "text",
    base_url: str = "http://localhost:1234",
    model: str = DEFAULT_OCR_MODEL,
    backend_name: str = "LM Studio",
    temperature: float = DEFAULT_OCR_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
    timeout: int = DEFAULT_OCR_TIMEOUT,
) -> str:
    """
    Perform OCR on an image using GLM-OCR.

    Args:
        image_bytes: Raw image bytes (PNG, JPEG, etc.)
        prompt_mode: One of "text", "formula", "table"
        base_url: Backend server URL
        model: Model name (e.g., "glm-ocr", "glm-ocr:latest")
        backend_name: "LM Studio" or "Ollama"
        temperature: Inference temperature (GLM-OCR recommends 0.01)
        max_tokens: Maximum output tokens
        timeout: Request timeout in seconds

    Returns:
        Recognized text (Markdown-formatted for tables/formulas)
    """
    prompt = OCR_PROMPTS.get(prompt_mode, OCR_PROMPTS["text"])

    if backend_name == "Ollama":
        return ocr_image_ollama(
            image_bytes, prompt, base_url, model,
            temperature, max_tokens, timeout,
        )
    else:
        return ocr_image_lmstudio(
            image_bytes, prompt, base_url, model,
            temperature, max_tokens, timeout,
        )


def ocr_image_with_retry(
    image_bytes: bytes,
    prompt_mode: str = "text",
    base_url: str = "http://localhost:1234",
    model: str = DEFAULT_OCR_MODEL,
    backend_name: str = "LM Studio",
    temperature: float = DEFAULT_OCR_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
    timeout: int = DEFAULT_OCR_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> str:
    """OCR with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            return ocr_image(
                image_bytes, prompt_mode, base_url, model,
                backend_name, temperature, max_tokens, timeout,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"OCR failed after {max_retries} attempts: {e}"
                ) from e
    return ""


def ocr_multi_images(
    image_list: list[bytes],
    prompt_mode: str = "text",
    base_url: str = "http://localhost:1234",
    model: str = DEFAULT_OCR_MODEL,
    backend_name: str = "LM Studio",
    temperature: float = DEFAULT_OCR_TEMPERATURE,
    max_tokens: int = DEFAULT_OCR_MAX_TOKENS,
    timeout: int = DEFAULT_OCR_TIMEOUT,
    progress_callback=None,
) -> list[str]:
    """
    Perform OCR on multiple images (e.g., pages of a document).

    Args:
        image_list: List of image bytes
        progress_callback: Optional callable(current, total) for progress
        ...other args: Same as ocr_image

    Returns:
        List of recognized text strings, one per image
    """
    results = []
    total = len(image_list)

    for i, img_bytes in enumerate(image_list):
        if progress_callback:
            progress_callback(i, total)

        text = ocr_image_with_retry(
            img_bytes, prompt_mode, base_url, model,
            backend_name, temperature, max_tokens, timeout,
        )
        results.append(text)

    if progress_callback:
        progress_callback(total, total)

    return results
