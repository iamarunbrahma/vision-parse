import os
import socket
from typing import Optional
from urllib.parse import urlparse

import pytest

from vision_parse import LLMError, VisionParser

pytestmark = pytest.mark.integration


def _can_connect(base_url: str, timeout: float = 1.5) -> bool:
    """Return True if a TCP connection to the provided base URL succeeds."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        return False

    host = parsed.hostname
    port: Optional[int] = parsed.port

    if host is None:
        return False

    if port is None:
        port = 443 if parsed.scheme == "https" else 80

    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def test_vllm_generate_markdown_integration(pdf_path):
    """Exercise the full VisionParser stack against a live vLLM endpoint."""
    base_url = (
        os.getenv("VLLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "http://localhost:8000/v1"
    )

    if not _can_connect(base_url):
        pytest.skip(
            "No reachable vLLM endpoint detected. Set VLLM_BASE_URL to a running server"
        )

    api_key = (
        os.getenv("VLLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "EMPTY"
    )

    parser = VisionParser(
        model_name="unsloth/Mistral-Small-3.1-24B-Instruct-2503-bnb-4bit",
        detailed_extraction=True,
        enable_concurrency=True,
        openai_config={
            "OPENAI_BASE_URL": base_url,
            "OPENAI_API_KEY": api_key,
        },
    )

    try:
        converted_pages = parser.convert_pdf(pdf_path)
    except LLMError as exc:
        pytest.skip(f"vLLM endpoint rejected the request: {exc}")

    assert isinstance(converted_pages, list)
    assert converted_pages, "Expected at least one page of markdown"
    assert any(page.strip() for page in converted_pages), "Pages should not be empty"
