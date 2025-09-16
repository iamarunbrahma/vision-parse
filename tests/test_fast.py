import os
from pathlib import Path

import pytest

from vision_parse import VisionParser, PDFPageConfig


@pytest.mark.skipif(
    any(
        __import__(mod, fromlist=["__dummy"]) is None
        for mod in ("pdfminer", "pdfplumber")
    ),
    reason="fast extras not installed; install with pip install 'vision-parse[fast]'",
)
def test_fast_mode_with_benchmark_pdf():
    # Use the benchmark PDF shipped with the repo
    repo_root = Path(__file__).resolve().parents[1]
    pdf_path = repo_root / "benchmarks" / "quantum_computing.pdf"

    # Ensure the file exists; if not, skip gracefully
    if not pdf_path.exists():
        pytest.skip("benchmark PDF not found: benchmarks/quantum_computing.pdf")

    parser = VisionParser(
        page_config=PDFPageConfig(dpi=300),
        fast_mode=True,
    )
    pages = parser.convert_pdf(pdf_path)

    # Dummy assertion for now
    assert isinstance(pages, list)

