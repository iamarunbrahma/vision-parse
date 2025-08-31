import pytest
from pathlib import Path
import pypdfium2 as pdfium  # pypdfium2 library for PDF handling
from vision_parse.parser import PDFPageConfig, VisionParser


# Fixture providing the path to the test PDF file
@pytest.fixture
def pdf_path():
    return Path("tests/Texas-Holdem-Rules.pdf")


# Fixture configuring PDF page rendering settings
@pytest.fixture
def page_config():
    return PDFPageConfig(
        dpi=400,  # High DPI for better image quality
        color_space="RGB",  # Standard color space for vision models
        include_annotations=True,  # Include PDF annotations in output
        preserve_transparency=False,  # Flatten transparency for consistent rendering
    )


# Fixture initializing the vision-based markdown parser with specific model settings
@pytest.fixture
def markdown_parser(page_config):
    return VisionParser(
        model_name="llama3.2-vision:11b",
        temperature=0.7,
        top_p=0.7,
        page_config=page_config,
    )


# Fixture handling PDF document lifecycle with proper cleanup
@pytest.fixture
def pdf_document(pdf_path):
    with pdfium.PdfDocument(pdf_path) as doc:
        yield doc
