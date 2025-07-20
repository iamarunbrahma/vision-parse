import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
from vision_parse import (
    VisionParserError,
    UnsupportedFileError,
    VisionParser,
    PDFPageConfig,
)


def test_convert_pdf_nonexistent_file(markdown_parser):
    """Test error handling for non-existent PDF files."""
    with pytest.raises(FileNotFoundError) as exc_info:
        markdown_parser.convert_pdf("non-existent.pdf")
    assert "PDF file not found" in str(exc_info.value)


def test_convert_pdf_invalid_file(markdown_parser, tmp_path):
    """Test error handling for invalid file types."""
    # Create a temporary text file
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("test content")

    with pytest.raises(UnsupportedFileError) as exc_info:
        markdown_parser.convert_pdf(invalid_file)
    assert "File is not a PDF" in str(exc_info.value)


def test_calculate_matrix(markdown_parser, pdf_document):
    """Test the matrix calculation for PDF page transformation."""
    page = pdf_document[0]
    matrix = markdown_parser._calculate_matrix(page)
    expected_zoom = markdown_parser.page_config.dpi / 72 * 2
    assert matrix.a == expected_zoom
    assert matrix.d == expected_zoom


def test_convert_pdf_integration(markdown_parser, pdf_path):
    """Integration test for PDF to markdown conversion."""
    converted_pages = markdown_parser.convert_pdf(pdf_path)
    assert isinstance(converted_pages, list)
    assert len(converted_pages) > 0
    assert any(
        isinstance(page, str) and len(page.strip()) > 0 for page in converted_pages
    )


def test_convert_pdf_vision_parser_error(markdown_parser, monkeypatch, pdf_path):
    """Test VisionParserError handling in convert_pdf method."""

    def mock_convert_page(*args, **kwargs):
        raise Exception("Failed to process page")

    monkeypatch.setattr(markdown_parser, "_convert_page", mock_convert_page)

    with pytest.raises(VisionParserError) as exc_info:
        markdown_parser.convert_pdf(pdf_path)
    assert "Failed to process page" in str(exc_info.value)


@pytest.mark.asyncio
@patch("ollama.AsyncClient")
@patch("tqdm.tqdm")
async def test_parser_with_base64_image_mode(mock_tqdm, mock_async_client, pdf_path):
    """Test parser with base64 image mode configuration."""
    # Mock the Ollama async client
    mock_client = AsyncMock()
    mock_async_client.return_value = mock_client

    # Mock the chat responses
    mock_chat = AsyncMock()
    mock_chat.side_effect = [
        {
            "message": {
                "content": json.dumps(
                    {
                        "text_detected": "Yes",
                        "tables_detected": "No",
                        "images_detected": "Yes",
                        "extracted_text": "Test content with image",
                        "confidence_score_text": 0.9,
                    }
                )
            }
        },
        {
            "message": {
                "content": "# Test Header\n\n![Image 1](data:image/png;base64,test_image)"
            }
        },
    ]
    mock_client.chat = mock_chat

    parser = VisionParser(
        page_config=PDFPageConfig(dpi=300),
        model_name="llama3.2-vision:11b",
        temperature=0.7,
        top_p=0.7,
        image_mode="base64",
        detailed_extraction=True,
        enable_concurrency=True,
    )

    # Mock the PDF document
    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.samples = b"\x00" * (200 * 200 * 3)  # Create correct size buffer
        mock_pixmap.height = 200
        mock_pixmap.width = 200
        mock_pixmap.n = 3
        mock_pixmap.tobytes = MagicMock(return_value=b"test_image_data")
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.page_count = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        # Mock tqdm
        mock_progress = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress

        converted_pages = parser.convert_pdf(pdf_path)
        assert isinstance(converted_pages, list)
        assert len(converted_pages) > 0
        # Check if any page contains base64 image data
        assert any("data:image/png;base64," in page for page in converted_pages if page)
        assert mock_chat.call_count == 2


@pytest.mark.asyncio
@patch("ollama.AsyncClient")
@patch("tqdm.tqdm")
async def test_parser_with_concurrent_processing(
    mock_tqdm, mock_async_client, pdf_path
):
    """Test parser with concurrent processing enabled."""
    # Mock the Ollama async client
    mock_client = AsyncMock()
    mock_async_client.return_value = mock_client

    # Mock the chat responses
    mock_chat = AsyncMock()
    mock_chat.side_effect = [
        {
            "message": {
                "content": json.dumps(
                    {
                        "text_detected": "Yes",
                        "tables_detected": "No",
                        "images_detected": "No",
                        "extracted_text": f"Test content for page {i}",
                        "confidence_score_text": 0.9,
                    }
                )
            }
        }
        for i in range(2)
    ] + [
        {"message": {"content": f"# Page {i}\n\nTest content for page {i}"}}
        for i in range(2)
    ]
    mock_client.chat = mock_chat

    parser = VisionParser(
        page_config=PDFPageConfig(dpi=300),
        model_name="llama3.2-vision:11b",
        temperature=0.7,
        top_p=0.7,
        enable_concurrency=True,
    )

    # Mock the PDF document
    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.samples = b"\x00" * (200 * 200 * 3)  # Create correct size buffer
        mock_pixmap.height = 200
        mock_pixmap.width = 200
        mock_pixmap.n = 3
        mock_pixmap.tobytes = MagicMock(return_value=b"test_image_data")
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.page_count = 2
        mock_doc.__getitem__.return_value = mock_page
        mock_open.return_value.__enter__.return_value = mock_doc

        # Mock tqdm
        mock_progress = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress

        converted_pages = parser.convert_pdf(pdf_path)
        assert isinstance(converted_pages, list)
        assert len(converted_pages) > 0
        assert mock_chat.call_count == 2
        assert all(
            f"Test content for page {i}" in page
            for i, page in enumerate(converted_pages)
        )


def test_parser_with_custom_page_config():
    """Test parser initialization with custom page configuration."""
    custom_config = PDFPageConfig(
        dpi=600,
        color_space="GRAY",
        include_annotations=False,
        preserve_transparency=True,
    )
    parser = VisionParser(
        page_config=custom_config,
        model_name="llama3.2-vision:11b",
        temperature=0.7,
        top_p=0.7,
    )
    assert parser.page_config.dpi == 600
    assert parser.page_config.color_space == "GRAY"
    assert not parser.page_config.include_annotations
    assert parser.page_config.preserve_transparency


def test_parser_with_ollama_config():
    """Test parser initialization with Ollama configuration."""
    ollama_config = {
        "OLLAMA_HOST": "http://localhost:11434",
        "OLLAMA_NUM_GPU": "1",
        "OLLAMA_NUM_THREAD": "4",
    }
    parser = VisionParser(
        model_name="llama3.2-vision:11b",
        temperature=0.7,
        top_p=0.7,
        ollama_config=ollama_config,
    )
    assert parser.llm.ollama_config == ollama_config


def test_parser_with_groq_config():
    """Test parser initialization with Groq configuration."""
    groq_config = {
        "GROQ_MAX_TOKENS": 4096,
        "GROQ_TIMEOUT": 300.0,
    }
    parser = VisionParser(
        page_config=PDFPageConfig(dpi=200),  # Lower DPI for Groq's pixel limit
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.7,
        top_p=0.7,
        api_key="test-key",
        groq_config=groq_config,
        enable_concurrency=True,
    )
    assert parser.llm.groq_config == groq_config
    assert parser.llm.provider == "groq"


def test_parser_with_vertex_config():
    """Test parser initialization with Vertex AI configuration."""
    vertex_config = {
        "PROJECT_ID": "test-project",
        "LOCATION": "us-central1",
        "CREDENTIALS_JSON": '{"type": "service_account", "project_id": "test-project"}'
    }
    parser = VisionParser(
        page_config=PDFPageConfig(dpi=200),  # Lower DPI for potential size limits
        model_name="gemini-1.5-pro-002",
        temperature=0.7,
        top_p=0.7,
        api_key="test-key",
        vertex_config=vertex_config,
    )
    assert parser.llm.vertex_config == vertex_config
    assert parser.llm.provider == "vertex"
