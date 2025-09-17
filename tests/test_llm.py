import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import base64
from vision_parse.llm import LLM, UnsupportedModelError, LLMError


@pytest.fixture
def sample_base64_image():
    return base64.b64encode(b"test_image").decode("utf-8")


@pytest.fixture
def mock_structured_response():
    return {
        "message": {
            "content": json.dumps(
                {
                    "text_detected": "Yes",
                    "tables_detected": "No",
                    "images_detected": "No",
                    "extracted_text": "Test content",
                    "confidence_score_text": 0.9,
                }
            )
        }
    }


@pytest.fixture
def mock_markdown_response():
    return {"message": {"content": "# Test Header\n\nThis is test content."}}


@pytest.fixture
def mock_pixmap():
    mock = MagicMock()
    mock.samples = b"\x00" * (200 * 200 * 3)  # Create correct size buffer
    mock.height = 200
    mock.width = 200
    mock.n = 3
    mock.tobytes.return_value = b"test_image_data"
    return mock


def test_unsupported_model():
    """Test error handling for unsupported models."""
    with pytest.raises(UnsupportedModelError) as exc_info:
        LLM(
            model_name="unsupported-model",
            temperature=0.7,
            top_p=0.7,
            api_key=None,
            ollama_config=None,
            openai_config=None,
            gemini_config=None,
            image_mode=None,
            custom_prompt=None,
            detailed_extraction=False,
            enable_concurrency=False,
            device=None,
            num_workers=1,
        )
    assert "is not supported" in str(exc_info.value)


@pytest.mark.asyncio
@patch("ollama.AsyncClient")
async def test_ollama_generate_markdown(
    mock_async_client,
    sample_base64_image,
    mock_pixmap,
):
    """Test markdown generation using Ollama."""
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
                        "latex_equations_detected": "No",
                        "extracted_text": "Test content",
                        "confidence_score_text": 0.9,
                    }
                )
            }
        }
    ]
    mock_client.chat = mock_chat

    llm = LLM(
        model_name="llama3.2-vision:11b",
        temperature=0.7,
        top_p=0.7,
        api_key=None,
        ollama_config=None,
        openai_config=None,
        gemini_config=None,
        image_mode=None,
        custom_prompt=None,
        detailed_extraction=True,
        enable_concurrency=True,
        device=None,
        num_workers=1,
    )
    result = await llm.generate_markdown(sample_base64_image, mock_pixmap, 0)

    assert isinstance(result, str)
    assert "Test content" in result
    assert mock_chat.call_count == 1


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_openai_generate_markdown(
    MockAsyncOpenAI, sample_base64_image, mock_pixmap
):
    """Test markdown generation using OpenAI."""
    mock_client = AsyncMock()
    MockAsyncOpenAI.return_value = mock_client

    # Mock structured analysis response
    mock_parse = AsyncMock()
    mock_parse.choices = [
        AsyncMock(
            message=AsyncMock(
                content=json.dumps(
                    {
                        "text_detected": "Yes",
                        "tables_detected": "No",
                        "images_detected": "No",
                        "latex_equations_detected": "No",
                        "extracted_text": "Test content",
                        "confidence_score_text": 0.9,
                    }
                )
            )
        )
    ]
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=mock_parse)

    # Mock markdown conversion response
    mock_create = AsyncMock()
    mock_create.choices = [
        AsyncMock(message=AsyncMock(content="# Test Header\n\nTest content"))
    ]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_create)

    llm = LLM(
        model_name="gpt-4o",
        api_key="test-key",
        temperature=0.7,
        top_p=0.7,
        ollama_config=None,
        openai_config=None,
        gemini_config=None,
        image_mode=None,
        custom_prompt=None,
        detailed_extraction=True,
        enable_concurrency=True,
        device=None,
        num_workers=1,
    )
    result = await llm.generate_markdown(sample_base64_image, mock_pixmap, 0)

    assert isinstance(result, str)
    assert "Test content" in result
    assert mock_client.beta.chat.completions.parse.called
    assert mock_client.chat.completions.create.called


@pytest.mark.asyncio
@patch("vllm.AsyncOpenAI")
async def test_vllm_generate_markdown(
    MockAsyncOpenAI, sample_base64_image, mock_pixmap, monkeypatch
):
    """Test markdown generation routed through a vLLM OpenAI-compatible endpoint."""

    for env_var in [
        "OPENAI_BASE_URL",
        "VLLM_BASE_URL",
        "OPENAI_API_KEY",
        "VLLM_API_KEY",
    ]:
        monkeypatch.delenv(env_var, raising=False)

    mock_client = AsyncMock()
    MockAsyncOpenAI.return_value = mock_client

    structured_response = MagicMock()
    structured_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "text_detected": "Yes",
                        "tables_detected": "No",
                        "images_detected": "No",
                        "latex_equations_detected": "No",
                        "extracted_text": "Test content",
                        "confidence_score_text": 0.9,
                    }
                )
            )
        )
    ]

    markdown_response = MagicMock()
    markdown_response.choices = [
        MagicMock(message=MagicMock(content="# Test Header\n\nTest content"))
    ]

    mock_client.chat.completions.create = AsyncMock(
        side_effect=[structured_response, markdown_response]
    )
    mock_client.beta.chat.completions.parse = AsyncMock()

    llm = LLM(
        model_name="unsloth/Mistral-Small-3.1-24B-Instruct-2503-bnb-4bit",
        api_key=None,
        temperature=0.6,
        top_p=0.8,
        ollama_config=None,
        openai_config={},
        gemini_config=None,
        image_mode=None,
        custom_prompt=None,
        detailed_extraction=True,
        enable_concurrency=True,
        device=None,
        num_workers=1,
    )

    result = await llm.generate_markdown(sample_base64_image, mock_pixmap, 0)

    assert isinstance(result, str)
    assert "Test content" in result
    assert mock_client.chat.completions.create.call_count == 2
    mock_client.beta.chat.completions.parse.assert_not_called()

    call_kwargs = MockAsyncOpenAI.call_args.kwargs
    assert call_kwargs["base_url"] == "http://localhost:8000/v1"
    assert call_kwargs["api_key"] == "EMPTY"


@pytest.mark.asyncio
@patch("openai.AsyncAzureOpenAI")
async def test_azure_openai_generate_markdown(
    MockAsyncAzureOpenAI, sample_base64_image, mock_pixmap
):
    """Test markdown generation using Azure OpenAI."""

    mock_client = AsyncMock()
    MockAsyncAzureOpenAI.return_value = mock_client

    # Mock structured analysis response
    mock_parse = AsyncMock()
    mock_parse.choices = [
        AsyncMock(
            message=AsyncMock(
                content=json.dumps(
                    {
                        "text_detected": "Yes",
                        "tables_detected": "No",
                        "images_detected": "No",
                        "latex_equations_detected": "No",
                        "extracted_text": "Test content",
                        "confidence_score_text": 0.9,
                    }
                )
            )
        )
    ]

    # Mock markdown conversion response
    mock_create = AsyncMock()
    mock_create.choices = [
        AsyncMock(message=AsyncMock(content="# Test Header\n\nTest content"))
    ]
    # Set up side effects to return mock_parse first, then mock_create
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[mock_parse, mock_create]
    )

    llm = LLM(
        model_name="gpt-4o",
        api_key=None,
        temperature=0.7,
        top_p=0.7,
        ollama_config=None,
        openai_config={
            "AZURE_ENDPOINT_URL": "https://test.openai.azure.com/",
            "AZURE_DEPLOYMENT_NAME": "gpt-4o",
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_API_VERSION": "2024-08-01-preview",
        },
        gemini_config=None,
        image_mode=None,
        custom_prompt=None,
        detailed_extraction=True,
        enable_concurrency=True,
        device=None,
        num_workers=1,
    )
    result = await llm.generate_markdown(sample_base64_image, mock_pixmap, 0)

    assert isinstance(result, str)
    assert "Test content" in result
    assert mock_client.chat.completions.create.called


@pytest.mark.asyncio
@patch("google.genai.Client")
async def test_gemini_generate_markdown(
    MockGenaiClient, sample_base64_image, mock_pixmap
):
    """Test markdown generation using Gemini (google-genai SDK)."""
    # Prepare mocked responses
    mock_response1 = AsyncMock()
    mock_response1.text = json.dumps(
        {
            "text_detected": "Yes",
            "tables_detected": "No",
            "images_detected": "No",
            "latex_equations_detected": "No",
            "extracted_text": "Test content",
            "confidence_score_text": 0.9,
        }
    )
    mock_response2 = AsyncMock()
    mock_response2.text = "# Test Header\n\nTest content"

    # Build the nested async client shape used by the implementation
    mock_async_models = MagicMock()
    mock_async_models.generate_content = AsyncMock(
        side_effect=[mock_response1, mock_response2]
    )
    mock_client_instance = MagicMock()
    mock_client_instance.aio = MagicMock()
    mock_client_instance.aio.models = mock_async_models

    MockGenaiClient.return_value = mock_client_instance

    llm = LLM(
        model_name="gemini-2.5-pro",
        api_key="test-key",
        temperature=0.7,
        top_p=0.7,
        ollama_config=None,
        openai_config=None,
        gemini_config=None,
        image_mode=None,
        custom_prompt=None,
        detailed_extraction=True,
        enable_concurrency=True,
        device=None,
        num_workers=1,
    )
    result = await llm.generate_markdown(sample_base64_image, mock_pixmap, 0)

    assert isinstance(result, str)
    assert "Test content" in result
    assert mock_async_models.generate_content.call_count == 2


@pytest.mark.asyncio
@patch("ollama.AsyncClient")
async def test_ollama_base64_image_mode(
    mock_async_client,
    sample_base64_image,
    mock_pixmap,
):
    """Test markdown generation with base64 image mode."""
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

    # Mock the pixmap for image extraction
    mock_pixmap.samples = b"\x00" * (200 * 200 * 3)  # Create correct size buffer
    mock_pixmap.height = 200
    mock_pixmap.width = 200
    mock_pixmap.n = 3

    llm = LLM(
        model_name="llama3.2-vision:11b",
        temperature=0.7,
        top_p=0.7,
        api_key=None,
        ollama_config=None,
        openai_config=None,
        gemini_config=None,
        image_mode="base64",
        custom_prompt=None,
        detailed_extraction=True,
        enable_concurrency=True,
        device=None,
        num_workers=1,
    )
    result = await llm.generate_markdown(sample_base64_image, mock_pixmap, 0)

    assert isinstance(result, str)
    assert "# Test Header" in result
    assert "data:image/png;base64,test_image" in result
    assert mock_chat.call_count == 2


@pytest.mark.asyncio
@patch("ollama.AsyncClient")
async def test_ollama_llm_error(mock_async_client, sample_base64_image, mock_pixmap):
    """Test LLMError handling for Ollama."""
    # Mock the Ollama async client
    mock_client = AsyncMock()
    mock_async_client.return_value = mock_client

    # Mock a failed Ollama API call
    mock_client.chat.side_effect = Exception("Ollama processing failed")

    llm = LLM(
        model_name="llama3.2-vision:11b",
        temperature=0.7,
        top_p=0.7,
        api_key=None,
        ollama_config=None,
        openai_config=None,
        gemini_config=None,
        image_mode=None,
        custom_prompt=None,
        detailed_extraction=True,
        enable_concurrency=True,
        device=None,
        num_workers=1,
    )

    with pytest.raises(LLMError) as exc_info:
        await llm.generate_markdown(sample_base64_image, mock_pixmap, 0)
    assert "Ollama Model processing failed" in str(exc_info.value)
    # Retries and fallback can cause multiple chat attempts; ensure at least one occurred
    assert mock_client.chat.call_count >= 1
