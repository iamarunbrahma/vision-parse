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
@patch("ollama.AsyncClient")
async def test_ollama_deepseek_r1_generate_markdown(
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
        model_name="deepseek-r1:32b",
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
    assert "deepseek llm" in llm
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
@patch("google.generativeai.GenerativeModel")
async def test_gemini_generate_markdown(
    MockGenerativeModel, sample_base64_image, mock_pixmap
):
    """Test markdown generation using Gemini."""
    mock_client = AsyncMock()
    MockGenerativeModel.return_value = mock_client

    # Mock responses for both structured analysis and markdown generation
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

    mock_client.generate_content_async = AsyncMock(
        side_effect=[mock_response1, mock_response2]
    )

    llm = LLM(
        model_name="gemini-1.5-pro",
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
    assert mock_client.generate_content_async.call_count == 2


@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_deepseek_generate_markdown(
    MockAsyncOpenAI, sample_base64_image, mock_pixmap
):
    """Test markdown generation using Deepseek."""
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
        model_name="deepseek-chat",
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
    assert mock_client.chat.completions.create.call_count == 2


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

    # Mock the chat responses to raise an exception
    mock_chat = AsyncMock()
    mock_chat.side_effect = Exception("test error")
    mock_client.chat = mock_chat

    llm = LLM(
        model_name="llama3.2-vision:11b",
        temperature=0.7,
        top_p=0.7,
        api_key=None,
        ollama_config={"OLLAMA_BASE_URL": "http://localhost:11434"},
        image_mode="base64",
        detailed_extraction=True,
        enable_concurrency=True,
        num_workers=1,
    )

    # Test that LLMError is raised
    with pytest.raises(LLMError) as exc_info:
        await llm.generate_markdown(mock_pixmap)
    assert "Error processing image with Ollama" in str(exc_info.value)


@pytest.mark.asyncio
@patch("groq.AsyncGroq")
async def test_groq_generate_markdown(
    MockAsyncGroq, sample_base64_image, mock_pixmap,
):
    """Test markdown generation using Groq."""
    # Mock the Groq async client
    mock_client = AsyncMock()
    MockAsyncGroq.return_value = mock_client

    # Mock the chat completions responses
    mock_completions = AsyncMock()
    mock_client.chat.completions.create = mock_completions
    mock_completions.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "text_detected": "Yes",
                            "tables_detected": "No",
                            "images_detected": "No",
                            "latex_equations_detected": "No",
                            "extracted_text": "Test content from Groq",
                            "confidence_score_text": 0.95,
                        }
                    )
                )
            )
        ]
    )

    llm = LLM(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.7,
        top_p=0.7,
        api_key="test-key",
        groq_config={
            "GROQ_MAX_TOKENS": 4096,
            "GROQ_TIMEOUT": 300.0,
        },
        image_mode="base64",
        detailed_extraction=True,
        enable_concurrency=True,
        num_workers=1,
    )

    # Call the method we want to test
    result = await llm.generate_markdown(mock_pixmap)

    # Verify that the client was called with the correct parameters
    assert mock_completions.call_count == 1
    args, kwargs = mock_completions.call_args
    assert kwargs["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.7
    assert kwargs["max_tokens"] == 4096

    # Check the content of the messages
    messages = kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], list)
    assert len(messages[0]["content"]) == 2
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][1]["type"] == "image_url"
    assert "data:image/jpeg;base64," in messages[0]["content"][1]["image_url"]["url"]

    # Verify result contains expected data
    assert "text_detected" in result
    assert result["extracted_text"] == "Test content from Groq"
    assert result["confidence_score_text"] == 0.95


@pytest.mark.asyncio
@patch("groq.AsyncGroq")
async def test_groq_error_handling(
    MockAsyncGroq, sample_base64_image, mock_pixmap,
):
    """Test error handling with Groq API."""
    # Mock the Groq async client
    mock_client = AsyncMock()
    MockAsyncGroq.return_value = mock_client

    # Mock the chat completions to raise an exception
    mock_completions = AsyncMock()
    mock_client.chat.completions.create = mock_completions
    mock_completions.side_effect = Exception("Image too large - images can contain at most 33177600 pixels. but image contained 39289320")

    llm = LLM(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.7,
        top_p=0.7,
        api_key="test-key",
        groq_config={
            "GROQ_MAX_TOKENS": 4096,
            "GROQ_TIMEOUT": 300.0,
        },
        image_mode="base64",
        detailed_extraction=True,
        enable_concurrency=True,
        num_workers=1,
    )

    # Test that the appropriate error is raised
    with pytest.raises(LLMError) as exc_info:
        await llm.generate_markdown(mock_pixmap)
    
    # Check that the error message includes our custom handling
    error_msg = str(exc_info.value)
    assert "Image exceeds Groq's size limit" in error_msg
    assert "max 33,177,600 pixels" in error_msg
    assert "lower DPI setting" in error_msg
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
    assert mock_client.chat.call_count == 1
