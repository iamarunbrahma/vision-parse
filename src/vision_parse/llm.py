from typing import Literal, Dict, Any, Union
from pydantic import BaseModel
from jinja2 import Template
import re
import fitz
import os
from tqdm import tqdm
from .utils import ImageData
from tenacity import retry, stop_after_attempt, wait_exponential
from .constants import SUPPORTED_MODELS
import logging

logger = logging.getLogger(__name__)


class ImageDescription(BaseModel):
    """Model Schema for image description."""

    text_detected: Literal["Yes", "No"]
    tables_detected: Literal["Yes", "No"]
    images_detected: Literal["Yes", "No"]
    latex_equations_detected: Literal["Yes", "No"]
    extracted_text: str
    confidence_score_text: float


class UnsupportedModelError(BaseException):
    """Custom exception for unsupported model names"""

    pass


class LLMError(BaseException):
    """Custom exception for Vision LLM errors"""

    pass


class LLM:
    # Load prompts at class level
    try:
        from importlib.resources import files

        _image_analysis_prompt = Template(
            files("vision_parse").joinpath("image_analysis.j2").read_text()
        )
        _md_prompt_template = Template(
            files("vision_parse").joinpath("markdown_prompt.j2").read_text()
        )
        _image_summary_prompt = Template(
            files("vision_parse").joinpath("image_summary_prompt.j2").read_text()
        )
        _page_visuals_prompt = Template(
            files("vision_parse").joinpath("page_visuals_prompt.j2").read_text()
        )
    except Exception as e:
        raise FileNotFoundError(f"Failed to load prompt files: {str(e)}")

    def __init__(
        self,
        model_name: str,
        api_key: Union[str, None],
        temperature: float,
        top_p: float,
        ollama_config: Union[Dict, None],
        openai_config: Union[Dict, None],
        gemini_config: Union[Dict, None],
        groq_config: Union[Dict, None],
        vertex_config: Union[Dict, None],
        image_mode: Literal["url", "base64", None],
        custom_prompt: Union[str, None],
        detailed_extraction: bool,
        enable_concurrency: bool,
        device: Literal["cuda", "mps", None],
        num_workers: int,
        enable_image_summary: bool = True,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.ollama_config = ollama_config or {}
        self.openai_config = openai_config or {}
        self.gemini_config = gemini_config or {}
        self.groq_config = groq_config or {}
        self.vertex_config = vertex_config or {}
        self.temperature = temperature
        self.top_p = top_p
        self.image_mode = image_mode
        self.custom_prompt = custom_prompt
        self.detailed_extraction = detailed_extraction
        self.enable_image_summary = enable_image_summary
        self.kwargs = kwargs
        self.enable_concurrency = enable_concurrency
        self.device = device
        self.num_workers = num_workers

        self.provider = self._get_provider_name(model_name)
        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize the LLM client."""
        if self.provider == "groq":
            try:
                import groq
            except ImportError:
                raise ImportError(
                    "Groq is not installed. Please install it using pip install 'vision-parse[groq]' or pip install groq."
                )
            try:
                if self.enable_concurrency:
                    from groq import AsyncGroq

                    self.aclient = AsyncGroq(
                        api_key=self.api_key,
                        timeout=self.groq_config.get("GROQ_TIMEOUT", 240.0),
                        max_retries=self.groq_config.get("GROQ_MAX_RETRIES", 3),
                        base_url=self.groq_config.get("GROQ_BASE_URL", None),
                    )
                else:
                    self.client = groq.Groq(
                        api_key=self.api_key,
                        timeout=self.groq_config.get("GROQ_TIMEOUT", 240.0),
                        max_retries=self.groq_config.get("GROQ_MAX_RETRIES", 3),
                        base_url=self.groq_config.get("GROQ_BASE_URL", None),
                    )
            except Exception as e:
                raise LLMError(f"Unable to initialize Groq client: {str(e)}")

        elif self.provider == "vertex":
            try:
                import vertexai
                from vertexai.generative_models import GenerationConfig
                from google.oauth2 import credentials
            except ImportError:
                raise ImportError(
                    "Vertex AI dependencies are not installed. Please install them using pip install 'vision-parse[vertex]' or pip install google-cloud-aiplatform vertexai."
                )
            
            try:
                # Initialize Vertex AI with provided credentials and project details
                project_id = self.vertex_config.get("PROJECT_ID")
                location = self.vertex_config.get("LOCATION", "us-central1")
                
                # Use API key as OAuth token for credentials
                if self.api_key:  # If API key is provided, use it as OAuth token
                    creds = credentials.Credentials(token=self.api_key)
                    vertexai.init(project=project_id, location=location, credentials=creds)
                else:
                    # Use default credentials from environment
                    vertexai.init(project=project_id, location=location)
                
                # Store GenerationConfig for later use
                self.generation_config = GenerationConfig
                
            except Exception as e:
                raise LLMError(f"Unable to initialize Vertex AI client: {str(e)}")
                
        elif self.provider == "ollama":
            import ollama

            try:
                ollama.show(self.model_name)
            except ollama.ResponseError as e:
                if e.status_code == 404:
                    current_digest, bars = "", {}
                    for progress in ollama.pull(self.model_name, stream=True):
                        digest = progress.get("digest", "")
                        if digest != current_digest and current_digest in bars:
                            bars[current_digest].close()

                        if not digest:
                            logger.info(progress.get("status"))
                            continue

                        if digest not in bars and (total := progress.get("total")):
                            bars[digest] = tqdm(
                                total=total,
                                desc=f"pulling {digest[7:19]}",
                                unit="B",
                                unit_scale=True,
                            )

                        if completed := progress.get("completed"):
                            bars[digest].update(completed - bars[digest].n)

                        current_digest = digest
            except Exception as e:
                raise LLMError(
                    f"Unable to download {self.model_name} from Ollama: {str(e)}"
                )

            try:
                os.environ["OLLAMA_KEEP_ALIVE"] = str(
                    self.ollama_config.get("OLLAMA_KEEP_ALIVE", -1)
                )
                if self.enable_concurrency:
                    self.aclient = ollama.AsyncClient(
                        host=self.ollama_config.get(
                            "OLLAMA_HOST", "http://localhost:11434"
                        ),
                        timeout=self.ollama_config.get("OLLAMA_REQUEST_TIMEOUT", 240.0),
                    )
                    if self.device == "cuda":
                        os.environ["OLLAMA_NUM_GPU"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_GPU", self.num_workers // 2
                            )
                        )
                        os.environ["OLLAMA_NUM_PARALLEL"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_PARALLEL", self.num_workers * 8
                            )
                        )
                        os.environ["OLLAMA_GPU_LAYERS"] = str(
                            self.ollama_config.get("OLLAMA_GPU_LAYERS", "all")
                        )
                    elif self.device == "mps":
                        os.environ["OLLAMA_NUM_GPU"] = str(
                            self.ollama_config.get("OLLAMA_NUM_GPU", 1)
                        )
                        os.environ["OLLAMA_NUM_THREAD"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_THREAD", self.num_workers
                            )
                        )
                        os.environ["OLLAMA_NUM_PARALLEL"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_PARALLEL", self.num_workers * 8
                            )
                        )
                    else:
                        os.environ["OLLAMA_NUM_THREAD"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_THREAD", self.num_workers
                            )
                        )
                        os.environ["OLLAMA_NUM_PARALLEL"] = str(
                            self.ollama_config.get(
                                "OLLAMA_NUM_PARALLEL", self.num_workers * 10
                            )
                        )
                else:
                    self.client = ollama.Client(
                        host=self.ollama_config.get(
                            "OLLAMA_HOST", "http://localhost:11434"
                        ),
                        timeout=self.ollama_config.get("OLLAMA_REQUEST_TIMEOUT", 240.0),
                    )
            except Exception as e:
                raise LLMError(f"Unable to initialize Ollama client: {str(e)}")

        elif self.provider == "openai" or self.provider == "deepseek":
            #  support azure openai
            if self.provider == "openai" and self.openai_config.get(
                "AZURE_OPENAI_API_KEY"
            ):
                try:
                    import openai
                    from openai import AzureOpenAI, AsyncAzureOpenAI
                except ImportError:
                    raise ImportError(
                        "OpenAI is not installed. Please install it using pip install 'vision-parse[openai]'."
                    )

                try:
                    azure_subscription_key = self.openai_config.get(
                        "AZURE_OPENAI_API_KEY"
                    )
                    azure_endpoint_url = self.openai_config.get("AZURE_ENDPOINT_URL")

                    if not azure_endpoint_url or not azure_subscription_key:
                        raise LLMError(
                            "Set `AZURE_ENDPOINT_URL` and `AZURE_OPENAI_API_KEY` environment variables in `openai_config` parameter"
                        )

                    if self.openai_config.get("AZURE_DEPLOYMENT_NAME"):
                        self.model_name = self.openai_config.get(
                            "AZURE_DEPLOYMENT_NAME"
                        )

                    api_version = self.openai_config.get(
                        "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
                    )

                    # Initialize Azure OpenAI client with key-based authentication
                    if self.enable_concurrency:
                        self.aclient = AsyncAzureOpenAI(
                            azure_endpoint=azure_endpoint_url,
                            api_key=azure_subscription_key,
                            api_version=api_version,
                        )
                    else:
                        self.client = AzureOpenAI(
                            azure_endpoint=azure_endpoint_url,
                            api_key=azure_subscription_key,
                            api_version=api_version,
                        )

                except openai.OpenAIError as e:
                    raise LLMError(
                        f"Unable to initialize Azure OpenAI client: {str(e)}"
                    )

            else:
                try:
                    import openai
                except ImportError:
                    raise ImportError(
                        "OpenAI is not installed. Please install it using pip install 'vision-parse[openai]'."
                    )
                try:
                    if self.enable_concurrency:
                        self.aclient = openai.AsyncOpenAI(
                            api_key=self.api_key,
                            base_url=(
                                self.openai_config.get("OPENAI_BASE_URL", None)
                                if self.provider == "openai"
                                else "https://api.deepseek.com"
                            ),
                            max_retries=self.openai_config.get("OPENAI_MAX_RETRIES", 3),
                            timeout=self.openai_config.get("OPENAI_TIMEOUT", 240.0),
                            default_headers=self.openai_config.get(
                                "OPENAI_DEFAULT_HEADERS", None
                            ),
                        )
                    else:
                        self.client = openai.OpenAI(
                            api_key=self.api_key,
                            base_url=(
                                self.openai_config.get("OPENAI_BASE_URL", None)
                                if self.provider == "openai"
                                else "https://api.deepseek.com"
                            ),
                            max_retries=self.openai_config.get("OPENAI_MAX_RETRIES", 3),
                            timeout=self.openai_config.get("OPENAI_TIMEOUT", 240.0),
                            default_headers=self.openai_config.get(
                                "OPENAI_DEFAULT_HEADERS", None
                            ),
                        )
                except openai.OpenAIError as e:
                    raise LLMError(f"Unable to initialize OpenAI client: {str(e)}")

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "Gemini is not installed. Please install it using pip install 'vision-parse[gemini]'."
                )

            try:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(model_name=self.model_name)
                self.generation_config = genai.GenerationConfig
            except Exception as e:
                raise LLMError(f"Unable to initialize Gemini client: {str(e)}")

    def _get_provider_name(self, model_name: str) -> str:
        """Get the provider name for a given model name."""
        try:
            return SUPPORTED_MODELS[model_name]
        except KeyError:
            supported_models = ", ".join(
                f"'{model}' from {provider}"
                for model, provider in SUPPORTED_MODELS.items()
            )
            raise UnsupportedModelError(
                f"Model '{model_name}' is not supported. "
                f"Supported models are: {supported_models}"
            )

    async def _get_response(
        self, base64_encoded: str, prompt: str, structured: bool = False
    ):
        if self.provider == "ollama":
            return await self._ollama(base64_encoded, prompt, structured)
        elif self.provider == "openai" or self.provider == "deepseek":
            return await self._openai(base64_encoded, prompt, structured)
        elif self.provider == "gemini":
            return await self._gemini(base64_encoded, prompt, structured)
        elif self.provider == "groq":
            return await self._groq(base64_encoded, prompt, structured)
        elif self.provider == "vertex":
            return await self._vertex(base64_encoded, prompt, structured)

    async def generate_image_summary(
        self, base64_encoded: str, page_context: str = None
    ) -> str:
        """Generate a summary description of an image using the appropriate LLM model.
        
        Args:
            base64_encoded: Base64 encoded image data
            page_context: Optional text context from the page containing the image
            
        Returns:
            A concise summary of the image content
        """
        try:
            # Generate the prompt with optional page context
            prompt = self._image_summary_prompt.render(page_context=page_context)
            
            # Use existing response method to get summary from LLM
            summary = await self._get_response(base64_encoded, prompt)
            
            # Clean up response - remove any markdown formatting that might have been added
            summary = re.sub(r"```.*?```", "", summary, flags=re.DOTALL).strip()
            
            # Truncate if too long
            if len(summary) > 200:
                summary = summary[:197] + "..."
                
            return summary
        
        except Exception as e:
            logger.warning(f"Failed to generate image summary: {str(e)}")
            return "[Image summary unavailable]"
            
    async def detect_page_visuals(
        self, base64_encoded: str, custom_prompt: str = None
    ) -> str:
        """Detect and summarize visual elements (images, diagrams, charts, etc.) on an entire page.
        
        Args:
            base64_encoded: Base64 encoded image of the entire page
            custom_prompt: Optional custom prompt to override default
            
        Returns:
            A formatted summary of all visual elements detected on the page
        """
        try:
            # Generate the prompt with optional custom instructions
            prompt = self._page_visuals_prompt.render(custom_prompt=custom_prompt)
            
            # Use existing response method to get analysis from LLM
            visuals_summary = await self._get_response(base64_encoded, prompt)
            
            return visuals_summary
        
        except Exception as e:
            logger.warning(f"Failed to detect page visuals: {str(e)}")
            return ""
    
    async def generate_markdown(
        self, base64_encoded: str, pix: fitz.Pixmap, page_number: int
    ) -> Any:
        """Generate markdown formatted text from a base64-encoded image using appropriate model provider."""
        extracted_images = []
        if self.detailed_extraction:
            try:
                response = await self._get_response(
                    base64_encoded,
                    self._image_analysis_prompt.render(),
                    structured=True,
                )

                json_response = ImageDescription.model_validate_json(response)

                if json_response.text_detected.strip() == "No":
                    return ""

                if (
                    self.provider == "ollama"
                    and float(json_response.confidence_score_text) > 0.6
                    and json_response.tables_detected.strip() == "No"
                    and json_response.latex_equations_detected.strip() == "No"
                    and (
                        json_response.images_detected.strip() == "No"
                        or self.image_mode is None
                    )
                ):
                    return json_response.extracted_text

                # Skip individual image extraction since we're using whole page analysis now
                # Only set an empty list to avoid errors in code that may expect this attribute
                pix._extracted_images = []

                prompt = self._md_prompt_template.render(
                    extracted_text=json_response.extracted_text,
                    tables_detected=json_response.tables_detected,
                    latex_equations_detected=json_response.latex_equations_detected,
                    confidence_score_text=float(json_response.confidence_score_text),
                    custom_prompt=self.custom_prompt,
                )

            except Exception:
                logger.warning(
                    "Detailed extraction failed. Falling back to simple extraction."
                )
                self.detailed_extraction = False

        if not self.detailed_extraction:
            prompt = self._md_prompt_template.render(
                extracted_text="",
                tables_detected="Yes",
                latex_equations_detected="No",
                confidence_score_text=0.0,
                custom_prompt=self.custom_prompt,
            )

        markdown_content = await self._get_response(
            base64_encoded, prompt, structured=False
        )

        # Remove image embedding since we're using page-level visual analysis now
        return markdown_content

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _vertex(
        self, base64_encoded: str, prompt: str, structured: bool = False
    ) -> Any:
        """Process base64-encoded image through Vertex AI vision models."""
        try:
            # Import required libraries within the method to avoid dependency issues
            from vertexai.generative_models import GenerativeModel, Part
            import asyncio
            
            # Create model instance
            model = GenerativeModel(model_name=self.model_name)
            
            # Create content parts
            content = []
            
            # Add prompt text
            content.append(prompt)
            
            # Add image part
            image_part = Part.from_data(mimetype="image/jpeg", data=base64.b64decode(base64_encoded))
            content.append(image_part)
            
            # Set generation parameters
            if structured:
                # For structured data extraction, use more conservative parameters
                temperature = 0.0
                top_p = 0.4
            else:
                temperature = self.temperature
                top_p = self.top_p
            
            # Handle both sync and async operations
            if self.enable_concurrency:
                # In async mode, use coroutines directly
                response = await asyncio.to_thread(
                    model.generate_content,
                    content,
                    generation_config=self.generation_config(
                        temperature=temperature,
                        top_p=top_p,
                        **self.kwargs
                    ),
                )
            else:
                # In sync mode
                response = model.generate_content(
                    content,
                    generation_config=self.generation_config(
                        temperature=temperature,
                        top_p=top_p,
                        **self.kwargs
                    ),
                )
            
            # Extract and clean response text
            return re.sub(
                r"```(?:markdown)?\n(.*?)\n```", r"\1", response.text, flags=re.DOTALL
            )
            
        except Exception as e:
            error_msg = str(e)
            if "exceeds the maximum content length" in error_msg.lower():
                # Handle context length errors
                raise LLMError(
                    f"Image too complex for Vertex AI model context window: {error_msg}"
                )
            elif "image size" in error_msg.lower() or "exceeds allowed dimensions" in error_msg.lower():
                # Handle image size errors
                raise LLMError(
                    f"Image exceeds Vertex AI size limits. Please use a lower DPI setting in page_config: {error_msg}"
                )
            else:
                raise LLMError(f"Error processing image with Vertex AI: {error_msg}")
                
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _groq(
        self, base64_encoded: str, prompt: str, structured: bool = False
    ) -> Any:
        """Process base64-encoded image through Groq vision models."""
        try:
            # Basic content with text prompt
            content = [{"type": "text", "text": prompt}]
            
            # Handle image based on image_mode
            if self.image_mode == "url":
                # If image_mode is URL, we need to have created a URL for the image
                # This assumes extracted_images logic has created a URL
                logger.info("Using URL mode for Groq API is not currently supported in direct page processing")
                # Fall back to base64 mode
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_encoded}",
                    },
                }
            else:  # base64 or None (default to base64)
                # Use base64 data URL format
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_encoded}",
                    },
                }
            
            # Add image to content
            content.append(image_content)

            if self.enable_concurrency:
                response = await self.aclient.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.groq_config.get("GROQ_MAX_TOKENS", 4096),
                    response_format={"type": "json_object"} if structured else None,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.groq_config.get("GROQ_MAX_TOKENS", 4096),
                    response_format={"type": "json_object"} if structured else None,
                )

            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            if "maximum context length" in error_msg.lower():
                # Handle context length errors
                raise LLMError(
                    f"Image too complex for Groq model context window: {error_msg}"
                )
            elif (
                "image too large" in error_msg.lower() or "pixels" in error_msg.lower()
            ):
                # Handle image size errors
                raise LLMError(
                    f"Image exceeds Groq's size limit (max 33,177,600 pixels). Please use a lower DPI setting in page_config: {error_msg}"
                )
            else:
                raise LLMError(f"Error processing image with Groq API: {error_msg}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _ollama(
        self, base64_encoded: str, prompt: str, structured: bool = False
    ) -> Any:
        """Process base64-encoded image through Ollama vision models."""
        try:
            if self.enable_concurrency:
                response = await self.aclient.chat(
                    model=self.model_name,
                    format=ImageDescription.model_json_schema() if structured else None,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [base64_encoded],
                        }
                    ],
                    options={
                        "temperature": 0.0 if structured else self.temperature,
                        "top_p": 0.4 if structured else self.top_p,
                        **self.kwargs,
                    },
                    keep_alive=-1,
                )
            else:
                response = self.client.chat(
                    model=self.model_name,
                    format=ImageDescription.model_json_schema() if structured else None,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [base64_encoded],
                        }
                    ],
                    options={
                        "temperature": 0.0 if structured else self.temperature,
                        "top_p": 0.4 if structured else self.top_p,
                        **self.kwargs,
                    },
                    keep_alive=-1,
                )

            return re.sub(
                r"```(?:markdown)?\n(.*?)\n```",
                r"\1",
                response["message"]["content"],
                flags=re.DOTALL,
            )
        except Exception as e:
            raise LLMError(f"Ollama Model processing failed: {str(e)}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _openai(
        self, base64_encoded: str, prompt: str, structured: bool = False
    ) -> Any:
        """Process base64-encoded image through OpenAI vision models."""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_encoded}"
                            },
                        },
                    ],
                }
            ]

            if self.enable_concurrency:
                if structured:
                    if os.getenv("AZURE_OPENAI_API_KEY") or self.provider == "deepseek":
                        response = await self.aclient.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=0.0,
                            top_p=0.4,
                            response_format={"type": "json_object"},
                            stream=False,
                            **self.kwargs,
                        )
                    else:
                        response = await self.aclient.beta.chat.completions.parse(
                            model=self.model_name,
                            response_format=ImageDescription,
                            messages=messages,
                            temperature=0.0,
                            top_p=0.4,
                            **self.kwargs,
                        )
                    return response.choices[0].message.content

                response = await self.aclient.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False,
                    **self.kwargs,
                )
            else:
                if structured:
                    if os.getenv("AZURE_OPENAI_API_KEY") or self.provider == "deepseek":
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=0.0,
                            top_p=0.4,
                            response_format={"type": "json_object"},
                            stream=False,
                            **self.kwargs,
                        )
                    else:
                        response = self.client.beta.chat.completions.parse(
                            model=self.model_name,
                            response_format=ImageDescription,
                            messages=messages,
                            temperature=0.0,
                            top_p=0.4,
                            **self.kwargs,
                        )
                    return response.choices[0].message.content

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=False,
                    **self.kwargs,
                )

            return re.sub(
                r"```(?:markdown)?\n(.*?)\n```",
                r"\1",
                response.choices[0].message.content,
                flags=re.DOTALL,
            )
        except Exception as e:
            raise LLMError(f"OpenAI Model processing failed: {str(e)}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def _gemini(
        self, base64_encoded: str, prompt: str, structured: bool = False
    ) -> Any:
        """Process base64-encoded image through Gemini vision models."""
        try:
            if self.enable_concurrency:
                response = await self.client.generate_content_async(
                    [{"mime_type": "image/png", "data": base64_encoded}, prompt],
                    generation_config=self.generation_config(
                        response_mime_type="application/json" if structured else None,
                        response_schema=ImageDescription if structured else None,
                        temperature=0.0 if structured else self.temperature,
                        top_p=0.4 if structured else self.top_p,
                        **self.kwargs,
                    ),
                )
            else:
                response = self.client.generate_content(
                    [{"mime_type": "image/png", "data": base64_encoded}, prompt],
                    generation_config=self.generation_config(
                        response_mime_type="application/json" if structured else None,
                        response_schema=ImageDescription if structured else None,
                        temperature=0.0 if structured else self.temperature,
                        top_p=0.4 if structured else self.top_p,
                        **self.kwargs,
                    ),
                )

            return re.sub(
                r"```(?:markdown)?\n(.*?)\n```", r"\1", response.text, flags=re.DOTALL
            )
        except Exception as e:
            raise LLMError(f"Gemini Model processing failed: {str(e)}")
