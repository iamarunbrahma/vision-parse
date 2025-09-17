from typing import Literal, Dict, Any, Union
from pydantic import BaseModel
from jinja2 import Template
import re
import pypdfium2 as pdfium
import os
import base64
from tqdm import tqdm
from .utils import ImageData
from tenacity import retry, stop_after_attempt, wait_exponential
from .constants import SUPPORTED_MODELS, discover_ollama_vision_models
import logging

try:
    import vllm as _vllm_module
except ImportError:  # pragma: no cover - optional dependency
    _vllm_module = None
else:
    if not hasattr(_vllm_module, "AsyncOpenAI") or not hasattr(
        _vllm_module, "OpenAI"
    ):
        try:
            from openai import AsyncOpenAI as _OpenAIAsyncClient, OpenAI as _OpenAIClient
        except ImportError:  # pragma: no cover - optional dependency
            _OpenAIAsyncClient = None
            _OpenAIClient = None
        else:
            if not hasattr(_vllm_module, "AsyncOpenAI") and _OpenAIAsyncClient is not None:
                setattr(_vllm_module, "AsyncOpenAI", _OpenAIAsyncClient)
            if not hasattr(_vllm_module, "OpenAI") and _OpenAIClient is not None:
                setattr(_vllm_module, "OpenAI", _OpenAIClient)

_logger = logging.getLogger(__name__)


class ImageDescription(BaseModel):
    text_detected: Literal["Yes", "No"]
    tables_detected: Literal["Yes", "No"]
    images_detected: Literal["Yes", "No"]
    latex_equations_detected: Literal["Yes", "No"]
    extracted_text: str
    confidence_score_text: float


class UnsupportedModelError(Exception):
    pass


class LLMError(Exception):
    pass


class LLM:
    try:
        from importlib.resources import files

        _IMAGE_ANALYSIS_PROMPT = Template(
            files("vision_parse").joinpath("image_analysis.j2").read_text()
        )
        _MD_PROMPT_TEMPLATE = Template(
            files("vision_parse").joinpath("markdown_prompt.j2").read_text()
        )
    except Exception as e:
        raise FileNotFoundError(f"Failed to load prompt files: {str(e)}")

    # Accessors for class-level Jinja2 templates to keep instance usage consistent
    @property
    def _image_analysis_prompt(self) -> Template:
        return self._IMAGE_ANALYSIS_PROMPT

    @property
    def _md_prompt_template(self) -> Template:
        return self._MD_PROMPT_TEMPLATE

    def __init__(
        self,
        model_name: str,
        api_key: Union[str, None],
        temperature: float,
        top_p: float,
        ollama_config: Union[Dict, None],
        openai_config: Union[Dict, None],
        gemini_config: Union[Dict, None],
        image_mode: Literal["url", "base64", None],
        custom_prompt: Union[str, None],
        detailed_extraction: bool,
        enable_concurrency: bool,
        device: Literal["cuda", "mps", None],
        num_workers: int,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.ollama_config = ollama_config or {}
        self.openai_config = openai_config or {}
        self.gemini_config = gemini_config or {}
        self.temperature = temperature
        self.top_p = top_p
        self.image_mode = image_mode
        self.custom_prompt = custom_prompt
        self.detailed_extraction = detailed_extraction
        self.kwargs = kwargs
        self.enable_concurrency = enable_concurrency
        self.device = device
        self.num_workers = num_workers

        self.provider = self._get_provider_name(model_name)
        self._init_llm()

    def _init_llm(self) -> None:
        """Initialize the LLM client."""
        if self.provider == "ollama":
            import ollama

            host = self.ollama_config.get("OLLAMA_HOST", "http://localhost:11434")
            timeout = self.ollama_config.get("OLLAMA_REQUEST_TIMEOUT", 240.0)

            self.client = ollama.Client(host=host, timeout=timeout, trust_env=False)

            if self.enable_concurrency:
                self.aclient = ollama.AsyncClient(
                    host=host, timeout=timeout, trust_env=False
                )

            try:
                self.client.show(self.model_name)
            except ollama.ResponseError as e:
                if e.status_code == 404:
                    current_digest, bars = "", {}
                    for progress in self.client.pull(self.model_name, stream=True):
                        digest = progress.get("digest", "")
                        if digest != current_digest and current_digest in bars:
                            bars[current_digest].close()

                        if not digest:
                            _logger.info(progress.get("status"))
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
                error_message = str(e)
                if "Failed to connect to Ollama" in error_message:
                    _logger.warning(
                        "Unable to reach Ollama host '%s'. Continuing without model verification: %s",
                        host,
                        error_message,
                    )
                else:
                    raise LLMError(
                        f"Unable to download {self.model_name} from Ollama: {error_message}"
                    )

            try:
                os.environ["OLLAMA_KEEP_ALIVE"] = str(
                    self.ollama_config.get("OLLAMA_KEEP_ALIVE", -1)
                )
                if self.enable_concurrency:
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
            except Exception as e:
                raise LLMError(f"Unable to initialize Ollama client: {str(e)}")

        elif self.provider in {"openai", "vllm"}:
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
                openai_module = None
                async_client_cls = None
                client_cls = None
                error_cls: Any = Exception

                if self.provider == "openai":
                    try:
                        import openai as openai_module
                    except ImportError as exc:
                        raise ImportError(
                            "OpenAI is not installed. Please install it using pip install 'vision-parse[openai]'."
                        ) from exc

                    async_client_cls = openai_module.AsyncOpenAI
                    client_cls = openai_module.OpenAI
                    error_cls = openai_module.OpenAIError
                else:
                    vllm_module = _vllm_module
                    if vllm_module is None:
                        try:
                            import vllm as vllm_module  # type: ignore[no-redef]
                        except ImportError as exc:
                            raise ImportError(
                                "vLLM is not installed. Please install it using pip install 'vision-parse[vllm]'."
                            ) from exc

                    try:
                        import openai as openai_module
                    except ImportError:
                        openai_module = None

                    async_client_cls = getattr(vllm_module, "AsyncOpenAI", None)
                    client_cls = getattr(vllm_module, "OpenAI", None)

                    if async_client_cls is None or client_cls is None:
                        if openai_module is None:
                            missing_parts = []
                            if async_client_cls is None:
                                missing_parts.append("AsyncOpenAI")
                            if client_cls is None:
                                missing_parts.append("OpenAI")
                            missing_msg = " and ".join(missing_parts)
                            raise ImportError(
                                "vLLM does not expose {}. Install the OpenAI Python package via pip install 'vision-parse[openai]'."
                                .format(missing_msg)
                            )

                        if async_client_cls is None:
                            async_client_cls = openai_module.AsyncOpenAI
                            setattr(vllm_module, "AsyncOpenAI", async_client_cls)

                        if client_cls is None:
                            client_cls = openai_module.OpenAI
                            setattr(vllm_module, "OpenAI", client_cls)

                    if openai_module is not None:
                        error_cls = getattr(openai_module, "OpenAIError", Exception)

                try:
                    base_url = self.openai_config.get("OPENAI_BASE_URL")
                    if self.provider == "vllm":
                        base_url = (
                            base_url
                            or os.getenv("VLLM_BASE_URL")
                            or os.getenv("OPENAI_BASE_URL")
                            or "http://localhost:8000/v1"
                        )
                    else:
                        base_url = (
                            base_url
                            or os.getenv("OPENAI_BASE_URL")
                            or "https://api.openai.com"
                        )

                    api_key = self.api_key or self.openai_config.get("OPENAI_API_KEY")
                    if self.provider == "vllm":
                        api_key = (
                            api_key
                            or self.openai_config.get("VLLM_API_KEY")
                            or os.getenv("VLLM_API_KEY")
                            or os.getenv("OPENAI_API_KEY")
                            or "EMPTY"
                        )
                    else:
                        api_key = api_key or os.getenv("OPENAI_API_KEY")

                    client_kwargs = dict(
                        base_url=base_url,
                        max_retries=self.openai_config.get("OPENAI_MAX_RETRIES", 3),
                        timeout=self.openai_config.get("OPENAI_TIMEOUT", 240.0),
                        default_headers=self.openai_config.get(
                            "OPENAI_DEFAULT_HEADERS", None
                        ),
                    )

                    if api_key is not None:
                        client_kwargs["api_key"] = api_key

                    if self.enable_concurrency:
                        if async_client_cls is None:
                            raise LLMError("Async client class is not available for the selected provider.")
                        self.aclient = async_client_cls(**client_kwargs)
                    else:
                        if client_cls is None:
                            raise LLMError("Client class is not available for the selected provider.")
                        self.client = client_cls(**client_kwargs)
                except error_cls as e:  # type: ignore[arg-type]
                    provider_label = "OpenAI" if self.provider == "openai" else "vLLM"
                    raise LLMError(
                        f"Unable to initialize {provider_label} client: {str(e)}"
                    )
                except Exception as e:
                    provider_label = "OpenAI" if self.provider == "openai" else "vLLM"
                    raise LLMError(
                        f"Unable to initialize {provider_label} client: {str(e)}"
                    )

        elif self.provider == "gemini":
            try:
                from google import genai
            except ImportError:
                raise ImportError(
                    "Gemini is not installed. Please install it using pip install 'vision-parse[gemini]'."
                )

            try:
                self._genai = genai
                self.client = self._genai.Client(api_key=self.api_key)
                self.model_name = self.model_name
            except Exception as e:
                raise LLMError(f"Unable to initialize Gemini client: {str(e)}")

    def _get_provider_name(self, model_name: str) -> str:
        """Get the provider name for a given model name."""
        try:
            return SUPPORTED_MODELS[model_name]
        except KeyError:
            dynamic_models = discover_ollama_vision_models()
            if model_name in dynamic_models:
                return dynamic_models[model_name]

            all_models = {**SUPPORTED_MODELS, **dynamic_models}
            supported_models = ", ".join(
                f"'{model}' from {provider}" for model, provider in all_models.items()
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
        elif self.provider in {"openai", "vllm"}:
            return await self._openai(base64_encoded, prompt, structured)
        elif self.provider == "gemini":
            return await self._gemini(base64_encoded, prompt, structured)

    async def generate_markdown(
        self, base64_encoded: str, bitmap: pdfium.PdfBitmap, page_number: int
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

                if (
                    json_response.images_detected.strip() == "Yes"
                    and self.image_mode is not None
                ):
                    extracted_images = ImageData.extract_images(
                        bitmap, self.image_mode, page_number
                    )

                prompt = self._md_prompt_template.render(
                    extracted_text=json_response.extracted_text,
                    tables_detected=json_response.tables_detected,
                    latex_equations_detected=json_response.latex_equations_detected,
                    confidence_score_text=float(json_response.confidence_score_text),
                    custom_prompt=self.custom_prompt,
                )

            except Exception:
                _logger.warning(
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

        if extracted_images:
            if self.image_mode == "url":
                for image_data in extracted_images:
                    markdown_content += (
                        f"\n\n![{image_data.image_url}]({image_data.image_url})"
                    )
            elif self.image_mode == "base64":
                for image_data in extracted_images:
                    markdown_content += (
                        f"\n\n![{image_data.image_url}]({image_data.base64_encoded})"
                    )

        return markdown_content

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
            content_parts = [{"type": "text", "text": prompt}]

            if base64_encoded:
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_encoded}"},
                    }
                )

            messages = [{"role": "user", "content": content_parts}]

            azure_key_present = bool(
                self.openai_config.get("AZURE_OPENAI_API_KEY")
                or os.getenv("AZURE_OPENAI_API_KEY")
            )
            use_pydantic_parse = self.provider == "openai" and not azure_key_present

            if self.enable_concurrency:
                if structured:
                    if use_pydantic_parse:
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
                        temperature=0.0,
                        top_p=0.4,
                        response_format={"type": "json_object"},
                        stream=False,
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
                    if use_pydantic_parse:
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
                        temperature=0.0,
                        top_p=0.4,
                        response_format={"type": "json_object"},
                        stream=False,
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
            provider_label = "OpenAI" if self.provider == "openai" else "vLLM"
            raise LLMError(f"{provider_label} Model processing failed: {str(e)}")

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
            image_bytes = base64.b64decode(base64_encoded)

            if self.enable_concurrency:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=[
                        self._genai.types.Part.from_bytes(
                            data=image_bytes, mime_type="image/png"
                        ),
                        prompt,
                    ],
                    config=self._genai.types.GenerateContentConfig(
                        response_mime_type="application/json" if structured else None,
                        response_schema=ImageDescription if structured else None,
                        temperature=0.0 if structured else self.temperature,
                        top_p=0.4 if structured else self.top_p,
                        **self.kwargs,
                    ),
                )
            else:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[
                        self._genai.types.Part.from_bytes(
                            data=image_bytes, mime_type="image/png"
                        ),
                        prompt,
                    ],
                    config=self._genai.types.GenerateContentConfig(
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
        except self._genai.errors.APIError as e:
            raise LLMError(f"Gemini API error: {str(e)}")
        except Exception as e:
            raise LLMError(f"Gemini Model processing failed: {str(e)}")
