import pypdfium2 as pdfium  # pypdfium2 library for PDF processing
from pathlib import Path
from typing import Optional, List, Dict, Union, Literal, Any
from tqdm import tqdm
import base64
import io
from pydantic import BaseModel
import asyncio
from .utils import get_device_config
from .fast import FastMarkdown
from .llm import LLM
import nest_asyncio
import logging
import warnings
from contextlib import suppress

logger = logging.getLogger(__name__)
nest_asyncio.apply()


class PDFPageConfig(BaseModel):
    """Configuration settings for PDF page conversion."""

    dpi: int = 400  # Resolution for PDF to image conversion
    color_space: str = "RGB"  # Color mode for image output
    include_annotations: bool = True  # Include PDF annotations in conversion
    preserve_transparency: bool = False  # Control alpha channel in output


class UnsupportedFileError(BaseException):
    """Custom exception for handling unsupported file errors."""

    pass


class VisionParserError(BaseException):
    """Custom exception for handling Markdown Parser errors."""

    pass


class VisionParser:
    """Convert PDF pages to base64-encoded images and then extract text from the images in markdown format."""

    def __init__(
        self,
        page_config: Optional[PDFPageConfig] = None,
        model_name: str = "llama3.2-vision:11b",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.7,
        ollama_config: Optional[Dict] = None,
        openai_config: Optional[Dict] = None,
        gemini_config: Optional[Dict] = None,
        image_mode: Literal["url", "base64", None] = None,
        custom_prompt: Optional[str] = None,
        detailed_extraction: bool = False,
        extraction_complexity: bool = False,
        enable_concurrency: bool = False,
        num_workers: Optional[int] = None,
        fast_mode: bool = False,
        **kwargs: Any,
    ):
        """Initialize parser with PDFPageConfig and LLM configuration."""
        self.page_config = page_config or PDFPageConfig()
        self.device, auto_num_workers = get_device_config()
        self.num_workers = num_workers if num_workers is not None else auto_num_workers
        self.enable_concurrency = enable_concurrency
        self.fast_mode = fast_mode

        if extraction_complexity:
            if not detailed_extraction:
                detailed_extraction = True
                warnings.warn(
                    "`extraction_complexity` is deprecated, and was renamed to `detailed_extraction`.",
                    DeprecationWarning,
                )

            else:
                raise ValueError(
                    "`extraction_complexity` is deprecated, and was renamed to `detailed_extraction`. Please use `detailed_extraction` instead."
                )

        self.llm = LLM(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            ollama_config=ollama_config,
            openai_config=openai_config,
            gemini_config=gemini_config,
            image_mode=image_mode,
            detailed_extraction=detailed_extraction,
            custom_prompt=custom_prompt,
            enable_concurrency=enable_concurrency,
            device=self.device,
            num_workers=self.num_workers,
            **kwargs,
        )

    def _calculate_scale_and_rotation(self, page: pdfium.PdfPage) -> tuple[float, int]:
        """Calculate scale factor and rotation for page conversion."""
        # Calculate zoom factor based on target DPI
        zoom = self.page_config.dpi / 72
        scale = zoom * 2

        # Get page rotation
        rotation = page.get_rotation()

        return scale, rotation

    async def _convert_page(self, page: pdfium.PdfPage, page_number: int) -> str:
        """Convert a single PDF page into base64-encoded PNG and extract markdown formatted text."""
        bitmap = None
        try:
            scale, rotation = self._calculate_scale_and_rotation(page)

            # Create high-quality image from PDF page using pypdfium2
            bitmap = page.render(
                scale=scale,
                rotation=rotation,
                rev_byteorder=False,  # Keep BGR format for consistency
                may_draw_forms=self.page_config.include_annotations,
            )

            # Convert bitmap to PIL Image and then to PNG bytes
            pil_image = bitmap.to_pil()

            # Convert PIL image to base64 PNG
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            base64_encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return await self.llm.generate_markdown(base64_encoded, bitmap, page_number)

        except Exception as e:
            raise VisionParserError(
                f"Failed to convert page {page_number + 1} to base64-encoded PNG: {str(e)}"
            )
        finally:
            # Clean up bitmap to free memory
            if bitmap is not None:
                bitmap.close()

    async def _convert_pages_batch(self, pages: List[pdfium.PdfPage], start_idx: int):
        """Process a batch of PDF pages concurrently."""
        try:
            tasks = []
            for i, page in enumerate(pages):
                tasks.append(self._convert_page(page, start_idx + i))
            return await asyncio.gather(*tasks)
        finally:
            await asyncio.sleep(0.5)

    def convert_pdf(self, pdf_path: Union[str, Path]) -> List[str]:
        """Convert all pages in the given PDF file to markdown text.

        If fast_mode is enabled, use the pdfminer/pdfplumber-based fast extractor
        to produce markdown without invoking vision models.
        """
        pdf_path = Path(pdf_path)
        converted_pages = []

        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise UnsupportedFileError(f"File is not a PDF: {pdf_path}")

        # Fast path: use fast extractor and return per-page results
        if self.fast_mode:
            extractor = FastMarkdown(pdf_path)
            combined, pages = extractor.extract()
            # Return list of per-page markdowns
            return pages if pages else ([] if combined == "" else [combined])

        pdf_document = None
        try:
            pdf_document = pdfium.PdfDocument(pdf_path)
            total_pages = len(pdf_document)

            with tqdm(
                total=total_pages,
                desc="Converting pages in PDF file into markdown format",
            ) as pbar:
                if self.enable_concurrency:
                    # Process pages in batches based on num_workers
                    for i in range(0, total_pages, self.num_workers):
                        batch_size = min(self.num_workers, total_pages - i)
                        # Extract only required pages for the batch
                        batch_pages = [
                            pdf_document.get_page(j) for j in range(i, i + batch_size)
                        ]
                        batch_results = asyncio.run(
                            self._convert_pages_batch(batch_pages, i)
                        )
                        converted_pages.extend(batch_results)
                        pbar.update(len(batch_results))
                else:
                    for page_number in range(total_pages):
                        # For non-concurrent processing, still need to run async code
                        page = pdf_document.get_page(page_number)
                        text = asyncio.run(self._convert_page(page, page_number))
                        converted_pages.append(text)
                        pbar.update(1)

            return converted_pages
        except Exception as e:
            raise VisionParserError(
                f"Failed to convert PDF file into markdown content: {str(e)}"
            )
        finally:
            if pdf_document is not None:
                with suppress(Exception):
                    pdf_document.close()
