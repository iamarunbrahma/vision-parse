import fitz  # PyMuPDF library for PDF processing
from pathlib import Path
from typing import Optional, List, Dict, Union, Literal, Any
from tqdm import tqdm
import base64
from pydantic import BaseModel
import asyncio
from .utils import get_device_config
from .llm import LLM
import nest_asyncio
import logging
import warnings

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
        groq_config: Optional[Dict] = None,
        vertex_config: Optional[Dict] = None,
        image_mode: Literal["url", "base64", None] = None,
        custom_prompt: Optional[str] = None,
        detailed_extraction: bool = False,
        extraction_complexity: bool = False,  # Deprecated Parameter
        enable_concurrency: bool = False,
        enable_image_summary: bool = True,
        **kwargs: Any,
    ):
        """Initialize parser with PDFPageConfig and LLM configuration."""
        self.page_config = page_config or PDFPageConfig()
        self.device, self.num_workers = get_device_config()
        self.enable_concurrency = enable_concurrency

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
            groq_config=groq_config,
            vertex_config=vertex_config,
            image_mode=image_mode,
            detailed_extraction=detailed_extraction,
            custom_prompt=custom_prompt,
            enable_concurrency=enable_concurrency,
            device=self.device,
            num_workers=self.num_workers,
            enable_image_summary=enable_image_summary,
            **kwargs,
        )

    def _calculate_matrix(self, page: fitz.Page) -> fitz.Matrix:
        """Calculate transformation matrix for page conversion."""
        # Calculate zoom factor based on target DPI
        zoom = self.page_config.dpi / 72
        matrix = fitz.Matrix(zoom * 2, zoom * 2)

        # Handle page rotation if present
        if page.rotation != 0:
            matrix.prerotate(page.rotation)

        return matrix

    async def _convert_page(self, page: fitz.Page, page_number: int) -> str:
        """Convert a single PDF page into base64-encoded PNG and extract markdown formatted text."""
        pix = None
        try:
            matrix = self._calculate_matrix(page)

            # Create high-quality image from PDF page
            pix = page.get_pixmap(
                matrix=matrix,
                alpha=self.page_config.preserve_transparency,
                colorspace=self.page_config.color_space,
                annots=self.page_config.include_annotations,
            )

            # Convert image to base64 for LLM processing
            base64_encoded = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            
            # First, generate the basic markdown content from the page
            markdown = await self.llm.generate_markdown(base64_encoded, pix, page_number)
            
            # If detailed extraction and image summary are enabled, detect and summarize visual elements on the page
            if (hasattr(self.llm, 'detailed_extraction') and self.llm.detailed_extraction and 
                hasattr(self.llm, 'enable_image_summary') and self.llm.enable_image_summary):
                # Custom prompt that includes information about the page content
                custom_prompt = f"""Analyze this document page and identify any visual elements like images, diagrams, charts, 
                flowcharts, or visualizations. For each visual element, provide a brief description of what it shows 
                and its significance. Note that the page contains the following text content for context:
                
                {markdown[:300]}{'...' if len(markdown) > 300 else ''}
                
                Only describe visual elements that are not already properly captured in the text transcription above.
                If no significant visual elements are present, respond with an empty string."""
                
                # Get visual elements summary
                visuals_summary = await self.llm.detect_page_visuals(base64_encoded, custom_prompt)
                
                # Only append if there's actual visual content detected
                if visuals_summary and "No visual elements detected" not in visuals_summary:
                    # Add visual elements summary to the markdown
                    markdown += f"\n\n{visuals_summary}"
                    
                    # If image mode is set to include the actual images
                    if self.llm.image_mode in ["base64", "url"]:
                        # Add the full page image at the end if there are visual elements
                        if self.llm.image_mode == "base64":
                            # Use base64 encoding for the image
                            img_reference = f"\n\n![Page {page_number + 1}](data:image/png;base64,{base64_encoded})\n\n"
                            markdown += img_reference
                        elif self.llm.image_mode == "url":
                            # Save the image to a file and use the URL
                            img_path = f"page_{page_number + 1}.png"
                            pix.save(img_path)
                            markdown += f"\n\n![Page {page_number + 1}]({img_path})\n\n"
            
            return markdown

        except Exception as e:
            raise VisionParserError(
                f"Failed to convert page {page_number + 1} to base64-encoded PNG: {str(e)}"
            )
        finally:
            # Clean up pixmap to free memory
            if pix is not None:
                pix = None

    async def _convert_pages_batch(self, pages: List[fitz.Page], start_idx: int):
        """Process a batch of PDF pages concurrently."""
        try:
            tasks = []
            for i, page in enumerate(pages):
                tasks.append(self._convert_page(page, start_idx + i))
            return await asyncio.gather(*tasks)
        finally:
            await asyncio.sleep(0.5)

    def convert_pdf(self, pdf_path: Union[str, Path]) -> List[str]:
        """Convert all pages in the given PDF file to markdown text."""
        pdf_path = Path(pdf_path)
        converted_pages = []

        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise UnsupportedFileError(f"File is not a PDF: {pdf_path}")

        try:
            with fitz.open(pdf_path) as pdf_document:
                total_pages = pdf_document.page_count

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
                                pdf_document[j] for j in range(i, i + batch_size)
                            ]
                            batch_results = asyncio.run(
                                self._convert_pages_batch(batch_pages, i)
                            )
                            converted_pages.extend(batch_results)
                            pbar.update(len(batch_results))
                    else:
                        for page_number in range(total_pages):
                            # For non-concurrent processing, still need to run async code
                            text = asyncio.run(
                                self._convert_page(
                                    pdf_document[page_number], page_number
                                )
                            )
                            converted_pages.append(text)
                            pbar.update(1)

                return converted_pages

        except Exception as e:
            raise VisionParserError(
                f"Failed to convert PDF file into markdown content: {str(e)}"
            )
