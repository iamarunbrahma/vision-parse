#!/usr/bin/env python

"""
This example demonstrates how to use vision-parse with Vertex AI to convert a PDF to markdown.

Before running this example, make sure you:
1. Have installed vision-parse with Vertex AI dependencies: 
   pip install 'vision-parse[vertex]'

2. Have set up Google Cloud Project with Vertex AI API enabled

3. Have proper authentication for Vertex AI:
   - Using OAuth token via api_key parameter
   - Or using default credentials from environment variables

For testing with sample data, use the sample PDF in the examples directory.
"""

import os
import sys
from pathlib import Path
import logging

# Add parent directory to path to import vision_parse if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_parse import VisionParser, PDFPageConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Path to sample PDF (you can replace with your own)
    sample_pdf = Path(__file__).parent / "sample.pdf"
    
    # If sample PDF doesn't exist, display an error
    if not sample_pdf.exists():
        logger.error(f"Sample PDF not found at {sample_pdf}. Please add a PDF file named 'sample.pdf'.")
        return

    # Create the configuration for Vertex AI
    vertex_config = {
        "PROJECT_ID": os.environ.get("GOOGLE_CLOUD_PROJECT"),  # Your GCP Project ID
        "LOCATION": "us-central1",  # API region (e.g., us-central1)
    }
    
    # Initialize the parser with Vertex AI Gemini model
    # Using lower DPI (200) to prevent exceeding Vertex AI image size limits
    parser = VisionParser(
        page_config=PDFPageConfig(
            dpi=200,              # Lower DPI to avoid size limits
            color_space="rgb",    # Use RGB for better image recognition
            # Additional options:
            # preserve_transparency=False,
            # include_annotations=True,
        ),
        model_name="gemini-1.5-pro-002",  # Vertex AI model name
        api_key=os.environ.get("VERTEX_API_TOKEN"),  # OAuth token for authentication
        temperature=0.2,          # Lower temperature for more deterministic output
        top_p=0.9,                # Nucleus sampling parameter
        vertex_config=vertex_config,
        enable_concurrency=True,  # Process pages concurrently
        image_mode=None,          # Don't extract embedded images
        custom_prompt=None,       # Use default markdown conversion prompt
    )
    
    # Convert the PDF to markdown
    markdown_pages = parser.convert_pdf(sample_pdf)
    
    # Write the markdown to a file
    output_path = sample_pdf.with_suffix(".md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(markdown_pages))
    
    logger.info(f"PDF converted successfully to {output_path}")

if __name__ == "__main__":
    main()
