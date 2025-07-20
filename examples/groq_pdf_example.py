"""
Example demonstrating how to convert PDF to Markdown using Groq Vision API.
"""

import os
from vision_parse import VisionParser, PDFPageConfig

# Check if the API key is available
if "GROQ_API_KEY" not in os.environ:
    raise ValueError(
        "GROQ_API_KEY environment variable not set. "
        "Please set it with your Groq API key."
    )

# Create a page configuration with lower DPI to stay within Groq's pixel limits
# Groq has a max limit of 33,177,600 pixels per image
page_config = PDFPageConfig(
    dpi=200,  # Lower DPI (default is 400) to reduce image size
    color_space="RGB",
    include_annotations=True,
    preserve_transparency=False,
)

# Initialize parser with Groq configuration
parser = VisionParser(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",  # Groq vision model
    api_key=os.environ.get("GROQ_API_KEY"),  # Groq API key from environment
    temperature=0.4,
    top_p=0.5,
    page_config=page_config,  # Use the lower DPI config
    groq_config={
        "GROQ_MAX_TOKENS": 4096,
        "GROQ_TIMEOUT": 300.0,
    },
    image_mode="base64",  # Image mode can be "url", "base64" or None
    detailed_extraction=True,  # Set to True for more detailed extraction
    enable_concurrency=True,  # Set to True for parallel processing
)

# Convert PDF to markdown
pdf_path = "test.pdf"  # local path to your pdf file
markdown_pages = parser.convert_pdf(pdf_path)

# Process results
for i, page_content in enumerate(markdown_pages):
    print(f"\n--- Page {i+1} ---\n{page_content}")

    # Optionally save each page as a separate markdown file
    with open(f"output_page_{i+1}.md", "w", encoding="utf-8") as f:
        f.write(page_content)

# Optionally combine all pages into a single markdown file
with open("output_combined.md", "w", encoding="utf-8") as f:
    f.write("\n\n".join(markdown_pages))

print(f"Converted {len(markdown_pages)} pages to markdown.")
