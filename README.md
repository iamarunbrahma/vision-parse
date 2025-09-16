<div align='center'>

# Vision Parse ✨

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Author: Arun Brahma](https://img.shields.io/badge/Author-Arun%20Brahma-purple)](https://github.com/iamarunbrahma)
[![PyPI version](https://img.shields.io/pypi/v/vision-parse.svg)](https://pypi.org/project/vision-parse/)

> Parse PDF documents into beautifully formatted markdown content using state-of-the-art Vision Language Models - all with just a few lines of code!

[Getting Started](#-getting-started) •
[Usage](#-usage) •
[Supported Models](#-supported-models) •
[Parameters](#-customization-parameters) •
[Benchmarks](#-benchmarks)
</div>

## 🎯 Introduction

Vision Parse harnesses the power of Vision Language Models to revolutionize document processing:

- 📝 **Scanned Document Processing**: Intelligently identifies and extracts text, tables, and LaTeX equations from scanned documents into markdown-formatted content with high precision
- 🎨 **Advanced Content Formatting**: Preserves LaTeX equations, hyperlinks, images, and document hierarchy for markdown-formatted content
- 🤖 **Multi-LLM Support**: Seamlessly integrates with multiple Vision LLM providers such as OpenAI, Gemini, and Llama for optimal accuracy and speed
- 📁 **Local Model Hosting**: Supports local model hosting with Ollama for secure, no-cost, private, and offline document processing


## 🚀 Getting Started

### Prerequisites

- 🐍 Python >= 3.9
- 🖥️ Ollama (if you want to use local models)
- 🤖 API Key for OpenAI or Google Gemini (if you want to use OpenAI or Google Gemini)

### Installation

**Install the core package using pip (Recommended):**

```bash
pip install vision-parse
```

**Install the additional dependencies for OpenAI or Gemini:**

```bash
# To install all the additional dependencies
pip install 'vision-parse[all]'
```

**Install the package from source:**

```bash
pip install 'git+https://github.com/iamarunbrahma/vision-parse.git#egg=vision-parse[all]'
```

### Setting up Ollama (Optional)
See [Ollama Setup Guide](docs/ollama_setup.md) on how to setup Ollama locally.

> [!IMPORTANT]
> While Ollama provides free local model hosting, please note that vision models from Ollama can be significantly slower in processing documents and may not produce optimal results when handling complex PDF documents. For better accuracy and performance with complex layouts in PDF documents, consider using API-based models like OpenAI or Gemini.

### Setting up Vision Parse with Docker (Optional)
Check out [Docker Setup Guide](docs/docker_setup.md) on how to setup Vision Parse with Docker.

## 📚 Usage

### Basic Example Usage

```python
from vision_parse import VisionParser

# Initialize parser
parser = VisionParser(
    model_name="llama3.2-vision:11b", # For local models, you don't need to provide the api key
    temperature=0.4,
    top_p=0.5,
    image_mode="url", # Image mode can be "url", "base64" or None
    detailed_extraction=False, # Set to True for more detailed extraction
    enable_concurrency=False, # Set to True for parallel processing
)

# Convert PDF to markdown
pdf_path = "input_document.pdf" # local path to your pdf file
markdown_pages = parser.convert_pdf(pdf_path)

# Process results
for i, page_content in enumerate(markdown_pages):
    print(f"\n--- Page {i+1} ---\n{page_content}")
```

### Customize Ollama configuration for better performance

```python
from vision_parse import VisionParser

custom_prompt = """
Strictly preserve markdown formatting during text extraction from scanned document.
"""

# Initialize parser with Ollama configuration
parser = VisionParser(
    model_name="llama3.2-vision:11b",
    temperature=0.7,
    top_p=0.6,
    num_ctx=4096,
    image_mode="base64",
    custom_prompt=custom_prompt,
    detailed_extraction=True,
    ollama_config={
        "OLLAMA_NUM_PARALLEL": 8,
        "OLLAMA_REQUEST_TIMEOUT": 240,
    },
    enable_concurrency=True,
)

# Convert PDF to markdown
pdf_path = "input_document.pdf" # local path to your pdf file
markdown_pages = parser.convert_pdf(pdf_path)
```
> [!TIP]
> Please refer to [FAQs](docs/faq.md) for more details on how to improve the performance of locally hosted vision models.

### API Models Usage (OpenAI, Azure OpenAI, Gemini)

```python
from vision_parse import VisionParser


# Initialize parser with OpenAI model
parser = VisionParser(
    model_name="gpt-4o",
    api_key="your-openai-api-key", # Get the OpenAI API key from https://platform.openai.com/api-keys
    temperature=0.7,
    top_p=0.4,
    image_mode="url",
    detailed_extraction=False, # Set to True for more detailed extraction
    enable_concurrency=True,
)

# Initialize parser with Azure OpenAI model
parser = VisionParser(
    model_name="gpt-4o",
    image_mode="url",
    detailed_extraction=False, # Set to True for more detailed extraction
    enable_concurrency=True,
    openai_config={
        "AZURE_ENDPOINT_URL": "https://****.openai.azure.com/", # replace with your azure endpoint url
        "AZURE_DEPLOYMENT_NAME": "*******", # replace with azure deployment name, if needed
        "AZURE_OPENAI_API_KEY": "***********", # replace with your azure openai api key
        "AZURE_OPENAI_API_VERSION": "2024-08-01-preview", # replace with latest azure openai api version
    },
)


# Initialize parser with Google Gemini model
parser = VisionParser(
    model_name="gemini-2.5-flash",
    api_key="your-gemini-api-key", # Get the Gemini API key from https://aistudio.google.com/app/apikey
    temperature=0.7,
    top_p=0.4,
    image_mode="url",
    detailed_extraction=False, # Set to True for more detailed extraction
    enable_concurrency=True,
)

```

## ✅ Supported Models

This package supports the following Vision LLM models:

| **Model Name** | **Provider Name** |
|:------------:|:----------:|
| gpt-4o | OpenAI |
| gpt-4o-mini | OpenAI |
| gemini-2.5-pro | Google |
| gemini-2.5-flash | Google |
| gemini-2.0-flash | Google |
| llava:13b | Ollama |
| llava:34b | Ollama |
| llama3.2-vision:11b | Ollama |
| llama3.2-vision:70b | Ollama |
| deepseek-r1:32b | Ollama |

## 🔧 Customization Parameters

Vision Parse offers several customization parameters to enhance document processing:

| **Parameter** | **Description** | **Value Type** |
|:---------:|:-----------:|:-------------:|
| model_name | Name of the Vision LLM model to use | str |
| custom_prompt | Define custom prompt for the model and it will be used as a suffix to the default prompt | str |
| ollama_config | Specify custom configuration for Ollama client initialization | dict |
| openai_config | Specify custom configuration for OpenAI or Azure OpenAI client initialization | dict |
| gemini_config | Specify custom configuration for Gemini client initialization | dict |
| image_mode | Sets the image output format for the model i.e. if you want image url in markdown content or base64 encoded image | str |
| detailed_extraction | Enable advanced content extraction to extract complex information such as LaTeX equations, tables, images, etc. | bool |
| enable_concurrency | Enable parallel processing of multiple pages in a PDF document in a single request | bool |

> [!TIP]
> For more details on custom model configuration i.e. `openai_config`, `gemini_config`, and `ollama_config`; please refer to [Model Configuration](docs/model_config.md).

## 📊 Benchmarks

I conducted benchmarking to evaluate Vision Parse's performance against MarkItDown and Nougat. The benchmarking was conducted using a curated dataset of 100 diverse machine learning papers from arXiv, and the Marker library was used to generate the ground truth markdown formatted data.

Since there are no other ground truth data available for this task, I relied on the Marker library to generate the ground truth markdown formatted data.

### Results

| Parser | Accuracy Score |
|:-------:|:---------------:|
| Vision Parse | 92% |
| MarkItDown | 67% |
| Nougat | 79% |

> [!NOTE]
> I used gpt-4o model for Vision Parse to extract markdown content from the pdf documents. I have used model parameter settings as in `scoring.py` script. The above results may vary depending on the model you choose for Vision Parse and the model parameter settings.

### Run Your Own Benchmarks

You can benchmark the performance of Vision Parse on your machine using your own dataset. Run `scoring.py` to generate a detailed comparison report in the output directory.

1. Install packages from requirements.txt:
```bash
pip install --no-cache-dir -r benchmarks/requirements.txt
```

2. Run the benchmark script:
```bash
# Change `pdf_path` to your pdf file path and `benchmark_results_path` to your desired output path
python benchmarks/scoring.py
```

## 🤝 Contributing

Contributions to Vision Parse are welcome! Whether you're fixing bugs, adding new features, or creating example notebooks, your help is appreciated. Please check out [contributing guidelines](CONTRIBUTING.md) for instructions on setting up the development environment, code style requirements, and the pull request process.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
