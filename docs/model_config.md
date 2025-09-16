## Custom Configuration for Vision LLM providers:

### Ollama Client:

Pass the following configuration settings to `ollama_config` parameter while initializing the VisionParser class.

| **Custom Configuration** | **Description** |
|:---------:|:-----------:|
| OLLAMA_NUM_PARALLEL | Number of parallel requests to Ollama server |
| OLLAMA_REQUEST_TIMEOUT | Timeout for requests to the Ollama server (by default, it's set to 240.0) |
| OLLAMA_NUM_GPU | Number of GPUs to use if available |
| OLLAMA_NUM_THREAD | Number of CPU threads to use |
| OLLAMA_KEEP_ALIVE | Keep-alive timeout for the Ollama server (by default, it's set to -1) |
| OLLAMA_HOST | Host URL for the Ollama server (by default, it's set to http://localhost:11434 ) |
| OLLAMA_GPU_LAYERS | Number of layers to use if GPU is available |

For model-specific parameters (like temperature, top_p, etc.), please refer to the [Ollama Model Parameters documentation](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values). You can pass model-specific parameters as additional kwargs to the `VisionParser` class.

### OpenAI Client:

Pass the following configuration settings to `openai_config` parameter while initializing the VisionParser class.

| **Custom Configuration** | **Description** |
|:---------:|:-----------:|
| OPENAI_BASE_URL | Base URL for the OpenAI server (by default, it's set to None) |
| OPENAI_MAX_RETRIES | Maximum number of retries for OpenAI requests (by default, it's set to 3) |
| OPENAI_TIMEOUT | Timeout for OpenAI requests (by default, it's set to 240.0) |
| OPENAI_DEFAULT_HEADERS | Default headers for OpenAI requests (by default, it's set to None) |

For model-specific parameters (like temperature, max_tokens, etc.), please refer to the [OpenAI Chat Completion API documentation](https://platform.openai.com/docs/api-reference/chat/create). You can pass model-specific parameters as additional kwargs to the `VisionParser` class.

### Azure OpenAI Client:

Pass the following configuration settings to `openai_config` parameter while initializing the VisionParser class.

| **Custom Configuration** | **Description** |
|:---------:|:-----------:|
| AZURE_OPENAI_API_KEY | Azure OpenAI API key |
| AZURE_ENDPOINT_URL | Azure OpenAI Endpoint URL |
| AZURE_DEPLOYMENT_NAME | Azure OpenAI Deployment Name |
| AZURE_OPENAI_API_VERSION | Azure OpenAI API Version (by default, it's set to "2024-08-01-preview") |
