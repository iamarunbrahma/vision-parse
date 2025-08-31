from typing import Dict

SUPPORTED_MODELS: Dict[str, str] = {
    "llama3.2-vision:11b": "ollama",
    "llama3.2-vision:70b": "ollama",
    "llava:13b": "ollama",
    "llava:34b": "ollama",
    "deepseek-r1:32b": "ollama",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gemini-2.5-pro": "gemini",
    "gemini-2.5-flash": "gemini",
    "gemini-2.0-flash": "gemini",
}
