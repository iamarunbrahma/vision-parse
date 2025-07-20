from typing import Dict

SUPPORTED_MODELS: Dict[str, str] = {
    "llama3.2-vision:11b": "ollama",
    "llama3.2-vision:70b": "ollama",
    "llava:13b": "ollama",
    "llava:34b": "ollama",
    "deepseek-r1:32b": "ollama",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gemini-1.5-flash": "gemini",
    "gemini-2.0-flash-exp": "gemini",
    "gemini-1.5-pro": "gemini",
    "deepseek-chat": "deepseek",
    "meta-llama/llama-4-scout-17b-16e-instruct": "groq",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "groq",
}
