from typing import Dict

SUPPORTED_MODELS: Dict[str, str] = {
    "llama3.2-vision:11b": "ollama",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gemini-2.5-pro": "gemini",
    "gemini-2.5-flash": "gemini",
    "gemini-2.0-flash": "gemini",
}


def discover_ollama_vision_models() -> Dict[str, str]:
    try:
        import ollama
        from .model_detector import ModelDetector
        
        client = ollama.Client()
        detector = ModelDetector(client)
        vision_models = detector.get_vision_models()
        
        return {model: "ollama" for model in vision_models}
    except Exception:
        return {}
