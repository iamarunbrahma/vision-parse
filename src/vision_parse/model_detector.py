from typing import Dict, Set, Any
import logging

_logger = logging.getLogger(__name__)


class ModelDetector:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._vision_cache: Dict[str, bool] = {}

    def is_vision_capable(self, model_name: str) -> bool:
        if model_name in self._vision_cache:
            return self._vision_cache[model_name]

        try:
            result = self._detect_vision_capability(model_name)
            self._vision_cache[model_name] = result
            return result
        except Exception as e:
            _logger.warning(f"Failed to detect capabilities for {model_name}: {e}")
            return False

    def get_vision_models(self) -> Set[str]:
        vision_models = set()

        try:
            models = self._client.list()
            for model in models.get("models", []):
                model_name = model.get("name")
                if model_name and self.is_vision_capable(model_name):
                    vision_models.add(model_name)
        except Exception as e:
            _logger.warning(f"Failed to list models: {e}")

        return vision_models

    def _detect_vision_capability(self, model_name: str) -> bool:
        try:
            response = self._client.show(model_name)

            capabilities = response.get("capabilities", [])
            if isinstance(capabilities, list) and "vision" in capabilities:
                return True

            families = response.get("details", {}).get("families", [])
            return "clip" in families

        except Exception as e:
            _logger.debug(f"Error detecting capabilities for {model_name}: {e}")
            return False
