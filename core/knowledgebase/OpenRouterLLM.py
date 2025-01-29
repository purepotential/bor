python
from typing import Any, List, Optional

class OpenRouterLLM:
    model_name: str = "deepseek/deepseek-chat"
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        headers = {
            "Authorization": f"Bearer {constants.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        # ... (rest of the _call method) ...