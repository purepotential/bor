
from typing import Any, List, Optional, Dict
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
from core.knowledgebase import constants

class OpenRouterLLM(LLM):
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
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error from OpenRouter API: {response.text}")
            
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
        }
