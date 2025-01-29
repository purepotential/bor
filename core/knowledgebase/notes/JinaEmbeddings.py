
from __future__ import annotations
from typing import List, Union
import requests
from core.knowledgebase import constants

class JinaEmbeddings:
    @staticmethod
    def get_embedding(text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(text, str):
            text = text.replace("\n", " ")
            input_data = [{"text": text}]
        else:
            input_data = [{"text": t.replace("\n", " ")} for t in text]
            
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {constants.JINA_API_KEY}'
        }
        data = {
            "model": constants.JINA_MODEL_NAME,
            "dimensions": constants.JINA_DIMENSIONS,
            "normalized": True,
            "embedding_type": "float",
            "input": input_data
        }
        
        response = requests.post(constants.JINA_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            response_data = response.json()
            embeddings = [item['embedding'] for item in response_data['data']]
            return embeddings[0] if isinstance(text, str) else embeddings
        else:
            raise Exception(f"Error getting embedding: {response.text}")
