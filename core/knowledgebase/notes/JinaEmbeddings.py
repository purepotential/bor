
from __future__ import annotations
from typing import List
import requests
from core.knowledgebase import constants

class JinaEmbeddings:
    @staticmethod
    def get_embedding(text: str) -> List[float]:
        text = text.replace("\n", " ")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {constants.JINA_API_KEY}'
        }
        data = {
            "model": constants.JINA_MODEL_NAME,
            "dimensions": constants.JINA_DIMENSIONS,
            "normalized": True,
            "embedding_type": "float",
            "input": [{"text": text}]
        }
        
        response = requests.post(constants.JINA_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            raise Exception(f"Error getting embedding: {response.text}")
