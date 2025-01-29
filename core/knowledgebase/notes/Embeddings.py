
from __future__ import annotations
from typing import List
import openai
from core.knowledgebase import constants
from core.knowledgebase.notes.JinaEmbeddings import JinaEmbeddings

openai.api_key = constants.OPENAI_API_KEY

class Embeddings:
    @staticmethod
    def get_embedding(text: str, model=constants.EMBEDDING_MODEL_NAME) -> List[float]:
        if constants.EMBEDDING_PROVIDER == "jina":
            return JinaEmbeddings.get_embedding(text)
        
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

if __name__ == '__main__':
    print(Embeddings.get_embedding('bonaparte'))
