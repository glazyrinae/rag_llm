import os
from functools import lru_cache
from lib.rag import Rag

@lru_cache()
def get_rag_system() -> Rag:
    """Создает и кэширует единственный экземпляр RAG системы"""
    rag = Rag(
        embeddings="sentence-transformers/all-mpnet-base-v2",
        cache_dir="/app/embedding_models"
    )
    
    # Инициализируем LLM если нужно
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        rag.init_llm(api_key=api_key)
    
    return rag