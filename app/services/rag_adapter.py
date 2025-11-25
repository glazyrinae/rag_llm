# app/queue_tasks.py
import logging, requests
from lib.rag import Rag
from services.dependencies import get_rag_system

logger = logging.getLogger(__name__)

# Глобальная переменная для хранения экземпляра RAG
_rag_instance = None

def get_rag_instance() -> Rag:
    """Получить экземпляр RAG системы"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = get_rag_system()
    return _rag_instance

async def rag_ask_llm(**kwargs) -> dict:
    """
    Обертка для вызова метода ask_llm класса Rag
    """
    try:
        question = kwargs.get('question', '')
        url_response = kwargs.get('url_response', '')
        chat_id = kwargs.get('chat_id', '')
        if not question:
            raise ValueError("Question is required")
        
        logger.info(f"Processing RAG question: {question}")
        
        # Получаем экземпляр RAG и вызываем метод
        rag = get_rag_instance()
        result = rag.ask_llm(question)
        
        # Форматируем результат
        formatted_result = {
            "answer": result["result"],
            "sources": [
                {
                    "file": doc.metadata.get("file_path", ""),
                    "type": doc.metadata.get("type", ""),
                    "name": doc.metadata.get("name", "")
                }
                for doc in result.get("source_documents", [])
            ],
            "conversation_history": result["conversation_history"],
            #"history_used": result["history_used"],
        }
        
        logger.info(f"RAG question processed successfully: {question}")
        if url_response:
            # res = requests.get(url_response, params={"result": formatted_result})
            # telegram_url = f"https://api.telegram.org/bot{settings.BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": chat_id,          # ← это и есть ID пользователя
                "text": formatted_result,
                "parse_mode": "HTML"         # опционально
            }
            response = requests.post(url_response, json=payload)
            res = response.json()
            if res:
                logger.info(f"RAG response: {res.text}")
            return res.text if res else {}
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error processing RAG question: {e}")
        raise