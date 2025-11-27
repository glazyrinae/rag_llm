# app/queue_tasks.py
import logging, requests, os
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
        url_response = 1
        chat_id = kwargs.get('chat_id', '')
        if not question:
            raise ValueError("Question is required")
        
        logger.info(f"Processing RAG question: {question}")
        
        # Получаем экземпляр RAG и вызываем метод
        rag = get_rag_instance()
        response = rag.ask_llm(question)
        logger.info(f"RAG Result: {response}")
        # Форматируем результат
        formatted_result = {
            "answer": response["result"],
            "sources": [
                {
                    "file": doc.get("metadata", {}).get("file_path", "") if isinstance(doc, dict) else "",
                    "type": doc.get("metadata", {}).get("type", "") if isinstance(doc, dict) else "",
                    "name": doc.get("metadata", {}).get("name", "") if isinstance(doc, dict) else ""
                }
                for doc in response.get("source_documents", [])
            ],
            "conversation_history": response["conversation_history"],
            #"history_used": result["history_used"],
        }
        
        logger.info(f"RAG question processed successfully: {question}")
        if url_response:
            # res = requests.get(url_response, params={"result": formatted_result})
            # telegram_url = f"https://api.telegram.org/bot{settings.BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": chat_id,          # ← это и есть ID пользователя
                "text": response.get('result', ''),
                "parse_mode": "HTML"         # опционально
            }
            
            BOT_TOKEN = os.getenv("BOT_TOKEN")
            response = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", json=payload)
            
            res = response.json()
            if res:
                logger.info(f"Отправил ответ в телеграм: {res} c {payload} на {url_response}")
            #return res.get('result', '') if res else {}
        return formatted_result
        
    except Exception as e:
        logger.error(f"Error processing RAG question: {e}")
        raise