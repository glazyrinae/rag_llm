import os, logging
from fastapi import APIRouter, Depends, HTTPException
from api import queue
from services.dependencies import get_rag_system
from lib.rag import Rag

logger = logging.getLogger(__name__)

router = APIRouter()

# @app.post("/predict")
# def predict_doc(item: Item):
#     try:
#         id = queue.add(
#             function=get_predict,
#             kwargs={"file_link": item.file_link, "url_response": item.url_for_response},
#         )
#         return {"result": f"Задача добавлена в очередь {id}"}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))

@router.post("/ask_test")
async def ask_question(
    question: str,
    rag: Rag = Depends(get_rag_system)
):
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = rag.ask_llm(question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@router.post("/ask")
async def ask_question_async(question: str, chat_id: str):
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        task_id = queue.add(
            function_path="services.rag_adapter.rag_ask_llm",
            kwargs={"question": question, "chat_id": chat_id},
            queue="rag_questions",
            save_result_sec=3600,
        )
        logger.info(f"Added RAG question task {task_id} to queue")
        return {"result": "Запрос отправлен в модель"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@router.get("/health")
async def health_check(rag: Rag = Depends(get_rag_system)):
    """Проверка здоровья системы"""
    return {
        "status": "healthy",
        "vector_store": "connected" if rag.qdrant_client else "disconnected",
        "llm_initialized": rag.llm is not None
    }

@router.post("/index-local-project")
async def index_local_project(rag: Rag = Depends(get_rag_system), project_path: str = "/app/data"):
    """Индексация локального проекта (уже в контейнере)"""
    try:
        rag.clear_conversation_history()
        if not os.path.exists(project_path):
            raise HTTPException(status_code=404, detail="Директория проекта не найдена")

        documents = rag.scan_dataset(project_path)

        if not documents:
            raise HTTPException(status_code=400, detail="Не найдено Python файлов")

        rag.add_documents_to_vectorstore(documents)

        return {
            "status": "success",
            "documents_created": len(documents),
            "project_path": project_path,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

