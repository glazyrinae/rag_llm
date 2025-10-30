import os
from fastapi import APIRouter, Depends, HTTPException
from api import queue
from services.dependencies import get_rag_system
from lib.rag import Rag

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
        
        result = rag.ask_about_code(question)
        return {
            "answer": result["result"],
            "sources": [
                {
                    "file": doc.metadata.get("file_path", ""),
                    "type": doc.metadata.get("type", ""),
                    "name": doc.metadata.get("name", "")
                }
                for doc in result.get("source_documents", [])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@router.post("/ask")
async def ask_question_async(question: str):
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        task_id = queue.add(
            function_path="services.rag_adapter.rag_ask_about_code",
            kwargs={"question": question},
            queue="rag_questions",
            save_result_sec=3600,
        )
        return {"result": f"Задача добавлена в очередь {task_id}"}
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
        if not os.path.exists(project_path):
            raise HTTPException(status_code=404, detail="Директория проекта не найдена")

        documents = rag.scan_python_project(project_path)

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

