import os
from fastapi import APIRouter, Depends, HTTPException
from api import queue
from services.dependencies import get_rag_system
from lib.rag import Rag

router = APIRouter()

@router.post("/ask")
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

@router.get("/health")
async def health_check(rag: Rag = Depends(get_rag_system)):
    """Проверка здоровья системы"""
    return {
        "status": "healthy",
        "vector_store": "connected" if rag.qdrant_client else "disconnected",
        "llm_initialized": rag.llm is not None
    }

