import requests
from lib.rag import Rag
from typing import Optional

rag_system = Rag()


async def ask_model(question: str, url_response: str = "") -> Optional[str]:

    model_answer = {"status": "LLM не инициализирована"}

    if not rag_system.llm:
        return model_answer

    result = rag_system.ask_about_code(question)
    # Форматируем источники
    sources = []
    for doc in result["source_documents"]:
        sources.append(
            {
                "file": doc.metadata.get("file_path", "unknown"),
                "type": doc.metadata.get("type", "unknown"),
                "name": doc.metadata.get("name", ""),
                "content_preview": doc.page_content[:200] + "...",
            }
        )

    model_answer = {
        "question": question,
        "answer": result["result"],
        "sources_count": len(sources),
        "sources": sources,
        "status": "Получен ответ от LLM",
    }
    if url_response:
        res = requests.get(url_response, params={"result": model_answer})
        return res.text if res else None
    return model_answer


async def check_model() -> bool:
    return True if rag_system.llm is not None else False


async def init_model(api_key: str) -> None:
    rag_system.init_llm(api_key)
