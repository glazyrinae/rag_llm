# app/rag_system.py
import os
from typing import List, Dict, Any
import logging
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from lib.file_processor import FileProcessorFacade

logger = logging.getLogger(__name__)

class Rag:
    def __init__(
        self,
        embeddings: str = "sentence-transformers/all-mpnet-base-v2",
        cache_dir: str = "/app/embedding_models",
    ):
        os.makedirs(cache_dir, exist_ok=True)

        # Инициализация Qdrant клиента
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )

        # Эмбеддинги
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings,
            cache_folder=cache_dir,
            model_kwargs={"device": "cpu"},
        )

        # Векторная БД
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name="python_code",
            embeddings=self.embeddings
        )

        # Сплиттер для кода
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\nclass ", "\n\ndef ", "\n\nasync def ", "\n\n# ", "\n\n", "\n", " "],
            length_function=len,
        )

        self.llm = None
        self.qa_chain = None
        self.memory = None

    def init_llm(self, api_key: str, model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"):
        """Инициализация ConversationRetrievalChain"""
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=1000,
        )
        
        # Память с ограничением по количеству сообщений
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=6,  # Храним последние 3 пары вопрос-ответ
            output_key="answer"
        )
        
        # ⭐ ОБНОВЛЕННЫЙ ПРОМПТ С ИСТОРИЕЙ ДИАЛОГА
        custom_prompt = PromptTemplate(
            template="""Ты - ассистент по программированию. Отвечай на вопросы используя контекст кода и историю диалога.

ИСТОРИЯ ДИАЛОГА:
{chat_history}

КОНТЕКСТ КОДА:
{context}

ВОПРОС: {question}

ПРОАЛИЗИРУЙ КОНТЕКСТ:
- Если есть код - объясни его работу
- Если есть текст - используй информацию из него
- Свяжи концепции из кода и текста
- Приведи примеры если возможно

ТРЕБОВАНИЯ К ОТВЕТУ:
- Отвечай четко и фактологически
- НЕ используй: "судя по всему", "наверное", "вероятно", "возможно", "кажется", "скорее всего"
- НЕ используй выражения неуверенности
- Если информации недостаточно - скажи это прямо
- Ответ должен быть утвердительным и конкретным

ПРАВИЛА:
- Используй информацию для формирования ответа, но НЕ цитируй конкретные фрагменты
- НЕ приводи примеры кода или текста дословно из информации
- НЕ говори "в контексте есть", "в информации приведен", "как показано в примере"
- Обобщи информацию и сформулируй ответ своими словами
- Ответ должен быть основан на информации, но не содержать прямых цитат

ОТВЕТ:""",
            input_variables=["context", "question", "chat_history"]
        )
        
        # Создание цепочки с диалогом
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt},
            verbose=False,
            max_tokens_limit=None,
            condense_question_llm=self.llm,
            get_chat_history=lambda chat_history: chat_history
        )
        
        logger.info("✅ ConversationRetrievalChain инициализирована с поддержкой истории")

    def ask_llm(self, question: str) -> Dict[str, Any]:
        """Основной метод для вопросов с автоматической историей"""
        if not self.qa_chain:
            raise ValueError("Сначала вызовите init_llm()")
            
        try:
            result = self.qa_chain({"question": question})
            
            # Форматируем source_documents
            source_docs = []
            for doc in result.get("source_documents", []):
                source_docs.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown")
                })
            
            return {
                "result": result["answer"],
                "source_documents": source_docs,
                "conversation_history": self.get_conversation_history()
            }
            
        except Exception as e:
            logger.error(f"Ошибка в ask_llm: {e}")
            return {
                "result": f"Ошибка: {str(e)}",
                "source_documents": [],
                "conversation_history": self.get_conversation_history()
            }

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Получить историю диалога"""
        if not self.memory:
            return []
            
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", [])
        
        formatted = []
        for msg in chat_history:
            if hasattr(msg, 'type'):
                role = "user" if msg.type == "human" else "assistant"
                formatted.append({
                    "role": role,
                    "content": msg.content
                })
        
        return formatted

    def clear_conversation_history(self):
        """Очистить историю диалога Пока добавил опционально"""
        if self.memory:
            self.memory.clear()
            logger.info("История диалога очищена")

    # Остальные методы без изменений
    def scan_dataset(self, project_path: str, file_extensions: List[str] = None) -> List[Document]:
        """Сканирование проекта с поддержкой разных типов файлов"""
        if file_extensions is None:
            file_extensions = ['.py', '.md', '.txt', '.pdf', '.html']
        
        documents = []

        for root, dirs, files in os.walk(project_path):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d not in ["__pycache__", "venv", "env"]
            ]

            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in file_extensions:
                    file_path = os.path.join(root, file)
                    try:
                        file_docs = FileProcessorFacade.parse_file(file_path, Document)
                        documents.extend(file_docs)
                        print(f"✅ Обработан: {file_path} ({len(file_docs)} документов)")
                    except Exception as e:
                        print(f"❌ Ошибка обработки {file_path}: {e}")

        return documents

    def add_documents_to_vectorstore(self, documents: List[Document]):
        """Добавление документов в векторную БД"""
        if not documents:
            raise ValueError("Нет документов для обработки")

        chunks = self.code_splitter.split_documents(documents)
        print(f"📄 Создано {len(chunks)} код-чанков")

        self.vector_store.add_documents(chunks)
        print("✅ Код добавлен в векторную БД")
