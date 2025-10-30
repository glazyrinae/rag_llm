# app/rag_system.py
import os
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from lib.file_processor import FileProcessorFacade


class Rag:
    def __init__(
        self,
        embeddings: str = "sentence-transformers/all-mpnet-base-v2",
        cache_dir: str = "/app/embedding_models",
    ):
        # Создаем директорию для кэша, если не существует
        os.makedirs(cache_dir, exist_ok=True)

        # Инициализация Qdrant клиента
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )

        # Эмбеддинги для кода с локальным кэшированием
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings,
            cache_folder=cache_dir,  # Локальная директория для модели если модель там лежит он не будет скачивать повторно
            model_kwargs={"device": "cpu"},  # или 'cuda' если есть GPU
        )

        # Проверяем, есть ли модель локально, если нет - скачиваем
        # self._ensure_embeddings_available(embeddings, cache_dir)

        # Инициализация векторной БД
        self.vector_store = Qdrant(
            client=self.qdrant_client,
            collection_name="python_code",
            embeddings=self.embeddings,
        )

        # Сплиттер для Python кода
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=[
                "\n\nclass ",
                "\n\ndef ",
                "\n\nasync def ",
                "\n\n# ",
                "\n\n",
                "\n",
                " ",
            ],
            length_function=len,
        )

        self.llm = None

    # Остальные методы остаются без изменений...
    def init_llm(self, api_key: str, model: str = "mistralai/mistral-7b-instruct"):
        """Инициализация облачной LLM"""
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=1000,
        )

    def scan_dataset(self, project_path: str, file_extensions: List[str] = None) -> List[Document]:
        """Сканирование проекта с поддержкой разных типов файлов"""
        if file_extensions is None:
            file_extensions = ['.py', '.md', '.txt', '.pdf', '.html']
        
        documents = []

        for root, dirs, files in os.walk(project_path):
            # Пропускаем служебные директории
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
                        # Фасад автоматически выберет нужный парсер
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

        # Разбиваем на чанки
        chunks = self.code_splitter.split_documents(documents)
        print(f"📄 Создано {len(chunks)} код-чанков")

        # Добавляем в векторную БД
        self.vector_store.add_documents(chunks)
        print("✅ Код добавлен в векторную БД")

    def create_code_qa_chain(self):
        """Создание QA цепи для работы с кодом"""
        if not self.llm:
            raise ValueError("LLM не инициализирована")

        # Промпт для разработчика
        prompt_template = """Ты - опытный Python разработчик. Используй предоставленный код для ответа на вопрос.

КОНТЕКСТ КОДА:
{context}

ВОПРОС: {question}

Ответь максимально подробно и технически грамотно. Если в коде есть примеры - используй их. Если нужно объяснить архитектуру - сделай это.

ОТВЕТ:"""

        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 6}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    def ask_about_code(self, question: str) -> Dict[str, Any]:
        """Задать вопрос о коде"""
        qa_chain = self.create_code_qa_chain()
        return qa_chain({"query": question})
