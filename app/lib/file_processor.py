# app/utils/file_processor.py
import os
import ast
import re
import zipfile
import tempfile
from typing import List, Dict, Any, Callable
from pathlib import Path

import markdown
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup


class FileProcessor:
    """Класс для обработки файлов различных форматов"""
    
    # Константы для типов документов
    DOCUMENT_TYPES = {
        'full_file': 'full_file',
        'page': 'page',
        'section': 'section',
        'code_block': 'code_block',
        'paragraph': 'paragraph',
        'class': 'class',
        'function': 'function',
        'header': 'header',
        'list': 'list',
        'table': 'table',
        'link': 'link',
        'cleaned_html': 'cleaned_html'
    }
    
    # Маппинг расширений файлов на функции парсинга
    PARSER_MAPPING = {
        '.pdf': 'parse_pdf_file',
        '.md': 'parse_markdown_file',
        '.markdown': 'parse_markdown_file',
        '.txt': 'parse_text_file',
        '.py': 'parse_python_file',
        '.html': 'parse_html_file',
        '.htm': 'parse_html_file',
    }
    
    @staticmethod
    def _create_document(document_class, content: str, metadata: Dict[str, Any]) -> Any:
        """Создает документ с заданным контентом и метаданными"""
        return document_class(
            page_content=content,
            metadata=metadata
        )
    
    @staticmethod
    def _get_relative_path(file_path: str) -> str:
        """Возвращает относительный путь файла"""
        return os.path.relpath(file_path)
    
    @staticmethod
    def _read_file_content(file_path: str, encoding: str = 'utf-8') -> str:
        """Читает содержимое файла с указанной кодировкой"""
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    
    @staticmethod
    def _log_error(file_type: str, file_path: str, error: Exception):
        """Логирует ошибки обработки файлов"""
        print(f"❌ Ошибка чтения {file_type} {file_path}: {error}")


class PDFParser(FileProcessor):
    """Парсер PDF файлов"""
    
    @staticmethod
    def parse_pdf_file(file_path: str, document_class) -> List[Any]:
        """Парсинг одного PDF файла"""
        documents = []

        try:
            relative_path = FileProcessor._get_relative_path(file_path)
            reader = PdfReader(file_path)
            
            # Создаем полный текст документа
            full_text = PDFParser._extract_full_text(reader)
            
            # Базовый документ с полным содержимым
            full_doc = FileProcessor._create_document(
                document_class,
                content=f"Файл: {relative_path}\n\n{full_text}",
                metadata={
                    "file_path": relative_path,
                    "type": FileProcessor.DOCUMENT_TYPES['full_file'],
                    "language": "text",
                    "total_pages": len(reader.pages),
                }
            )
            documents.append(full_doc)
            
            # Документы по страницам
            documents.extend(PDFParser._create_page_documents(reader, relative_path, document_class))

        except Exception as e:
            FileProcessor._log_error("PDF", file_path, e)

        return documents
    
    @staticmethod
    def _extract_full_text(reader: PdfReader) -> str:
        """Извлекает текст со всех страниц PDF"""
        full_text = ""
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                full_text += f"--- Страница {page_num} ---\n{page_text}\n\n"
        return full_text
    
    @staticmethod
    def _create_page_documents(reader: PdfReader, relative_path: str, document_class) -> List[Any]:
        """Создает документы для каждой страницы PDF"""
        page_documents = []
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text.strip():
                page_doc = FileProcessor._create_document(
                    document_class,
                    content=f"Страница {page_num} из {relative_path}:\n\n{page_text}",
                    metadata={
                        "file_path": relative_path,
                        "type": FileProcessor.DOCUMENT_TYPES['page'],
                        "page_number": page_num,
                        "language": "text",
                    }
                )
                page_documents.append(page_doc)
        return page_documents


class MarkdownParser(FileProcessor):
    """Парсер Markdown файлов"""
    
    @staticmethod
    def parse_markdown_file(file_path: str, document_class) -> List[Any]:
        """Парсинг одного Markdown файла"""
        documents = []

        try:
            content = FileProcessor._read_file_content(file_path)
            relative_path = FileProcessor._get_relative_path(file_path)
            
            # Базовый документ
            full_doc = FileProcessor._create_document(
                document_class,
                content=f"Файл: {relative_path}\n\n{content}",
                metadata={
                    "file_path": relative_path,
                    "type": FileProcessor.DOCUMENT_TYPES['full_file'],
                    "language": "markdown",
                }
            )
            documents.append(full_doc)
            
            # Структурные элементы
            documents.extend(MarkdownParser._extract_sections(content, relative_path, document_class))
            documents.extend(MarkdownParser._extract_code_blocks(content, relative_path, document_class))

        except Exception as e:
            FileProcessor._log_error("Markdown", file_path, e)

        return documents
    
    @staticmethod
    def _extract_sections(content: str, relative_path: str, document_class) -> List[Any]:
        """Извлекает секции из Markdown документа"""
        documents = []
        lines = content.split('\n')
        current_section = []
        section_title = "Введение"
        section_level = 0

        for line in lines:
            header_match = re.match(r'^(#+)\s+(.+)$', line.strip())
            
            if header_match:
                # Сохраняем предыдущую секцию
                MarkdownParser._save_section(
                    current_section, section_title, section_level, 
                    relative_path, document_class, documents
                )
                
                # Начинаем новую секцию
                hashes, title = header_match.groups()
                section_level = len(hashes)
                section_title = title
                current_section = [line]
            else:
                current_section.append(line)

        # Сохраняем последнюю секцию
        MarkdownParser._save_section(
            current_section, section_title, section_level,
            relative_path, document_class, documents
        )
        
        return documents
    
    @staticmethod
    def _save_section(current_section: List[str], section_title: str, section_level: int,
                     relative_path: str, document_class, documents: List[Any]):
        """Сохраняет текущую секцию как документ"""
        if current_section:
            section_content = '\n'.join(current_section).strip()
            if section_content:
                section_doc = FileProcessor._create_document(
                    document_class,
                    content=f"Раздел '{section_title}' в {relative_path}:\n\n{section_content}",
                    metadata={
                        "file_path": relative_path,
                        "type": FileProcessor.DOCUMENT_TYPES['section'],
                        "title": section_title,
                        "level": section_level,
                        "language": "markdown",
                    }
                )
                documents.append(section_doc)
    
    @staticmethod
    def _extract_code_blocks(content: str, relative_path: str, document_class) -> List[Any]:
        """Извлекает блоки кода из Markdown"""
        documents = []
        code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)
        
        for lang, code in code_blocks:
            if code.strip():
                code_doc = FileProcessor._create_document(
                    document_class,
                    content=f"Блок кода ({lang or 'unknown'}) в {relative_path}:\n\n{code}",
                    metadata={
                        "file_path": relative_path,
                        "type": FileProcessor.DOCUMENT_TYPES['code_block'],
                        "language": lang or "unknown",
                    }
                )
                documents.append(code_doc)
        
        return documents


class TextParser(FileProcessor):
    """Парсер текстовых файлов"""
    
    @staticmethod
    def parse_text_file(file_path: str, document_class) -> List[Any]:
        """Парсинг одного текстового файла"""
        documents = []

        try:
            content = FileProcessor._read_file_content(file_path)
            relative_path = FileProcessor._get_relative_path(file_path)
            
            # Базовый документ
            full_doc = FileProcessor._create_document(
                document_class,
                content=f"Файл: {relative_path}\n\n{content}",
                metadata={
                    "file_path": relative_path,
                    "type": FileProcessor.DOCUMENT_TYPES['full_file'],
                    "language": "text",
                }
            )
            documents.append(full_doc)
            
            # Абзацы и структурные элементы
            documents.extend(TextParser._extract_paragraphs(content, relative_path, document_class))
            documents.extend(TextParser._extract_sections(content, relative_path, document_class))

        except Exception as e:
            FileProcessor._log_error("текстового файла", file_path, e)

        return documents
    
    @staticmethod
    def _extract_paragraphs(content: str, relative_path: str, document_class) -> List[Any]:
        """Извлекает абзацы из текста"""
        documents = []
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) >= 20:  # Пропускаем короткие абзацы
                paragraph_doc = FileProcessor._create_document(
                    document_class,
                    content=f"Абзац {i+1} из {relative_path}:\n\n{paragraph}",
                    metadata={
                        "file_path": relative_path,
                        "type": FileProcessor.DOCUMENT_TYPES['paragraph'],
                        "paragraph_number": i + 1,
                        "language": "text",
                    }
                )
                documents.append(paragraph_doc)
        
        return documents
    
    @staticmethod
    def _extract_sections(content: str, relative_path: str, document_class) -> List[Any]:
        """Извлекает структурные элементы из текста"""
        documents = []
        lines = content.split('\n')
        current_section = []
        section_title = "Начало"

        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if TextParser._is_heading(line) and current_section:
                TextParser._save_text_section(
                    current_section, section_title, relative_path, document_class, documents
                )
                section_title = line
                current_section = [line]
            else:
                current_section.append(line)

        TextParser._save_text_section(
            current_section, section_title, relative_path, document_class, documents
        )
        
        return documents
    
    @staticmethod
    def _is_heading(line: str) -> bool:
        """Определяет, является ли строка заголовком"""
        return (
            len(line) < 100 and 
            not line.endswith(('.', '!', '?')) and
            (line.isupper() or 
             re.match(r'^[IVXLC]+\.', line) or
             re.match(r'^\d+\.', line) or
             re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', line))
        )
    
    @staticmethod
    def _save_text_section(current_section: List[str], section_title: str,
                          relative_path: str, document_class, documents: List[Any]):
        """Сохраняет текстовую секцию как документ"""
        if current_section:
            section_content = '\n'.join(current_section).strip()
            if section_content:
                section_doc = FileProcessor._create_document(
                    document_class,
                    content=f"Раздел '{section_title}' в {relative_path}:\n\n{section_content}",
                    metadata={
                        "file_path": relative_path,
                        "type": FileProcessor.DOCUMENT_TYPES['section'],
                        "title": section_title,
                        "language": "text",
                    }
                )
                documents.append(section_doc)


class HTMLParser(FileProcessor):
    """Парсер HTML файлов"""
    
    @staticmethod
    def parse_html_file(file_path: str, document_class) -> List[Any]:
        """Парсинг одного HTML файла"""
        documents = []

        try:
            content = FileProcessor._read_file_content(file_path)
            relative_path = FileProcessor._get_relative_path(file_path)
            
            # Базовый документ
            full_doc = FileProcessor._create_document(
                document_class,
                content=f"Файл: {relative_path}\n\n{content}",
                metadata={
                    "file_path": relative_path,
                    "type": FileProcessor.DOCUMENT_TYPES['full_file'],
                    "language": "html",
                }
            )
            documents.append(full_doc)
            
            # Структурные элементы HTML
            soup = BeautifulSoup(content, 'html.parser')
            HTMLParser._clean_html(soup)
            
            documents.extend(HTMLParser._extract_cleaned_text(soup, relative_path, document_class))
            documents.extend(HTMLParser._extract_headers(soup, relative_path, document_class))
            documents.extend(HTMLParser._extract_paragraphs(soup, relative_path, document_class))
            documents.extend(HTMLParser._extract_lists(soup, relative_path, document_class))
            documents.extend(HTMLParser._extract_tables(soup, relative_path, document_class))
            documents.extend(HTMLParser._extract_links(soup, relative_path, document_class))

        except Exception as e:
            FileProcessor._log_error("HTML", file_path, e)

        return documents
    
    @staticmethod
    def _clean_html(soup: BeautifulSoup):
        """Очищает HTML от скриптов и стилей"""
        for script in soup(["script", "style"]):
            script.decompose()
    
    @staticmethod
    def _extract_cleaned_text(soup: BeautifulSoup, relative_path: str, document_class) -> List[Any]:
        """Извлекает очищенный текст HTML"""
        title = soup.find('title')
        page_title = title.get_text().strip() if title else "Без названия"
        text_content = soup.get_text(separator='\n', strip=True)
        
        cleaned_doc = FileProcessor._create_document(
            document_class,
            content=f"Очищенный текст HTML файла {relative_path}:\n\n{text_content}",
            metadata={
                "file_path": relative_path,
                "type": FileProcessor.DOCUMENT_TYPES['cleaned_html'],
                "title": page_title,
                "language": "text",
            }
        )
        return [cleaned_doc]
    
    @staticmethod
    def _extract_headers(soup: BeautifulSoup, relative_path: str, document_class) -> List[Any]:
        """Извлекает заголовки из HTML"""
        documents = []
        for i, header in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
            header_text = header.get_text().strip()
            if header_text:
                header_doc = FileProcessor._create_document(
                    document_class,
                    content=f"Заголовок {header.name} в {relative_path}:\n\n{header_text}",
                    metadata={
                        "file_path": relative_path,
                        "type": FileProcessor.DOCUMENT_TYPES['header'],
                        "level": header.name,
                        "header_number": i + 1,
                        "language": "text",
                    }
                )
                documents.append(header_doc)
        return documents
    
    @staticmethod
    def _extract_paragraphs(soup: BeautifulSoup, relative_path: str, document_class) -> List[Any]:
        """Извлекает абзацы из HTML"""
        documents = []
        for i, paragraph in enumerate(soup.find_all('p')):
            paragraph_text = paragraph.get_text().strip()
            if paragraph_text and len(paragraph_text) > 20:
                paragraph_doc = FileProcessor._create_document(
                    document_class,
                    content=f"Абзац {i+1} в {relative_path}:\n\n{paragraph_text}",
                    metadata={
                        "file_path": relative_path,
                        "type": FileProcessor.DOCUMENT_TYPES['paragraph'],
                        "paragraph_number": i + 1,
                        "language": "text",
                    }
                )
                documents.append(paragraph_doc)
        return documents
    
    @staticmethod
    def _extract_lists(soup: BeautifulSoup, relative_path: str, document_class) -> List[Any]:
        """Извлекает списки из HTML"""
        documents = []
        for i, list_elem in enumerate(soup.find_all(['ul', 'ol'])):
            list_items = [li.get_text().strip() for li in list_elem.find_all('li')]
            if list_items:
                list_text = "\n".join([f"• {item}" for item in list_items])
                list_doc = FileProcessor._create_document(
                    document_class,
                    content=f"Список {i+1} в {relative_path}:\n\n{list_text}",
                    metadata={
                        "file_path": relative_path,
                        "type": FileProcessor.DOCUMENT_TYPES['list'],
                        "list_number": i + 1,
                        "list_type": list_elem.name,
                        "items_count": len(list_items),
                        "language": "text",
                    }
                )
                documents.append(list_doc)
        return documents
    
    @staticmethod
    def _extract_tables(soup: BeautifulSoup, relative_path: str, document_class) -> List[Any]:
        """Извлекает таблицы из HTML"""
        documents = []
        for i, table in enumerate(soup.find_all('table')):
            table_data = []
            for row in table.find_all('tr'):
                row_data = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                if row_data:
                    table_data.append(" | ".join(row_data))
            
            if table_data:
                table_text = "\n".join(table_data)
                table_doc = FileProcessor._create_document(
                    document_class,
                    content=f"Таблица {i+1} в {relative_path}:\n\n{table_text}",
                    metadata={
                        "file_path": relative_path,
                        "type": FileProcessor.DOCUMENT_TYPES['table'],
                        "table_number": i + 1,
                        "rows_count": len(table_data),
                        "language": "text",
                    }
                )
                documents.append(table_doc)
        return documents
    
    @staticmethod
    def _extract_links(soup: BeautifulSoup, relative_path: str, document_class) -> List[Any]:
        """Извлекает ссылки из HTML"""
        documents = []
        for i, link in enumerate(soup.find_all('a', href=True)):
            link_text = link.get_text().strip()
            if link_text:
                link_doc = FileProcessor._create_document(
                    document_class,
                    content=f"Ссылка {i+1} в {relative_path}:\nТекст: {link_text}\nURL: {link['href']}",
                    metadata={
                        "file_path": relative_path,
                        "type": FileProcessor.DOCUMENT_TYPES['link'],
                        "link_number": i + 1,
                        "url": link['href'],
                        "language": "text",
                    }
                )
                documents.append(link_doc)
        return documents


class PythonParser(FileProcessor):
    """Парсер Python файлов"""
    
    @staticmethod
    def parse_python_file(file_path: str, document_class) -> List[Any]:
        """Парсинг одного Python файла"""
        documents = []

        try:
            content = FileProcessor._read_file_content(file_path)
            relative_path = FileProcessor._get_relative_path(file_path)
            
            # Базовый документ
            full_doc = FileProcessor._create_document(
                document_class,
                content=f"Файл: {relative_path}\n\n{content}",
                metadata={
                    "file_path": relative_path,
                    "type": FileProcessor.DOCUMENT_TYPES['full_file'],
                    "language": "python",
                }
            )
            documents.append(full_doc)
            
            # AST анализ
            documents.extend(PythonParser._extract_ast_elements(content, relative_path, document_class))

        except Exception as e:
            FileProcessor._log_error("Python", file_path, e)

        return documents
    
    @staticmethod
    def _extract_ast_elements(content: str, relative_path: str, document_class) -> List[Any]:
        """Извлекает элементы AST из Python кода"""
        documents = []
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    PythonParser._extract_class(node, content, relative_path, document_class, documents)
                elif isinstance(node, ast.FunctionDef):
                    PythonParser._extract_function(node, content, relative_path, document_class, documents)
                    
        except SyntaxError as e:
            print(f"⚠️ Ошибка AST в {relative_path}: {e}")
        
        return documents
    
    @staticmethod
    def _extract_class(node: ast.ClassDef, content: str, relative_path: str, 
                      document_class, documents: List[Any]):
        """Извлекает класс из AST"""
        class_code = ast.get_source_segment(content, node)
        class_doc = FileProcessor._create_document(
            document_class,
            content=f"Класс {node.name} в файле {relative_path}:\n\n{class_code}",
            metadata={
                "file_path": relative_path,
                "type": FileProcessor.DOCUMENT_TYPES['class'],
                "name": node.name,
                "language": "python",
            }
        )
        documents.append(class_doc)
    
    @staticmethod
    def _extract_function(node: ast.FunctionDef, content: str, relative_path: str,
                         document_class, documents: List[Any]):
        """Извлекает функцию из AST"""
        func_code = ast.get_source_segment(content, node)
        func_doc = FileProcessor._create_document(
            document_class,
            content=f"Функция {node.name} в файле {relative_path}:\n\n{func_code}",
            metadata={
                "file_path": relative_path,
                "type": FileProcessor.DOCUMENT_TYPES['function'],
                "name": node.name,
                "language": "python",
            }
        )
        documents.append(func_doc)


class ArchiveProcessor:
    """Обработчик архивных файлов"""
    
    @staticmethod
    def extract_zip_to_temp(zip_file) -> str:
        """Распаковывает ZIP во временную директорию"""
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        return temp_dir


class FileFinder:
    """Поиск файлов в директории"""
    
    @staticmethod
    def find_python_files(directory: str) -> List[str]:
        """Находит все Python файлы в директории"""
        python_files = []
        for root, dirs, files in os.walk(directory):
            # Игнорируем служебные директории
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files
    
    @staticmethod
    def find_files_by_extension(directory: str, extensions: List[str]) -> List[str]:
        """Находит файлы по расширениям"""
        found_files = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    found_files.append(os.path.join(root, file))
        return found_files


# Фасад для удобного использования
class FileProcessorFacade:
    """Фасад для обработки файлов"""
    
    PARSERS = {
        'pdf': PDFParser.parse_pdf_file,
        'markdown': MarkdownParser.parse_markdown_file,
        'text': TextParser.parse_text_file,
        'html': HTMLParser.parse_html_file,
        'python': PythonParser.parse_python_file,
    }
    
    @staticmethod
    def parse_file(file_path: str, document_class) -> List[Any]:
        """Парсит файл на основе его расширения"""
        file_ext = Path(file_path).suffix.lower()
        
        for ext, parser_name in FileProcessor.PARSER_MAPPING.items():
            if file_ext == ext:
                parser_func = FileProcessorFacade.PARSERS.get(parser_name.replace('parse_', '').replace('_file', ''))
                if parser_func:
                    return parser_func(file_path, document_class)
        
        # По умолчанию используем текстовый парсер
        return TextParser.parse_text_file(file_path, document_class)
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Возвращает список поддерживаемых расширений"""
        return list(FileProcessor.PARSER_MAPPING.keys())