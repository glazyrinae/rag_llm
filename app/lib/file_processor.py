# app/utils/file_processor.py
import os
import zipfile
import tempfile
from typing import List


def extract_zip_to_temp(zip_file) -> str:
    """Распаковывает ZIP во временную директорию"""
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    return temp_dir


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
