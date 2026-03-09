"""Dosya yükleme ve metin çıkarma yardımcıları."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def read_text_from_file(file_path: str) -> str:
    """Verilen dosya uzantısına göre ham metni okur."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Desteklenmeyen dosya uzantısı: {extension}")

    if extension == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if extension == ".pdf":
        return _read_pdf(path)
    if extension == ".docx":
        return _read_docx(path)

    raise ValueError(f"Dosya işlenemedi: {file_path}")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_docx(path: Path) -> str:
    document = Document(str(path))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n".join(paragraphs)
