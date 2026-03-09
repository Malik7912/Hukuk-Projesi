"""Metin temizleme ve bölüm ayrıştırma işlemleri."""

from __future__ import annotations

import re
from typing import Dict

SECTION_PATTERNS = {
    "mahkeme_başlığı": r"(\b\w+\s+mahkemesi\b.*?)\n",
    "davacı": r"davacı\s*:?\s*(.*?)(?:\n|$)",
    "davalı": r"davalı\s*:?\s*(.*?)(?:\n|$)",
    "dava_konusu": r"dava\s+konusu\s*:?\s*(.*?)(?:\n|$)",
    "açıklamalar": r"açıklamalar\s*:?\s*(.*?)(?:deliller|hukuki\s+sebepler|sonuç\s+ve\s+talep|$)",
    "deliller": r"deliller\s*:?\s*(.*?)(?:hukuki\s+sebepler|sonuç\s+ve\s+talep|$)",
    "hukuki_sebepler": r"hukuki\s+sebepler\s*:?\s*(.*?)(?:sonuç\s+ve\s+talep|$)",
    "sonuç_ve_talep": r"sonuç\s+ve\s+talep\s*:?\s*(.*)$",
}


def clean_text(text: str) -> str:
    """Temel metin temizliği: boşluk, satır sonu ve tekrar eden karakterleri düzenler."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def extract_sections(text: str) -> Dict[str, str]:
    """Dilekçedeki olası bölümleri regex tabanlı olarak çıkarır."""
    lowered = text.lower()
    sections: Dict[str, str] = {}

    for section_name, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL)
        sections[section_name] = match.group(1).strip() if match else ""

    return sections
