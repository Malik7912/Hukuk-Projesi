"""Karmaşıklık ve öncelik puanı hesaplama işlemleri."""

from __future__ import annotations

from .config import CASE_BASE_PRIORITY, URGENCY_KEYWORDS


def compute_complexity_score(
    taraf_sayisi: int,
    tanik_sayisi: int,
    delil_sayisi: int,
    metin_uzunlugu: int,
) -> float:
    """Dava karmaşıklığını 0-100 arası normalize ederek hesaplar."""
    raw_score = (
        taraf_sayisi * 8
        + tanik_sayisi * 10
        + delil_sayisi * 7
        + min(metin_uzunlugu / 120, 25)
    )
    return round(min(raw_score, 100), 2)


def compute_priority_score(case_type: str, complexity: float, text: str) -> float:
    """Dava türü taban puanı + karmaşıklık + aciliyet anahtar kelimeleri ile öncelik puanı üretir."""
    base_priority = CASE_BASE_PRIORITY.get(case_type, 50)
    urgency_bonus = sum(weight for keyword, weight in URGENCY_KEYWORDS.items() if keyword in text.lower())

    score = (base_priority * 0.5) + (complexity * 0.4) + urgency_bonus
    return round(min(score, 100), 2)
