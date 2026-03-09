"""Metinden dava özellikleri çıkarımı."""

from __future__ import annotations

import re
from typing import Dict

from .config import COURT_PATTERNS, EVIDENCE_KEYWORDS


def detect_court_type(text: str) -> str:
    lowered = text.lower()
    for court_name, patterns in COURT_PATTERNS.items():
        for pattern in patterns:
            if pattern in lowered:
                return court_name
    return "belirlenemedi"


def extract_features(text: str) -> Dict[str, int | str]:
    lowered = text.lower()

    taraf_sayisi = _count_parties(lowered)
    tanik_sayisi = _count_witnesses(lowered)
    delil_sayisi = _count_evidences(lowered)
    metin_uzunlugu = len(lowered.split())

    return {
        "mahkeme_turu": detect_court_type(lowered),
        "taraf_sayisi": taraf_sayisi,
        "tanik_sayisi": tanik_sayisi,
        "delil_sayisi": delil_sayisi,
        "metin_uzunlugu": metin_uzunlugu,
    }


def _count_parties(text: str) -> int:
    davaci_count = len(re.findall(r"\bdavacı\b", text))
    davali_count = len(re.findall(r"\bdavalı\b", text))
    unique = davaci_count + davali_count
    return unique if unique > 0 else 2


def _count_witnesses(text: str) -> int:
    explicit_numbers = re.findall(r"(\d+)\s*(?:adet\s*)?tanık", text)
    if explicit_numbers:
        return sum(int(number) for number in explicit_numbers)

    return len(re.findall(r"\btanık\b", text))


def _count_evidences(text: str) -> int:
    evidence_matches = 0
    for keyword in EVIDENCE_KEYWORDS:
        evidence_matches += len(re.findall(rf"\b{re.escape(keyword)}\b", text))

    explicit_delil = re.findall(r"(\d+)\s*(?:adet\s*)?delil", text)
    if explicit_delil:
        evidence_matches += sum(int(number) for number in explicit_delil)

    return evidence_matches
