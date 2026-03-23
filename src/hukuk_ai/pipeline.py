"""Uçtan uca analiz akışını yöneten ana pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .classifier import BerturkCaseTypeClassifier
from .complexity_model import BerturkComplexityRegressor
from .information_extraction import extract_features
from .io_utils import read_text_from_file
from .preprocessing import clean_text, extract_sections
from .scoring import compute_priority_score


@dataclass
class CaseAnalysisResult:
    dosya_adi: str
    dava_turu: str
    guven_skoru: float
    tahmin_modu: str
    mahkeme_turu: str
    taraf_sayisi: int
    tanik_sayisi: int
    delil_sayisi: int
    metin_uzunlugu: int
    karmasiklik_puani: float
    karmasiklik_modu: str
    oncelik_puani: float
    bolumler: Dict[str, str]


class CaseAnalysisPipeline:
    """Dava dosyalarını analiz eder ve öncelik sırasına göre sonuç döndürür."""

    def __init__(
        self,
        classifier: BerturkCaseTypeClassifier | None = None,
        complexity_model: BerturkComplexityRegressor | None = None,
    ) -> None:
        self.classifier = classifier or BerturkCaseTypeClassifier()
        self.complexity_model = complexity_model or BerturkComplexityRegressor()

    def analyze_files(self, file_paths: List[str]) -> List[CaseAnalysisResult]:
        results = [self._analyze_single_file(file_path) for file_path in file_paths]
        return sorted(results, key=lambda row: row.oncelik_puani, reverse=True)

    def _analyze_single_file(self, file_path: str) -> CaseAnalysisResult:
        raw_text = read_text_from_file(file_path)
        cleaned_text = clean_text(raw_text)
        sections = extract_sections(cleaned_text)

        classification_input = self._build_classification_text(cleaned_text, sections)
        classification = self.classifier.predict(classification_input)

        features = extract_features(cleaned_text)
        complexity_prediction = self.complexity_model.predict(cleaned_text, features=features)
        complexity = float(complexity_prediction["complexity"])
        priority = compute_priority_score(
            case_type=str(classification["predicted_case_type"]),
            complexity=complexity,
            text=classification_input,
        )

        return CaseAnalysisResult(
            dosya_adi=Path(file_path).name,
            dava_turu=str(classification["predicted_case_type"]),
            guven_skoru=float(classification["confidence"]),
            tahmin_modu=str(classification.get("mode", "unknown")),
            mahkeme_turu=str(features["mahkeme_turu"]),
            taraf_sayisi=int(features["taraf_sayisi"]),
            tanik_sayisi=int(features["tanik_sayisi"]),
            delil_sayisi=int(features["delil_sayisi"]),
            metin_uzunlugu=int(features["metin_uzunlugu"]),
            karmasiklik_puani=complexity,
            karmasiklik_modu=str(complexity_prediction.get("mode", "unknown")),
            oncelik_puani=priority,
            bolumler=sections,
        )

    @staticmethod
    def _build_classification_text(cleaned_text: str, sections: Dict[str, str]) -> str:
        important_chunks = [
            sections.get("dava_konusu", ""),
            sections.get("açıklamalar", ""),
            sections.get("sonuç_ve_talep", ""),
        ]
        merged = "\n".join(chunk for chunk in important_chunks if chunk)
        return merged if merged else cleaned_text
