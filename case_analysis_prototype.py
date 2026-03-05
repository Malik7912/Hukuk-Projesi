"""
Türkiye'deki dava dilekçelerini analiz etmek için araştırma amaçlı prototip.

Bu modül; metin temizleme, bilgi çıkarma, dava türü tahmini,
karmaşıklık ve öncelik puanı hesaplama adımlarını modüler şekilde içerir.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re
import unicodedata
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    """Türkçe karakterleri koruyarak karşılaştırma için normalize eder."""
    lowered = text.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


@dataclass
class CaseAnalysisResult:
    """Tek bir dava dosyasının analiz sonucunu tutar."""

    file_name: str
    case_type: str
    court_type: str
    party_count: int
    witness_count: int
    evidence_count: int
    text_length: int
    complexity_score: float
    priority_score: float
    detected_sections: Dict[str, str]


class FileReader:
    """Dosyadan metin okuma işlemlerini yönetir."""

    @staticmethod
    def read_text(path: str) -> str:
        """UTF-8 metin dosyasını okur.

        Args:
            path: Okunacak dosya yolu.

        Returns:
            Dosya içeriği.

        Raises:
            ValueError: Dosya boş ise.
            OSError: Dosya erişiminde hata olursa.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()

        if not data.strip():
            raise ValueError(f"Dosya boş görünüyor: {path}")

        return data


class TextPreprocessor:
    """Metin temizleme ve normalize etme adımları."""

    @staticmethod
    def clean_text(text: str) -> str:
        # Fazla boşlukları sadeleştirir, baş/son boşlukları kırpar.
        text = text.replace("\r", "\n")
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()


class PetitionSectionParser:
    """Dava dilekçesindeki olası bölümleri ayırt etmeye çalışır."""

    SECTION_PATTERNS = {
        "mahkeme_başlığı": r"(\b[\wçğıöşüİĞÜŞÖÇ\s]+MAHKEMES[İI]\b.*?)\n",
        "davacı": r"\bDAVACI\b\s*:?\s*(.+)",
        "davalı": r"\bDAVALI\b\s*:?\s*(.+)",
        "dava_konusu": r"\bKONU\b\s*:?\s*(.+)",
        "açıklamalar": r"\bAÇIKLAMALAR\b\s*:?\s*(.+?)(?=\bDEL[İI]LLER\b|\bHUKUK[İI]\s+SEBEPLER\b|\bSONUÇ\s+VE\s+TALEP\b|$)",
        "deliller": r"\bDEL[İI]LLER\b\s*:?\s*(.+?)(?=\bHUKUK[İI]\s+SEBEPLER\b|\bSONUÇ\s+VE\s+TALEP\b|$)",
        "hukuki_sebepler": r"\bHUKUK[İI]\s+SEBEPLER\b\s*:?\s*(.+?)(?=\bSONUÇ\s+VE\s+TALEP\b|$)",
        "sonuç_ve_talep": r"\bSONUÇ\s+VE\s+TALEP\b\s*:?\s*(.+)$",
    }

    @classmethod
    def detect_sections(cls, text: str) -> Dict[str, str]:
        sections: Dict[str, str] = {}
        for name, pattern in cls.SECTION_PATTERNS.items():
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                sections[name] = match.group(1).strip()
        return sections


class InformationExtractor:
    """Metinden temel sayısal/anahtar özellikleri çıkartır."""

    EVIDENCE_KEYWORDS = [
        "delil",
        "tanık beyanı",
        "banka kaydı",
        "sözleşme",
        "fatura",
        "whatsapp",
        "kamera kaydı",
        "bilirkişi raporu",
        "tapu kaydı",
    ]

    PARTY_LABELS = ["davacı", "davalı", "müşteki", "şikayetçi", "vekil"]

    @classmethod
    def count_parties(cls, text: str) -> int:
        # Etiket bazlı sayım + minimum 2 (davacı/davalı varsayımı) yaklaşımı.
        hits = 0
        for label in cls.PARTY_LABELS:
            hits += len(re.findall(rf"\b{label}\b", text, flags=re.IGNORECASE))
        return max(2, hits)

    @staticmethod
    def count_witnesses(text: str) -> int:
        patterns = [r"\btanık\b", r"\bşahit\b", r"\btanıklarımız\b"]
        return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)

    @classmethod
    def count_evidence(cls, text: str, sections: Dict[str, str]) -> int:
        evidence_text = sections.get("deliller", text)
        count = 0
        for key in cls.EVIDENCE_KEYWORDS:
            count += len(re.findall(rf"\b{re.escape(key)}\b", evidence_text, flags=re.IGNORECASE))

        # DELİLLER kısmındaki satır maddelerini de delil adedi olarak değerlendir.
        line_items = len(re.findall(r"(^|\n)\s*(?:[-*]|\d+[.)])\s+", evidence_text))
        return max(count, line_items)

    @staticmethod
    def text_length(text: str) -> int:
        return len(text.split())


class CaseTypeClassifier:
    """Basit kural tabanlı dava türü sınıflandırıcısı."""

    CASE_KEYWORDS: Dict[str, List[str]] = {
        "kira_davası": ["kira", "tahliye", "kiraya veren", "kiracı"],
        "alacak_davası": ["alacak", "borç", "ödeme", "icra"],
        "boşanma_davası": ["boşanma", "evlilik birliği", "nafaka", "anlaşmalı boşanma"],
        "velayet_davası": ["velayet", "çocuğun üstün yararı", "kişisel ilişki"],
        "iş_davası": ["iş sözleşmesi", "işveren", "işçi", "mesai"],
        "kıdem_tazminatı_davası": ["kıdem tazminatı", "ihbar tazminatı", "haksız fesih"],
        "işe_iade_davası": ["işe iade", "geçersiz fesih", "işe başlatmama"],
        "tapu_davası": ["tapu", "tescil", "kadastro", "mülkiyet"],
    }

    COURT_KEYWORDS: Dict[str, List[str]] = {
        "iş_mahkemesi": ["iş mahkemesi"],
        "aile_mahkemesi": ["aile mahkemesi"],
        "asliye_hukuk_mahkemesi": ["asliye hukuk mahkemesi"],
        "sulh_hukuk_mahkemesi": ["sulh hukuk mahkemesi"],
        "asliye_ceza_mahkemesi": ["asliye ceza mahkemesi"],
    }

    @classmethod
    def classify_case_type(cls, text: str) -> str:
        lowered = normalize_text(text)
        best_type = "belirsiz"
        best_score = 0
        for case_type, keys in cls.CASE_KEYWORDS.items():
            score = sum(lowered.count(normalize_text(k)) for k in keys)
            if score > best_score:
                best_type = case_type
                best_score = score
        return best_type

    @classmethod
    def detect_court_type(cls, text: str) -> str:
        lowered = normalize_text(text)
        for court, keys in cls.COURT_KEYWORDS.items():
            if any(normalize_text(k) in lowered for k in keys):
                return court
        return "belirsiz"


class ScoringEngine:
    """Karmaşıklık ve öncelik puanlarını hesaplar."""

    PRIORITY_BY_CASE_TYPE = {
        "işe_iade_davası": 85,
        "velayet_davası": 82,
        "kıdem_tazminatı_davası": 78,
        "iş_davası": 75,
        "boşanma_davası": 72,
        "kira_davası": 68,
        "alacak_davası": 65,
        "tapu_davası": 62,
        "belirsiz": 55,
    }

    URGENCY_KEYWORDS = {
        "acil": 8,
        "tedbir": 7,
        "çocuk": 6,
        "mağdur": 5,
        "şiddet": 9,
        "hak kaybı": 7,
    }

    @staticmethod
    def complexity_score(party_count: int, witness_count: int, evidence_count: int, text_length: int) -> float:
        # 0-100 aralığına normalize bir skor.
        raw = (
            party_count * 1.8
            + witness_count * 2.3
            + evidence_count * 2.7
            + math.log(max(text_length, 1), 2) * 4.2
        )
        return round(min(raw, 100), 2)

    @classmethod
    def priority_score(cls, case_type: str, complexity_score: float, text: str) -> float:
        base = cls.PRIORITY_BY_CASE_TYPE.get(case_type, cls.PRIORITY_BY_CASE_TYPE["belirsiz"])
        lowered = normalize_text(text)
        urgency_boost = sum(weight for key, weight in cls.URGENCY_KEYWORDS.items() if normalize_text(key) in lowered)
        score = 0.45 * base + 0.45 * complexity_score + 0.10 * urgency_boost
        return round(min(score, 100), 2)


class PetitionAnalyzer:
    """Uçtan uca dava dilekçesi analiz servisi."""

    def analyze_text(self, text: str, file_name: str = "<memory>") -> CaseAnalysisResult:
        cleaned = TextPreprocessor.clean_text(text)
        sections = PetitionSectionParser.detect_sections(cleaned)

        case_type = CaseTypeClassifier.classify_case_type(cleaned)
        court_type = CaseTypeClassifier.detect_court_type(cleaned)

        party_count = InformationExtractor.count_parties(cleaned)
        witness_count = InformationExtractor.count_witnesses(cleaned)
        evidence_count = InformationExtractor.count_evidence(cleaned, sections)
        text_length = InformationExtractor.text_length(cleaned)

        complexity = ScoringEngine.complexity_score(
            party_count=party_count,
            witness_count=witness_count,
            evidence_count=evidence_count,
            text_length=text_length,
        )
        priority = ScoringEngine.priority_score(
            case_type=case_type,
            complexity_score=complexity,
            text=cleaned,
        )

        return CaseAnalysisResult(
            file_name=file_name,
            case_type=case_type,
            court_type=court_type,
            party_count=party_count,
            witness_count=witness_count,
            evidence_count=evidence_count,
            text_length=text_length,
            complexity_score=complexity,
            priority_score=priority,
            detected_sections=sections,
        )

    def analyze_file(self, path: str) -> CaseAnalysisResult:
        text = FileReader.read_text(path)
        return self.analyze_text(text, file_name=path)

    def rank_cases(self, texts: List[Tuple[str, str]]) -> List[CaseAnalysisResult]:
        """Birden fazla davayı analiz edip öncelik puanına göre sıralar.

        Args:
            texts: (dosya_adı, metin) çiftleri.
        """
        results = [self.analyze_text(text, file_name=name) for name, text in texts]
        return sorted(results, key=lambda r: r.priority_score, reverse=True)


def demo() -> None:
    """Örnek kullanım: tek dosya analizi + çoklu dava sıralama."""
    sample_1 = """
    İSTANBUL İŞ MAHKEMESİ SAYIN HAKİMLİĞİ'NE
    DAVACI: Ahmet Yılmaz
    DAVALI: ABC Tekstil A.Ş.
    KONU: İşe iade talebimizden ibarettir.
    AÇIKLAMALAR:
    Davacı işçi, geçersiz fesih nedeniyle mağdur olmuştur. Acil değerlendirme talep edilir.
    Davacı lehine tanıklarımız mevcuttur. Tanık: Mehmet Kaya, Ayşe Demir.
    DELİLLER:
    - İş sözleşmesi
    - SGK kayıtları
    - Tanık beyanı
    HUKUKİ SEBEPLER:
    İş Kanunu ve ilgili mevzuat.
    SONUÇ VE TALEP:
    Davanın kabulü ile davacının işe iadesine karar verilmesini talep ederiz.
    """

    sample_2 = """
    ANKARA SULH HUKUK MAHKEMESİ'NE
    DAVACI: Zeynep Kara
    DAVALI: Murat Kara
    KONU: Kira alacağı ve tahliye talebi.
    AÇIKLAMALAR:
    Davalı uzun süredir kira bedelini ödememiştir.
    DELİLLER:
    1) Kira sözleşmesi
    2) Banka dekontları
    SONUÇ VE TALEP:
    Tahliye ve alacağın tahsilini talep ederiz.
    """

    analyzer = PetitionAnalyzer()

    single_result = analyzer.analyze_text(sample_1, file_name="örnek_dava_1")
    print("--- Tek Dava Analizi ---")
    for key, value in asdict(single_result).items():
        if key != "detected_sections":
            print(f"{key}: {value}")

    ranked = analyzer.rank_cases([
        ("örnek_dava_1", sample_1),
        ("örnek_dava_2", sample_2),
    ])

    print("\n--- Öncelik Sıralaması ---")
    for idx, item in enumerate(ranked, start=1):
        print(f"{idx}. {item.file_name} -> öncelik={item.priority_score}, tür={item.case_type}")


if __name__ == "__main__":
    demo()
