"""Araştırma amaçlı Türkçe dava dilekçesi analiz prototipi (BERT tabanlı)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re
from typing import Dict, List, Tuple


@dataclass
class CaseAnalysisResult:
    file_name: str
    case_type: str
    court_type: str
    party_count: int
    witness_count: int
    evidence_count: int
    text_length: int
    token_count: int
    top_keywords: List[str]
    complexity_score: float
    priority_score: float
    detected_sections: Dict[str, str]


class FileReader:
    @staticmethod
    def read_text(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
        if not data.strip():
            raise ValueError(f"Dosya boş görünüyor: {path}")
        return data


class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()


class PetitionSectionParser:
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
    PARTY_LABELS = ["davacı", "davalı", "müşteki", "şikayetçi", "vekil"]

    @classmethod
    def count_parties(cls, text: str) -> int:
        hits = 0
        for label in cls.PARTY_LABELS:
            hits += len(re.findall(rf"\b{label}\b", text, flags=re.IGNORECASE))
        return max(2, hits)

    @staticmethod
    def count_witnesses(text: str) -> int:
        patterns = [r"\btanık\b", r"\bşahit\b", r"\btanıklarımız\b"]
        return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)

    @staticmethod
    def count_evidence(text: str, sections: Dict[str, str]) -> int:
        evidence_text = sections.get("deliller", text)
        line_items = len(re.findall(r"(^|\n)\s*(?:[-*]|\d+[.)])\s+", evidence_text))
        evidence_signals = len(re.findall(r"\b(delil|sözleşme|fatura|dekont|bilirkişi|kayıt)\b", evidence_text, flags=re.IGNORECASE))
        return max(line_items, evidence_signals)

    @staticmethod
    def text_length(text: str) -> int:
        return len(text.split())


class TurkishBertNLP:
    """dbmdz/bert-base-turkish-cased ile token/embedding tabanlı analiz."""

    MODEL_NAME = "dbmdz/bert-base-turkish-cased"

    CASE_LABEL_HINTS: Dict[str, str] = {
        "kira_davası": "kira bedeli tahliye kiracı kiraya veren uyuşmazlığı",
        "alacak_davası": "alacak borç ödeme para tahsili uyuşmazlığı",
        "boşanma_davası": "boşanma evlilik birliği nafaka anlaşmazlığı",
        "velayet_davası": "velayet çocuk kişisel ilişki aile hukuku uyuşmazlığı",
        "iş_davası": "işçi işveren iş sözleşmesi mesai çalışma uyuşmazlığı",
        "kıdem_tazminatı_davası": "kıdem tazminatı ihbar tazminatı işten çıkarma",
        "işe_iade_davası": "işe iade geçersiz fesih işten çıkarma iptali",
        "tapu_davası": "tapu tescil kadastro mülkiyet taşınmaz uyuşmazlığı",
    }

    COURT_LABEL_HINTS: Dict[str, str] = {
        "iş_mahkemesi": "iş mahkemesi işçi işveren uyuşmazlıkları",
        "aile_mahkemesi": "aile mahkemesi boşanma velayet nafaka davaları",
        "asliye_hukuk_mahkemesi": "asliye hukuk mahkemesi genel hukuk davaları",
        "sulh_hukuk_mahkemesi": "sulh hukuk mahkemesi kira tahliye uyuşmazlıkları",
        "asliye_ceza_mahkemesi": "asliye ceza mahkemesi ceza yargılamaları",
    }

    STOPWORDS = {
        "ve", "ile", "bir", "bu", "için", "olan", "olarak", "davanın", "davacı", "davalı", "mahkemesi",
        "talep", "ederek", "edilmesini", "sayın", "hukuki", "sebepler", "sonuç", "açıklamalar"
    }

    def __init__(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "Bu sürüm için 'transformers' ve 'torch' gerekir. Kurulum: pip install transformers torch"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()

    def tokenize(self, text: str, max_length: int = 512):
        return self.tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.tokenize(text))

    def _embed(self, text: str, max_length: int = 256):
        encoded = self.tokenizer(text, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(**encoded)
        hidden = out.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        return summed / denom

    def semantic_label(self, text: str, label_hints: Dict[str, str]) -> str:
        doc_vec = self._embed(text)
        best_label = "belirsiz"
        best_score = -1.0
        for label, hint in label_hints.items():
            hint_vec = self._embed(hint)
            score = self.torch.nn.functional.cosine_similarity(doc_vec, hint_vec).item()
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    def extract_keywords(self, text: str, top_k: int = 8) -> List[str]:
        cleaned = re.sub(r"[^\wçğıöşüİĞÜŞÖÇ\s]", " ", text.lower())
        words = [w for w in cleaned.split() if len(w) > 2 and w not in self.STOPWORDS]

        candidates = set(words)
        for i in range(len(words) - 1):
            candidates.add(f"{words[i]} {words[i+1]}")

        if not candidates:
            return []

        doc_vec = self._embed(text)
        scored = []
        for cand in candidates:
            cand_vec = self._embed(cand, max_length=32)
            score = self.torch.nn.functional.cosine_similarity(doc_vec, cand_vec).item()
            scored.append((cand, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [cand for cand, _ in scored[:top_k]]


class ScoringEngine:
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
    def complexity_score(party_count: int, witness_count: int, evidence_count: int, text_length: int, token_count: int) -> float:
        raw = (
            party_count * 1.7
            + witness_count * 2.2
            + evidence_count * 2.5
            + math.log(max(text_length, 1), 2) * 3.5
            + math.log(max(token_count, 1), 2) * 1.8
        )
        return round(min(raw, 100), 2)

    @classmethod
    def priority_score(cls, case_type: str, complexity_score: float, text: str) -> float:
        base = cls.PRIORITY_BY_CASE_TYPE.get(case_type, cls.PRIORITY_BY_CASE_TYPE["belirsiz"])
        lowered = text.lower()
        urgency_boost = sum(weight for key, weight in cls.URGENCY_KEYWORDS.items() if key in lowered)
        score = 0.45 * base + 0.45 * complexity_score + 0.10 * urgency_boost
        return round(min(score, 100), 2)


class PetitionAnalyzer:
    def __init__(self) -> None:
        self.bert = TurkishBertNLP()

    def analyze_text(self, text: str, file_name: str = "<memory>") -> CaseAnalysisResult:
        cleaned = TextPreprocessor.clean_text(text)
        sections = PetitionSectionParser.detect_sections(cleaned)

        case_type = self.bert.semantic_label(cleaned, TurkishBertNLP.CASE_LABEL_HINTS)
        court_type = self.bert.semantic_label(cleaned, TurkishBertNLP.COURT_LABEL_HINTS)

        party_count = InformationExtractor.count_parties(cleaned)
        witness_count = InformationExtractor.count_witnesses(cleaned)
        evidence_count = InformationExtractor.count_evidence(cleaned, sections)
        text_length = InformationExtractor.text_length(cleaned)
        token_count = self.bert.token_count(cleaned)
        top_keywords = self.bert.extract_keywords(cleaned, top_k=8)

        complexity = ScoringEngine.complexity_score(
            party_count=party_count,
            witness_count=witness_count,
            evidence_count=evidence_count,
            text_length=text_length,
            token_count=token_count,
        )
        priority = ScoringEngine.priority_score(case_type=case_type, complexity_score=complexity, text=cleaned)

        return CaseAnalysisResult(
            file_name=file_name,
            case_type=case_type,
            court_type=court_type,
            party_count=party_count,
            witness_count=witness_count,
            evidence_count=evidence_count,
            text_length=text_length,
            token_count=token_count,
            top_keywords=top_keywords,
            complexity_score=complexity,
            priority_score=priority,
            detected_sections=sections,
        )

    def analyze_file(self, path: str) -> CaseAnalysisResult:
        return self.analyze_text(FileReader.read_text(path), file_name=path)

    def rank_cases(self, texts: List[Tuple[str, str]]) -> List[CaseAnalysisResult]:
        results = [self.analyze_text(text, file_name=name) for name, text in texts]
        return sorted(results, key=lambda r: r.priority_score, reverse=True)


def demo() -> None:
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
    print("--- Tek Dava Analizi (BERT) ---")
    for key, value in asdict(single_result).items():
        if key != "detected_sections":
            print(f"{key}: {value}")

    print("\n--- Token Örneği ---")
    tokenized = analyzer.bert.tokenize(sample_1)
    print("input_ids şekli:", tuple(tokenized["input_ids"].shape))
    print("attention_mask şekli:", tuple(tokenized["attention_mask"].shape))

    ranked = analyzer.rank_cases([
        ("örnek_dava_1", sample_1),
        ("örnek_dava_2", sample_2),
    ])

    print("\n--- Öncelik Sıralaması ---")
    for idx, item in enumerate(ranked, start=1):
        print(f"{idx}. {item.file_name} -> öncelik={item.priority_score}, tür={item.case_type}, mahkeme={item.court_type}")


if __name__ == "__main__":
    demo()
