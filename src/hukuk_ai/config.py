"""Sistem genelinde kullanılan sabitler ve yapılandırmalar."""

CASE_TYPES = [
    "kira davası",
    "alacak davası",
    "boşanma davası",
    "velayet davası",
    "iş davası",
    "kıdem tazminatı davası",
    "işe iade davası",
    "tapu davası",
]

COURT_PATTERNS = {
    "asliye hukuk mahkemesi": ["asliye hukuk mahkemesi"],
    "aile mahkemesi": ["aile mahkemesi"],
    "iş mahkemesi": ["iş mahkemesi"],
    "sulh hukuk mahkemesi": ["sulh hukuk mahkemesi"],
    "icra hukuk mahkemesi": ["icra hukuk mahkemesi"],
    "idare mahkemesi": ["idare mahkemesi"],
    "tüketici mahkemesi": ["tüketici mahkemesi"],
}

EVIDENCE_KEYWORDS = [
    "delil",
    "tanık",
    "bilirkişi",
    "kamera kaydı",
    "whatsapp yazışması",
    "fatura",
    "dekont",
    "sözleşme",
    "tapu kaydı",
]

URGENCY_KEYWORDS = {
    "ihtiyati tedbir": 15,
    "acil": 10,
    "çocuk": 8,
    "şiddet": 12,
    "gecikmesinde sakınca": 10,
    "maaş": 7,
    "nafaka": 8,
}

CASE_BASE_PRIORITY = {
    "boşanma davası": 65,
    "velayet davası": 70,
    "işe iade davası": 68,
    "iş davası": 60,
    "kıdem tazminatı davası": 62,
    "alacak davası": 55,
    "kira davası": 58,
    "tapu davası": 57,
}
