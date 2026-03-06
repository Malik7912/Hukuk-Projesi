"""BERTurk tabanlı dava türü sınıflandırıcı."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .config import CASE_TYPES


class BerturkCaseTypeClassifier:
    """BERTurk gömlemeleri ile prototip tabanlı sınıflandırma yapar."""

    def __init__(self, model_name: str = "dbmdz/bert-base-turkish-cased") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.label_embeddings = self._build_label_embeddings(CASE_TYPES)

    @torch.inference_mode()
    def _encode(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        outputs = self.model(**encoded)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        norm = np.linalg.norm(cls_embedding)
        return cls_embedding if norm == 0 else cls_embedding / norm

    def _build_label_embeddings(self, labels: List[str]) -> Dict[str, np.ndarray]:
        embeddings: Dict[str, np.ndarray] = {}
        for label in labels:
            prompt = f"Bu metin bir {label} içermektedir."
            embeddings[label] = self._encode(prompt)
        return embeddings

    def predict(self, text: str) -> Dict[str, float | str]:
        text_embedding = self._encode(text)
        similarity_scores = {
            label: float(np.dot(text_embedding, label_embedding))
            for label, label_embedding in self.label_embeddings.items()
        }
        predicted_label = max(similarity_scores, key=similarity_scores.get)
        confidence = similarity_scores[predicted_label]

        return {
            "predicted_case_type": predicted_label,
            "confidence": round(confidence, 4),
            "scores": similarity_scores,
        }
