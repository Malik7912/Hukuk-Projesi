"""BERTurk tabanlı dava türü sınıflandırıcı."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .config import CASE_TYPES


class BerturkCaseTypeClassifier:
    """BERTurk ile ya eğitilmiş sınıflandırma ya da embedding tabanlı tahmin yapar."""

    def __init__(
        self,
        model_name: str = "dbmdz/bert-base-turkish-cased",
        finetuned_model_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.uses_finetuned_classifier = bool(
            finetuned_model_path and Path(finetuned_model_path).exists()
        )
        if self.uses_finetuned_classifier:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                finetuned_model_path
            )
            self.model.eval()
            self.id2label = {
                int(idx): label for idx, label in self.model.config.id2label.items()
            }
            self.label_embeddings: Dict[str, np.ndarray] = {}
        else:
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.id2label = {}
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
        if self.uses_finetuned_classifier:
            return self._predict_with_finetuned_model(text)

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

    @torch.inference_mode()
    def _predict_with_finetuned_model(self, text: str) -> Dict[str, float | str]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        logits = self.model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)
        predicted_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_id].item())
        predicted_label = self.id2label.get(predicted_id, str(predicted_id))
        scores = {
            self.id2label.get(idx, str(idx)): float(prob.item())
            for idx, prob in enumerate(probabilities)
        }
        return {
            "predicted_case_type": predicted_label,
            "confidence": round(confidence, 4),
            "scores": scores,
        }
