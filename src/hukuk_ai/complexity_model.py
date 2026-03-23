"""Metin tabanlı öğrenilebilir karmaşıklık modeli."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from .information_extraction import extract_features
from .scoring import compute_complexity_score


@dataclass
class ComplexityTrainingSummary:
    model_dir: str
    train_samples: int
    train_mae: float
    train_rmse: float


class BerturkComplexityRegressor:
    """BERTurk gömlemeleri ile karmaşıklık skoru tahmini yapan lineer model."""

    def __init__(
        self,
        base_model_name: str = "dbmdz/bert-base-turkish-cased",
        model_dir: str = "models/berturk_complexity_regressor",
    ) -> None:
        self.base_model_name = base_model_name
        self.model_dir = Path(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.encoder.eval()

        self.weights: np.ndarray | None = None
        self._load_if_trained()

    def _load_if_trained(self) -> None:
        weights_file = self.model_dir / "weights.npy"
        if not weights_file.exists():
            return
        self.weights = np.load(weights_file)

    @torch.inference_mode()
    def _encode(self, text: str, max_length: int = 256) -> np.ndarray:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        outputs = self.encoder(**encoded)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        return cls_embedding.astype(np.float32)

    def train_from_csv(
        self,
        train_csv_path: str,
        text_column: str = "text",
        complexity_column: str = "complexity",
        ridge_alpha: float = 1.0,
        max_length: int = 256,
    ) -> ComplexityTrainingSummary:
        train_df = pd.read_csv(train_csv_path)
        if text_column not in train_df.columns or complexity_column not in train_df.columns:
            raise ValueError(
                f"CSV içinde '{text_column}' ve '{complexity_column}' kolonları bulunmalıdır."
            )

        train_df = train_df[[text_column, complexity_column]].dropna().copy()
        train_df[text_column] = train_df[text_column].astype(str)
        train_df[complexity_column] = pd.to_numeric(train_df[complexity_column], errors="coerce")
        train_df = train_df.dropna(subset=[complexity_column])

        if train_df.empty:
            raise ValueError("Eğitim CSV'sinde geçerli örnek bulunamadı.")

        texts = train_df[text_column].tolist()
        targets = np.clip(train_df[complexity_column].to_numpy(dtype=np.float32), 0, 100)

        embeddings = np.vstack([self._encode(text, max_length=max_length) for text in texts])
        bias = np.ones((embeddings.shape[0], 1), dtype=np.float32)
        design_matrix = np.hstack([embeddings, bias])

        identity = np.eye(design_matrix.shape[1], dtype=np.float32)
        identity[-1, -1] = 0.0

        lhs = design_matrix.T @ design_matrix + (ridge_alpha * identity)
        rhs = design_matrix.T @ targets
        self.weights = np.linalg.solve(lhs, rhs).astype(np.float32)

        preds = np.clip(design_matrix @ self.weights, 0, 100)
        mae = float(np.mean(np.abs(preds - targets)))
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

        self.model_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.model_dir / "weights.npy", self.weights)
        (self.model_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "base_model_name": self.base_model_name,
                    "text_column": text_column,
                    "complexity_column": complexity_column,
                    "ridge_alpha": ridge_alpha,
                    "train_samples": len(train_df),
                    "train_mae": round(mae, 4),
                    "train_rmse": round(rmse, 4),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return ComplexityTrainingSummary(
            model_dir=str(self.model_dir),
            train_samples=len(train_df),
            train_mae=round(mae, 4),
            train_rmse=round(rmse, 4),
        )

    def predict(self, text: str, features: Dict[str, int | str] | None = None) -> Dict[str, float | str]:
        if self.weights is None:
            extracted = features or extract_features(text)
            complexity = compute_complexity_score(
                taraf_sayisi=int(extracted["taraf_sayisi"]),
                tanik_sayisi=int(extracted["tanik_sayisi"]),
                delil_sayisi=int(extracted["delil_sayisi"]),
                metin_uzunlugu=int(extracted["metin_uzunlugu"]),
            )
            return {
                "complexity": complexity,
                "mode": "heuristic_fallback",
                "confidence": 0.0,
            }

        embedding = self._encode(text)
        vector = np.hstack([embedding, np.array([1.0], dtype=np.float32)])
        pred = float(np.clip(vector @ self.weights, 0, 100))

        return {
            "complexity": round(pred, 2),
            "mode": "learned_regression",
            "confidence": 1.0,
        }
