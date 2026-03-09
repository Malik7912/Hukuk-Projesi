"""BERTurk tabanlı dava türü sınıflandırıcı ve eğitim yardımcıları."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .config import CASE_TYPES


@dataclass
class TrainingSummary:
    model_dir: str
    train_samples: int
    validation_samples: int
    epochs: int
    train_loss: float | None
    eval_loss: float | None


class PetitionTextDataset(Dataset):
    """Trainer için basit metin sınıflandırma dataset'i."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = self.labels[idx]
        return encoded


class BerturkEmbeddingClassifier:
    """Fine-tuning yoksa BERTurk gömlemeleriyle tahmin yapan yedek yaklaşım."""

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
        return {label: self._encode(f"Bu metin bir {label} içermektedir.") for label in labels}

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
            "mode": "embedding_fallback",
        }


class BerturkCaseTypeClassifier:
    """Hugging Face modelini fine-tune eder ve test/prediction sağlar."""

    def __init__(
        self,
        base_model_name: str = "dbmdz/bert-base-turkish-cased",
        model_dir: str = "models/berturk_case_classifier",
    ) -> None:
        self.base_model_name = base_model_name
        self.model_dir = Path(model_dir)
        self.embedding_fallback = BerturkEmbeddingClassifier(base_model_name)
        self.tokenizer = None
        self.model = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}
        self._load_if_trained()

    def _load_if_trained(self) -> None:
        label_map_file = self.model_dir / "label_mapping.json"
        if not (self.model_dir.exists() and label_map_file.exists()):
            return

        mapping = json.loads(label_map_file.read_text(encoding="utf-8"))
        self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
        self.label2id = {k: int(v) for k, v in mapping["label2id"].items()}
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()

    def train_from_csv(
        self,
        train_csv_path: str,
        validation_csv_path: str | None = None,
        text_column: str = "text",
        label_column: str = "label",
        epochs: int = 2,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        max_length: int = 256,
    ) -> TrainingSummary:
        train_df = pd.read_csv(train_csv_path)
        if text_column not in train_df.columns or label_column not in train_df.columns:
            raise ValueError(f"CSV içinde '{text_column}' ve '{label_column}' kolonları bulunmalıdır.")

        train_df = train_df[[text_column, label_column]].dropna().copy()
        train_df[text_column] = train_df[text_column].astype(str)
        train_df[label_column] = train_df[label_column].astype(str).str.strip().str.lower()

        val_df = pd.DataFrame(columns=[text_column, label_column])
        if validation_csv_path:
            val_df = pd.read_csv(validation_csv_path)[[text_column, label_column]].dropna().copy()
            val_df[text_column] = val_df[text_column].astype(str)
            val_df[label_column] = val_df[label_column].astype(str).str.strip().str.lower()

        unique_labels = sorted(train_df[label_column].unique().tolist())
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        train_labels = [self.label2id[label] for label in train_df[label_column].tolist()]
        val_labels = [self.label2id[label] for label in val_df[label_column].tolist() if label in self.label2id]
        val_texts = [
            text
            for text, label in zip(val_df[text_column].tolist(), val_df[label_column].tolist())
            if label in self.label2id
        ]

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=len(unique_labels),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        train_dataset = PetitionTextDataset(
            texts=train_df[text_column].tolist(),
            labels=train_labels,
            tokenizer=tokenizer,
            max_length=max_length,
        )

        eval_dataset = None
        if val_texts:
            eval_dataset = PetitionTextDataset(
                texts=val_texts,
                labels=val_labels,
                tokenizer=tokenizer,
                max_length=max_length,
            )

        self.model_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(self.model_dir / "checkpoints"),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset is not None else "no",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        train_result = trainer.train()
        eval_result = trainer.evaluate() if eval_dataset is not None else {}

        trainer.save_model(str(self.model_dir))
        tokenizer.save_pretrained(str(self.model_dir))

        mapping_payload = {
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "label2id": {k: str(v) for k, v in self.label2id.items()},
        }
        (self.model_dir / "label_mapping.json").write_text(
            json.dumps(mapping_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._load_if_trained()

        return TrainingSummary(
            model_dir=str(self.model_dir),
            train_samples=len(train_df),
            validation_samples=len(val_texts),
            epochs=epochs,
            train_loss=(train_result.training_loss if train_result is not None else None),
            eval_loss=eval_result.get("eval_loss"),
        )

    @torch.inference_mode()
    def predict(self, text: str) -> Dict[str, float | str]:
        if self.model is None or self.tokenizer is None or not self.id2label:
            return self.embedding_fallback.predict(text)

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        outputs = self.model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        predicted_idx = int(torch.argmax(probs).item())
        confidence = float(probs[predicted_idx].item())

        scores = {
            self.id2label[idx]: float(prob.item())
            for idx, prob in enumerate(probs)
        }
        return {
            "predicted_case_type": self.id2label[predicted_idx],
            "confidence": round(confidence, 4),
            "scores": scores,
            "mode": "fine_tuned",
        }

    def evaluate_csv(
        self,
        test_csv_path: str,
        text_column: str = "text",
        label_column: str = "label",
    ) -> Dict[str, float | int]:
        test_df = pd.read_csv(test_csv_path)
        if text_column not in test_df.columns or label_column not in test_df.columns:
            raise ValueError(f"CSV içinde '{text_column}' ve '{label_column}' kolonları bulunmalıdır.")

        test_df = test_df[[text_column, label_column]].dropna().copy()
        total = len(test_df)
        if total == 0:
            raise ValueError("Test CSV boş görünüyor.")

        correct = 0
        for _, row in test_df.iterrows():
            predicted = self.predict(str(row[text_column]))["predicted_case_type"]
            actual = str(row[label_column]).strip().lower()
            if str(predicted).strip().lower() == actual:
                correct += 1

        accuracy = correct / total
        return {"samples": total, "correct": correct, "accuracy": round(accuracy, 4)}
