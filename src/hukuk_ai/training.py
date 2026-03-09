"""BERTurk modelini eğitme ve test etme yardımcıları."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "dbmdz/bert-base-turkish-cased"


class BerturkTrainer:
    """Hugging Face BERTurk ile metin sınıflandırma eğitimi yapar."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"Eğitim verisinde şu kolonlar eksik: {sorted(missing)}")

    def _encode_dataframe(self, df: pd.DataFrame, label_to_id: Dict[str, int]) -> Dataset:
        self._validate_dataframe(df)
        prepared = df.copy()
        prepared["label"] = prepared["label"].map(label_to_id)
        if prepared["label"].isna().any():
            raise ValueError("Etiket eşleme sırasında boş değer oluştu. 'label' kolonunu kontrol edin.")

        dataset = Dataset.from_pandas(prepared[["text", "label"]], preserve_index=False)

        def tokenize(batch: Dict[str, list]) -> Dict[str, list]:
            return self.tokenizer(batch["text"], truncation=True, max_length=256)

        return dataset.map(tokenize, batched=True)

    @staticmethod
    def _compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
        }

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        output_dir: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> Dict[str, float]:
        labels = sorted(set(train_df["label"]).union(set(eval_df["label"])))
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}

        train_dataset = self._encode_dataframe(train_df, label_to_id)
        eval_dataset = self._encode_dataframe(eval_df, label_to_id)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(labels),
            id2label=id_to_label,
            label2id=label_to_id,
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self._compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        metadata = {
            "model_name": self.model_name,
            "labels": labels,
            "metrics": metrics,
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {key: float(value) for key, value in metrics.items() if isinstance(value, (float, int))}


class BerturkPredictor:
    """Eğitilmiş modeli yükleyip tekil metin veya veri seti testi yapar."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    @torch.inference_mode()
    def predict_text(self, text: str) -> Dict[str, float | str]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        logits = self.model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)
        pred_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[pred_id].item())
        label = self.model.config.id2label[pred_id]
        return {
            "label": label,
            "confidence": round(confidence, 4),
        }

    def evaluate_dataframe(self, df: pd.DataFrame) -> Dict[str, float]:
        BerturkTrainer._validate_dataframe(df)
        y_true = []
        y_pred = []
        for _, row in df.iterrows():
            result = self.predict_text(str(row["text"]))
            y_true.append(str(row["label"]))
            y_pred.append(str(result["label"]))

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }
