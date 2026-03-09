"""BERTurk sınıflandırma modeli eğitimi için yardımcı fonksiyonlar."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


REQUIRED_COLUMNS = {"text", "label"}


def train_case_type_model(
    dataset_path: str,
    output_dir: str,
    model_name: str = "dbmdz/bert-base-turkish-cased",
    epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
) -> str:
    """CSV veri setiyle BERTurk tabanlı dava türü sınıflandırıcıyı eğitir.

    Veri seti `text` ve `label` sütunlarını içermelidir.
    """

    data_path = Path(dataset_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Veri seti bulunamadı: {dataset_path}")

    df = pd.read_csv(data_path)
    if not REQUIRED_COLUMNS.issubset(df.columns):
        raise ValueError(
            "CSV dosyasında 'text' ve 'label' sütunları bulunmalıdır. "
            f"Bulunan sütunlar: {list(df.columns)}"
        )

    df = df.dropna(subset=["text", "label"]).copy()
    if len(df) < 4:
        raise ValueError("Eğitim için en az 4 satır veri gereklidir.")

    labels = sorted(df["label"].astype(str).unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    split = dataset.train_test_split(test_size=0.2, seed=42)

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=256)

    tokenized_train = split["train"].map(tokenize_batch, batched=True)
    tokenized_eval = split["test"].map(tokenize_batch, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return (
        f"Eğitim tamamlandı. Model kaydedildi: {output_dir}\n"
        f"Sınıf sayısı: {len(labels)}\n"
        f"Etiketler: {', '.join(labels)}"
    )
