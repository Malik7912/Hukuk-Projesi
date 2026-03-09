"""BERTurk eğitim ve test arayüzü."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from src.hukuk_ai.training import BerturkPredictor, BerturkTrainer

predictor: BerturkPredictor | None = None


def _read_csv(file) -> pd.DataFrame:
    if file is None:
        raise gr.Error("Lütfen bir CSV dosyası yükleyin.")
    return pd.read_csv(file.name)


def train_model(
    train_file,
    eval_file,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> str:
    train_df = _read_csv(train_file)
    eval_df = _read_csv(eval_file)

    model_path = output_dir.strip() or "trained_models/berturk-finetuned"
    trainer = BerturkTrainer()
    metrics = trainer.train(
        train_df=train_df,
        eval_df=eval_df,
        output_dir=model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    return (
        "Eğitim tamamlandı.\n"
        f"Model dizini: {Path(model_path).resolve()}\n"
        f"Doğruluk: {metrics.get('eval_accuracy', 0):.4f}\n"
        f"Macro F1: {metrics.get('eval_macro_f1', 0):.4f}"
    )


def load_model(model_dir: str) -> str:
    global predictor
    if not model_dir.strip():
        raise gr.Error("Model dizini boş bırakılamaz.")
    predictor = BerturkPredictor(model_dir.strip())
    return f"Model yüklendi: {Path(model_dir).resolve()}"


def test_single_text(text: str) -> str:
    if predictor is None:
        raise gr.Error("Önce modeli yükleyin.")
    if not text.strip():
        raise gr.Error("Lütfen test metni girin.")

    prediction = predictor.predict_text(text)
    return (
        f"Tahmin: {prediction['label']}\n"
        f"Güven: {prediction['confidence']:.4f}"
    )


def test_dataset(test_file) -> str:
    if predictor is None:
        raise gr.Error("Önce modeli yükleyin.")

    test_df = _read_csv(test_file)
    metrics = predictor.evaluate_dataframe(test_df)
    return (
        "Test tamamlandı.\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )


with gr.Blocks(title="BERTurk Dava Türü Eğitimi") as demo:
    gr.Markdown(
        """
        # BERTurk Dava Türü Eğitim ve Test Arayüzü
        Bu arayüz, Hugging Face'den `dbmdz/bert-base-turkish-cased` modelini kullanarak
        dava türü sınıflandırma eğitimi yapar ve test menüsü üzerinden doğrulama sunar.

        CSV dosyalarında zorunlu kolonlar:
        - `text`: dava metni
        - `label`: dava türü etiketi
        """
    )

    with gr.Tab("1) Eğitim Arayüzü"):
        train_file = gr.File(label="Eğitim CSV", file_types=[".csv"])
        eval_file = gr.File(label="Doğrulama CSV", file_types=[".csv"])
        output_dir = gr.Textbox(label="Model Çıkış Dizini", value="trained_models/berturk-finetuned")

        with gr.Row():
            epochs = gr.Slider(label="Epoch", minimum=1, maximum=10, step=1, value=3)
            batch_size = gr.Slider(label="Batch Size", minimum=2, maximum=32, step=2, value=8)
            learning_rate = gr.Number(label="Learning Rate", value=2e-5)

        train_btn = gr.Button("Modeli Eğit", variant="primary")
        train_output = gr.Textbox(label="Eğitim Çıktısı", lines=6)
        train_btn.click(
            fn=train_model,
            inputs=[train_file, eval_file, output_dir, epochs, batch_size, learning_rate],
            outputs=[train_output],
        )

    with gr.Tab("2) Test Menüsü"):
        model_dir = gr.Textbox(label="Eğitilmiş Model Dizini", value="trained_models/berturk-finetuned")
        load_btn = gr.Button("Modeli Yükle")
        load_output = gr.Textbox(label="Model Durumu", lines=2)
        load_btn.click(fn=load_model, inputs=[model_dir], outputs=[load_output])

        test_text = gr.Textbox(label="Tekil Metin Testi", lines=8)
        test_text_btn = gr.Button("Metni Test Et")
        text_output = gr.Textbox(label="Tahmin")
        test_text_btn.click(fn=test_single_text, inputs=[test_text], outputs=[text_output])

        test_file = gr.File(label="Test CSV", file_types=[".csv"])
        test_file_btn = gr.Button("Test Veri Setini Çalıştır")
        test_file_output = gr.Textbox(label="Test Sonucu", lines=4)
        test_file_btn.click(fn=test_dataset, inputs=[test_file], outputs=[test_file_output])

if __name__ == "__main__":
    demo.launch()
