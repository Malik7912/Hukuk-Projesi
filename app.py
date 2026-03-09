"""Gradio tabanlı eğitim + test arayüzü."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from src.hukuk_ai.pipeline import CaseAnalysisPipeline
from src.hukuk_ai.training import train_case_type_model


def start_training(
    dataset_file,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> str:
    if dataset_file is None:
        raise gr.Error("Lütfen eğitim için bir CSV dosyası yükleyin.")

    dataset_path = dataset_file.name
    if not dataset_path.endswith(".csv"):
        raise gr.Error("Eğitim verisi CSV formatında olmalıdır.")

    output_dir = output_dir.strip() or "models/berturk-case-type"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return train_case_type_model(
        dataset_path=dataset_path,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def analyze_uploaded_files(files: list, model_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not files:
        raise gr.Error("Lütfen en az bir dosya yükleyin.")

    model_dir = model_dir.strip()
    finetuned_path = model_dir if model_dir and Path(model_dir).exists() else None

    pipeline = CaseAnalysisPipeline(finetuned_model_path=finetuned_path)
    file_paths = [file.name for file in files]
    results = pipeline.analyze_files(file_paths)

    summary_rows = []
    details_rows = []

    for result in results:
        summary_rows.append(
            {
                "Dosya": result.dosya_adi,
                "Dava Türü": result.dava_turu,
                "Mahkeme Türü": result.mahkeme_turu,
                "Taraf Sayısı": result.taraf_sayisi,
                "Tanık Sayısı": result.tanik_sayisi,
                "Delil Sayısı": result.delil_sayisi,
                "Karmaşıklık Puanı": result.karmasiklik_puani,
                "Öncelik Puanı": result.oncelik_puani,
            }
        )
        details_rows.append(
            {
                "Dosya": result.dosya_adi,
                "Model Güven Skoru": result.guven_skoru,
                "Metin Uzunluğu": result.metin_uzunlugu,
                "Dava Konusu Bölümü": result.bolumler.get("dava_konusu", ""),
                "Açıklamalar Bölümü": result.bolumler.get("açıklamalar", ""),
                "Sonuç ve Talep Bölümü": result.bolumler.get("sonuç_ve_talep", ""),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(details_rows)


with gr.Blocks(title="BERTurk Dava Analiz Sistemi") as demo:
    gr.Markdown(
        """
        # BERTurk Dava Analiz Sistemi
        Bu arayüz iki adımlı çalışır:
        1. **Model Eğitimi** sekmesinde Hugging Face'ten `dbmdz/bert-base-turkish-cased` modeli ile eğitim yapılır.
        2. **Test Menüsü** sekmesinde eğitilmiş modelle dava dosyaları analiz edilir.
        """
    )

    with gr.Tab("Model Eğitimi"):
        gr.Markdown("CSV formatında (`text`, `label`) veri yükleyerek modeli eğitin.")
        train_file = gr.File(label="Eğitim Veri Seti (CSV)", file_types=[".csv"], file_count="single")
        train_output_dir = gr.Textbox(label="Model Çıktı Klasörü", value="models/berturk-case-type")
        with gr.Row():
            train_epochs = gr.Slider(label="Epoch", minimum=1, maximum=10, value=2, step=1)
            train_batch = gr.Slider(label="Batch Size", minimum=2, maximum=32, value=8, step=2)
            train_lr = gr.Number(label="Learning Rate", value=2e-5, precision=6)

        train_button = gr.Button("Modeli Eğit", variant="primary")
        train_log = gr.Textbox(label="Eğitim Çıktısı", lines=8)

        train_button.click(
            fn=start_training,
            inputs=[train_file, train_output_dir, train_epochs, train_batch, train_lr],
            outputs=[train_log],
        )

    with gr.Tab("Test Menüsü"):
        gr.Markdown("Eğitilen model klasörünü girin ve dava dilekçelerini analiz edin.")
        model_dir_input = gr.Textbox(
            label="Eğitilmiş Model Klasörü (opsiyonel)",
            value="models/berturk-case-type",
        )
        uploader = gr.File(
            label="Dava Dilekçesi Dosyaları",
            file_count="multiple",
            file_types=[".txt", ".pdf", ".docx"],
        )
        analyze_button = gr.Button("Test / Analiz Başlat", variant="primary")

        summary_output = gr.Dataframe(label="Özet Sonuçlar (Öncelik Sırasına Göre)")
        detail_output = gr.Dataframe(label="Detay Sonuçlar")

        analyze_button.click(
            fn=analyze_uploaded_files,
            inputs=[uploader, model_dir_input],
            outputs=[summary_output, detail_output],
        )

if __name__ == "__main__":
    demo.launch()
