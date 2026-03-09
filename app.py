"""Gradio tabanlı dava dilekçesi analiz, eğitim ve test arayüzü."""

from __future__ import annotations

import pandas as pd
import gradio as gr

from src.hukuk_ai.classifier import BerturkCaseTypeClassifier
from src.hukuk_ai.pipeline import CaseAnalysisPipeline

pipeline: CaseAnalysisPipeline | None = None
classifier: BerturkCaseTypeClassifier | None = None


def _get_pipeline() -> CaseAnalysisPipeline:
    global pipeline
    if pipeline is None:
        pipeline = CaseAnalysisPipeline()
    return pipeline


def _get_classifier() -> BerturkCaseTypeClassifier:
    global classifier
    if classifier is None:
        classifier = BerturkCaseTypeClassifier()
    return classifier


def train_model(
    train_file,
    validation_file,
    text_column: str,
    label_column: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> str:
    if not train_file:
        raise gr.Error("Lütfen eğitim CSV dosyasını yükleyin.")

    cls = _get_classifier()
    summary = cls.train_from_csv(
        train_csv_path=train_file.name,
        validation_csv_path=validation_file.name if validation_file else None,
        text_column=text_column,
        label_column=label_column,
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
    )

    global pipeline
    pipeline = CaseAnalysisPipeline(classifier=cls)

    return (
        "Model eğitimi tamamlandı.\n"
        f"- Model klasörü: {summary.model_dir}\n"
        f"- Eğitim örnek sayısı: {summary.train_samples}\n"
        f"- Doğrulama örnek sayısı: {summary.validation_samples}\n"
        f"- Epoch: {summary.epochs}\n"
        f"- Train loss: {summary.train_loss}\n"
        f"- Eval loss: {summary.eval_loss}"
    )


def evaluate_model(test_file, text_column: str, label_column: str) -> pd.DataFrame:
    if not test_file:
        raise gr.Error("Lütfen test CSV dosyası yükleyin.")

    metrics = _get_classifier().evaluate_csv(
        test_csv_path=test_file.name,
        text_column=text_column,
        label_column=label_column,
    )
    return pd.DataFrame([metrics])


def analyze_uploaded_files(files: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not files:
        raise gr.Error("Lütfen en az bir dosya yükleyin.")

    results = _get_pipeline().analyze_files([file.name for file in files])

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
                "Tahmin Modu": result.tahmin_modu,
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
        1) **Model Eğitimi** sekmesinde `dbmdz/bert-base-turkish-cased` modeli için fine-tuning başlatın.
        2) **Test Menüsü** sekmesinde etiketli test CSV ile doğruluk ölçün.
        3) **Dilekçe Analizi** sekmesinde gerçek dava dosyalarını analiz edin.
        """
    )

    with gr.Tab("Model Eğitimi"):
        train_file = gr.File(label="Eğitim CSV", file_types=[".csv"])
        validation_file = gr.File(label="Doğrulama CSV (opsiyonel)", file_types=[".csv"])
        text_column = gr.Textbox(label="Metin Kolon Adı", value="text")
        label_column = gr.Textbox(label="Etiket Kolon Adı", value="label")

        with gr.Row():
            epochs = gr.Slider(label="Epoch", minimum=1, maximum=10, value=2, step=1)
            batch_size = gr.Slider(label="Batch Size", minimum=2, maximum=32, value=8, step=1)
            learning_rate = gr.Number(label="Learning Rate", value=2e-5)

        train_button = gr.Button("Eğitimi Başlat", variant="primary")
        train_output = gr.Textbox(label="Eğitim Durumu", lines=8)

        train_button.click(
            fn=train_model,
            inputs=[train_file, validation_file, text_column, label_column, epochs, batch_size, learning_rate],
            outputs=[train_output],
        )

    with gr.Tab("Test Menüsü"):
        test_file = gr.File(label="Test CSV", file_types=[".csv"])
        test_text_column = gr.Textbox(label="Metin Kolon Adı", value="text")
        test_label_column = gr.Textbox(label="Etiket Kolon Adı", value="label")
        test_button = gr.Button("Testi Çalıştır")
        test_output = gr.Dataframe(label="Test Sonucu")

        test_button.click(
            fn=evaluate_model,
            inputs=[test_file, test_text_column, test_label_column],
            outputs=[test_output],
        )

    with gr.Tab("Dilekçe Analizi"):
        uploader = gr.File(
            label="Dava Dilekçesi Dosyaları",
            file_count="multiple",
            file_types=[".txt", ".pdf", ".docx"],
        )
        analyze_button = gr.Button("Analizi Başlat", variant="primary")
        summary_output = gr.Dataframe(label="Özet Sonuçlar (Öncelik Sırasına Göre)")
        detail_output = gr.Dataframe(label="Detay Sonuçlar")

        analyze_button.click(
            fn=analyze_uploaded_files,
            inputs=[uploader],
            outputs=[summary_output, detail_output],
        )

if __name__ == "__main__":
    demo.launch()
