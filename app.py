"""Gradio tabanlı dava dilekçesi analiz arayüzü."""

from __future__ import annotations

import pandas as pd
import gradio as gr

from src.hukuk_ai.pipeline import CaseAnalysisPipeline

pipeline: CaseAnalysisPipeline | None = None


def analyze_uploaded_files(files: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not files:
        raise gr.Error("Lütfen en az bir dosya yükleyin.")

    global pipeline
    if pipeline is None:
        pipeline = CaseAnalysisPipeline()

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


with gr.Blocks(title="Dava Dilekçesi Analiz Prototipi") as demo:
    gr.Markdown(
        """
        # Dava Dilekçesi Analiz Prototipi
        Türkçe dava dilekçelerinden dava türü, temel özellikler, karmaşıklık ve öncelik puanı üretir.
        Desteklenen dosya türleri: **.txt, .pdf, .docx**
        """
    )

    with gr.Row():
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
