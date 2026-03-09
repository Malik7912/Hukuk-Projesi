# Hukuk-Projesi: BERTurk Eğitim + Test Arayüzü

Bu proje, Türkiye'deki dava dilekçelerini analiz etmek için Hugging Face'den
`dbmdz/bert-base-turkish-cased` modelini kullanan araştırma amaçlı bir prototiptir.

## Özellikler

- **Model Eğitimi sekmesi** ile CSV veri setinden (`text`, `label`) doğrudan fine-tuning
- **Test Menüsü sekmesi** ile eğitilen model üzerinden dava dosyası analizi
- Bir veya birden fazla dosya yükleme (`.txt`, `.pdf`, `.docx`)
- Metin çıkarma ve temizleme
- Dilekçe bölümlerini (konu, açıklamalar, sonuç ve talep) ayırt etme
- Bilgi çıkarma: taraf sayısı, tanık sayısı, delil sayısı, metin uzunluğu, mahkeme türü
- Karmaşıklık puanı ve öncelik puanı hesaplama
- Çoklu dosyaları öncelik puanına göre sıralama

## Mimari

- `src/hukuk_ai/training.py`: BERTurk model eğitimi (Trainer API)
- `src/hukuk_ai/classifier.py`: Eğitilmiş model varsa sınıflandırma, yoksa temel BERTurk tahmini
- `src/hukuk_ai/pipeline.py`: Uçtan uca analiz akışı
- `src/hukuk_ai/io_utils.py`: Dosya okuma
- `src/hukuk_ai/preprocessing.py`: Metin temizleme + bölüm çıkarımı
- `src/hukuk_ai/information_extraction.py`: Özellik çıkarımı
- `src/hukuk_ai/scoring.py`: Karmaşıklık/öncelik hesaplama
- `app.py`: Gradio arayüzü (Model Eğitimi + Test Menüsü)

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python app.py
```

## Arayüz Akışı

1. **Model Eğitimi** sekmesinde eğitim CSV dosyasını yükleyin.
2. `Modeli Eğit` butonu ile modeli örneğin `models/berturk-case-type` klasörüne kaydedin.
3. **Test Menüsü** sekmesine geçin.
4. Model klasörünü girin ve dava dilekçelerini yükleyin.
5. `Test / Analiz Başlat` butonu ile sonuçları görüntüleyin.

## Eğitim Verisi Formatı

CSV dosyasında en az şu sütunlar olmalıdır:

- `text`: Dilekçe metni
- `label`: Dava türü etiketi

Örnek:

```csv
text,label
"... dilekçe metni ...","boşanma davası"
"... dilekçe metni ...","iş davası"
```

## Not

Bu sistem hukuki danışmanlık vermez; yalnızca eğitim ve araştırma amaçlı prototiptir.
