# Hukuk-Projesi: Dava Dilekçesi Analiz Prototipi

Bu proje, Türkiye'deki dava dilekçelerini otomatik analiz edebilen ve dava dosyalarını öncelik puanına göre sıralayabilen **araştırma amaçlı bir yapay zeka prototipi** sunar.

## Özellikler

- Bir veya birden fazla dava dosyası yükleme (`.txt`, `.pdf`, `.docx`)
- Metin çıkarma ve temizleme
- Dilekçe bölümlerini (konu, açıklamalar, sonuç ve talep vb.) ayırt etme
- **BERTurk** tabanlı dava türü tahmini
- Bilgi çıkarma:
  - taraf sayısı
  - tanık sayısı
  - delil sayısı
  - metin uzunluğu
  - mahkeme türü
- Karmaşıklık puanı hesaplama
- Öncelik puanı hesaplama
- Çoklu dosyaları öncelik puanına göre sıralama

## Mimari

Kod, aşağıdaki modüllere ayrılmıştır:

- `src/hukuk_ai/io_utils.py`: Dosya yükleme ve metin çıkarma
- `src/hukuk_ai/preprocessing.py`: Temizleme ve bölüm ayrıştırma
- `src/hukuk_ai/classifier.py`: BERTurk tabanlı sınıflandırma
- `src/hukuk_ai/information_extraction.py`: Özellik çıkarımı
- `src/hukuk_ai/scoring.py`: Karmaşıklık/öncelik hesaplama
- `src/hukuk_ai/pipeline.py`: Uçtan uca analiz akışı
- `app.py`: Gradio kullanıcı arayüzü

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python app.py
```

Ardından açılan Gradio arayüzünde dosyalarınızı yükleyip **Analizi Başlat** butonuna tıklayın.

## Google Colab Kullanımı

Colab üzerinde şu akışla çalıştırabilirsiniz:

1. Projeyi Colab çalışma alanına yükleyin.
2. Aşağıdaki komutları çalıştırın:

```python
!pip install -r requirements.txt
!python app.py
```

> Not: Colab'da arayüze erişim için Gradio'nun verdiği public linki kullanabilirsiniz.

## Notlar

- Bu sistem hukuki danışmanlık vermez; yalnızca akademik/araştırma prototipidir.
- Dava türü sınıflandırması, BERTurk gömlemeleri üzerinden prototip benzerliği yaklaşımı ile yapılır.
- Daha yüksek doğruluk için etiketli gerçek dava veri seti ile fine-tuning önerilir.
