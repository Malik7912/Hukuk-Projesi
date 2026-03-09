# Hukuk-Projesi: BERTurk Dava Türü Eğitim ve Test Sistemi

Bu proje, Hugging Face'deki `dbmdz/bert-base-turkish-cased` modelini doğrudan kullanarak
Türkçe dava metinleri üzerinde **sınıflandırma eğitimi** ve **test** yapmanızı sağlar.

## Özellikler

- Gradio üzerinde iki aşamalı akış:
  1. **Eğitim Arayüzü**
  2. **Test Menüsü**
- CSV tabanlı fine-tuning (`text`, `label` kolonları)
- Hugging Face `Trainer` ile eğitim
- Eğitilen modeli klasöre kaydetme
- Tekil metin testi
- Test CSV ile accuracy ve macro-F1 hesaplama

## Proje Yapısı

- `app.py`: Eğitim ve test arayüzü
- `src/hukuk_ai/training.py`: BERTurk eğitim/test yardımcıları
- `requirements.txt`: Gerekli bağımlılıklar

## Veri Formatı

Eğitim, doğrulama ve test CSV dosyalarında şu kolonlar zorunludur:

- `text`: Dava metni
- `label`: Dava türü etiketi

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python app.py
```

## Arayüz Akışı

1. **Eğitim Arayüzü** sekmesine geçin.
2. Eğitim ve doğrulama CSV dosyalarını yükleyin.
3. Epoch, batch size ve learning rate ayarlarını girin.
4. **Modeli Eğit** butonuna basın.
5. **Test Menüsü** sekmesinde modeli yükleyin.
6. Tekil metin veya test CSV ile performansı ölçün.

## Not

Bu sistem araştırma/deney amaçlıdır; hukuki danışmanlık üretmez.
