# Hukuk-Projesi: BERTurk Eğitim + Test + Analiz Sistemi

Bu proje, `dbmdz/bert-base-turkish-cased` modelini doğrudan Hugging Face üzerinden kullanarak **dava türü sınıflandırma modeli eğitimi**, **metinden öğrenen karmaşıklık modeli eğitimi**, **test menüsü ile doğrulama** ve **dilekçe analizi** sunan araştırma amaçlı bir prototiptir.

## Arayüz Akışı

Gradio arayüzü üç sekmeden oluşur:

1. **Model Eğitimi**
   - Eğitim CSV yüklenir.
   - Opsiyonel doğrulama CSV yüklenir.
   - `text` ve `label` kolon adları girilir.
   - Epoch, batch size ve learning rate ayarlanır.
   - Fine-tuning başlatılır ve model `models/berturk_case_classifier` altına kaydedilir.

2. **Test Menüsü**
   - Etiketli test CSV yüklenir.
   - Aynı kolon adları ile model doğruluğu (accuracy) hesaplanır.

3. **Dilekçe Analizi**
   - `.txt`, `.pdf`, `.docx` dava dosyaları yüklenir.
   - Dava türü, taraf/tanık/delil, karmaşıklık ve öncelik puanı üretilir.
   - Karmaşıklık modeli eğitildiyse `learned_regression`, eğitilmediyse `heuristic_fallback` modu kullanılır.

## CSV Formatı

Eğitim ve test için CSV dosyaları en az şu kolonları içermelidir:

- `text`: Dava metni
- `label`: Dava türü etiketi (ör: `iş davası`, `kira davası`)


Karmaşıklık modeli eğitimi için ek CSV kolonları:

- `text`: Dava metni
- `complexity`: 0-100 arası hedef karmaşıklık puanı

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python app.py
```

## Notlar

- Eğitim yapılmamışsa sistem dava türü tahmininde embedding tabanlı fallback yöntemi kullanır.
- Eğitim sonrası model otomatik olarak yüklenir ve test/analiz adımlarında fine-tuned model kullanılır.
- Bu sistem hukuki danışmanlık vermez; yalnızca akademik/araştırma prototipidir.
