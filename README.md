# Hukuk-Projesi

Türkiye'deki dava dilekçelerini **BERT tabanlı yapay zeka** ile analiz eden araştırma prototipi.

Bu sürümde ana model: **`dbmdz/bert-base-turkish-cased`**

## Neler yapar?

- Dava metninden bölümleri ayıklar (mahkeme başlığı, davacı, davalı, konu, deliller vb.)
- `dbmdz/bert-base-turkish-cased` ile:
  - anlamsal dava türü tahmini
  - anlamsal mahkeme türü tahmini
  - yapay zeka destekli anahtar kelime çıkarımı (`top_keywords`)
- Bilgi çıkarımı:
  - taraf sayısı
  - tanık sayısı
  - delil sayısı
  - metin uzunluğu (kelime)
  - token sayısı (BERT tokenizer)
- Karmaşıklık puanı
- Öncelik puanı
- Birden fazla davayı öncelik puanına göre sıralama

## Gerekli dosyalar ve türleri

- `case_analysis_prototype.py` → Python kaynak kodu (ana analiz modülü)
- `requirements.txt` → Python bağımlılık listesi
- Dava giriş dosyaları → `.txt` (UTF-8 önerilir)

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python case_analysis_prototype.py
```

## Token kısmı (istenen)

Kodda token işlemleri doğrudan BERT tokenizer ile yapılır:

- `token_count`: metnin toplam BERT token adedi
- demo çıktısında `input_ids` ve `attention_mask` tensor şekilleri yazdırılır

Teknik olarak:
- tokenizer: `AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")`
- model: `AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")`

## Colab için hızlı kullanım

```python
!pip install -r requirements.txt
!python case_analysis_prototype.py
```

## Not

Bu proje eğitim/araştırma amaçlıdır; gerçek yargısal karar sistemi değildir.
