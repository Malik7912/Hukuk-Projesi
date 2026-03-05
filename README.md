# Hukuk-Projesi

Türkiye'deki dava dilekçelerini analiz etmek için geliştirilmiş araştırma amaçlı bir yapay zeka prototipi.

## Özellikler

- Dava metninden temel bölümleri ayıklama (mahkeme başlığı, davacı, davalı, konu, deliller vb.)
- Dava türü tahmini (kira, alacak, boşanma, velayet, iş, kıdem tazminatı, işe iade, tapu)
- Mahkeme türü tespiti
- Özellik çıkarımı:
  - taraf sayısı
  - tanık sayısı
  - delil sayısı
  - metin uzunluğu
- Karmaşıklık puanı hesaplama
- Öncelik puanı hesaplama
- Birden fazla davayı öncelik puanına göre sıralama

## Çalıştırma

```bash
python case_analysis_prototype.py
```

Bu komut dosya içindeki örnek metinlerle:

1. Tek dava analizi yapar
2. Birden çok davayı öncelik puanına göre sıralar

## Colab Kullanımı

Google Colab'da tek hücrede çalıştırmak için:

```python
!python case_analysis_prototype.py
```

## Not

Bu proje **akademik/araştırma amaçlı bir prototiptir**. Gerçek mahkeme süreçlerinde doğrudan kullanılmak üzere tasarlanmamıştır.
