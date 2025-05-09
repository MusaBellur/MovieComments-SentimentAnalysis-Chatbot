# 🎬 Film Yorumlarında Duygu Analizi Chatbotu

Bu Python tabanlı web uygulaması, Türkçe film yorumlarını analiz ederek yorumun olumlu mu yoksa olumsuz mu olduğunu belirler ve sonucu kullanıcıya bir sohbet botu aracılığıyla sunar.

## Özellikler

- Makine öğrenmesi ile duygu analizi (Logistic Regression)  
- Kullanıcı yorumlarını analiz eden chatbot arayüzü  
- Karanlık mod destekli kullanıcı dostu tasarım  
- Duygu sonucunu emoji ve açıklama ile birlikte gösterme  
- Flask tabanlı REST API  
- Model ve veri yönetimi için pickle, JSON, TF-IDF kullanımı  

## ⚙️ Kurulum

## Kurulum

1. Repoyu klonlayın:
```bash
git clone https://github.com/MusaBellur/MovieComments-SentimentAnalysis-Chatbot.git
```

2. Proje dizinine gidin:
```bash
cd MovieComments-SentimentAnalysis-Chatbot
```

3. Gereksinimleri yükleyin:
```bash
cd MovieComments-SentimentAnalysis-Chatbot
```

## Kullanım

Uygulamayı çalıştırmak için:
```bash
python chatbot.py
```
Tarayıcınızda http://localhost:5000 adresine giderek chatbot arayüzünü kullanabilirsiniz

## Proje Yapısı

film-duygu-analizi-chatbotu/
├── chatbot.py                  # Flask uygulaması ve API
├── istatistic.py              # Model istatistikleri ve değerlendirme
├── model/
│   ├── logistic_regression_model.pkl
│   └── tfidf_vectorizer.pkl
├── static/
│   └── style.css              # Arayüz tasarımı (karanlık mod)
├── templates/
│   └── index.html             # Chatbot arayüzü
├── dataset.xlsx               # Film yorumları veri seti        

## Kullanılan Teknolojiler ve Kütüphaneler

- Python
- Flask
- scikit-learn
- nltk
- pandas
- HTML/CSS
- pickle

## Gelecekteki Geliştirmeler

- Derin öğrenme tabanlı modeller (BERT, LSTM) entegrasyonu
- Çoklu dil desteği
- Daha büyük veri setleriyle yeniden eğitim
- Örnek yorumlar ve kullanıcı ipuçları
