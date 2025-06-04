from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import nltk
import re

# Stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('turkish'))
important_words = {"hiç", "asla", "değil", "kötü", "beğenmedim", "nefret"}
stop_words -= important_words
stop_words.add("film")

# Temizleme fonksiyonu
def pre_processing(text):
    text = text.lower()
    text = re.sub("[^a-zçğıöşü]", " ", text)
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

# Veri setini yükle (CSV varsayımıyla)
df = pd.read_excel('dataset.xlsx') # Burada kendi veri seti yolunu gir
df["clean_text"] = df["comment"].apply(pre_processing)

# Özellik ve etiket
X = df["clean_text"]
y = df["Label"]  # Etiket kolonunun adı "etiket" ise

# Vektörleştirme
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# Eğitim ve test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model eğitimi
model = LogisticRegression()
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# Değerlendirme
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))

# Model ve vectorizer'ı kaydetme
import pickle
with open("logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
