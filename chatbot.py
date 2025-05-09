import os
import re
import nltk
import pickle
import pandas as pd
from flask import Flask, request, render_template, jsonify
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
nltk.download('punkt')

# --- Stopwords ve ön işlem fonksiyonu ---
stop_words = set(stopwords.words('turkish'))
important_words = {"hiç", "asla", "değil", "kötü", "beğenmedim", "nefret"}
stop_words -= important_words
stop_words.add("film")

def pre_processing(text):
    text = text.lower()
    text = re.sub("[^a-zçğıöşü]", " ", text)
    text = nltk.word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

# --- Model ve vektörleştiriciyi yükle veya eğit ---
model_path = "logistic_regression_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    print("📊 Model eğitiliyor...")

    df = pd.read_excel("dataset.xlsx")
    df["clean_text"] = df["comment"].apply(pre_processing)

    X_train, X_test, y_train, y_test = train_test_split(df["clean_text"], df["Label"], test_size=0.1, random_state=42)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=500))
    ])

    param_grid = {
        'clf__C': [0.1, 1, 10],
        'clf__solver': ['liblinear', 'lbfgs']
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    with open(model_path, "wb") as f:
        pickle.dump(best_model.named_steps["clf"], f)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(best_model.named_steps["tfidf"], f)

    print(f"✅ Eğitim tamamlandı. Test doğruluğu: {best_model.score(X_test, y_test):.4f}")

# --- Model ve vektörleştirici yükleniyor ---
with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# --- Flask Web App ---
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # index.html dosyası public klasöründe olmalı

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_input = data.get("message", "")
    cleaned = pre_processing(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized).max()

    sentiment = "Pozitif 😊" if prediction == 1 else "Negatif 😞"
    return jsonify({
        "text": user_input,
        "sentiment": sentiment,
        "confidence": round(float(confidence), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
