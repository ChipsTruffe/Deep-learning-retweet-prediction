import pandas as pd
import csv
from data_utils import process_text_and_features
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib

# Charger les objets nécessaires
model = load_model("simple/simple_model.keras")
vectorizer = joblib.load("simple/tfidf_vectorizer.pkl")
scaler = joblib.load("simple/scaler.pkl")

# Charger les données
eval_data = pd.read_csv("data/evaluation.csv")

# Transformer les données
X_eval_final, _, _ = process_text_and_features(eval_data, vectorizer, scaler)

# Prédictions
predictions = model.predict(X_eval_final).flatten()

# Sauvegarde des résultats
with open("simple/keras_predictions.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "NoRetweets"])
    for idx, pred in enumerate(predictions):
        writer.writerow([str(eval_data['id'].iloc[idx]), str(int(pred))])
