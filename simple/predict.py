import pandas as pd
import csv
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_utils import preprocess_extra_features

# 1. Charger le modèle et les objets
model = load_model("/home/julien/DL/Deep-learning-retweet-prediction/simple/simple_model.keras")
tokenizer = joblib.load("/home/julien/DL/Deep-learning-retweet-prediction/simple/tokenizer.pkl")
scaler = joblib.load("/home/julien/DL/Deep-learning-retweet-prediction/simple/scaler.pkl")

# 2. Charger les données d’évaluation
eval_data = pd.read_csv("/home/julien/DL/data/evaluation.csv")

# 3. Prétraitement texte
X_eval_seq = tokenizer.texts_to_sequences(eval_data["text"])
X_eval_pad = pad_sequences(X_eval_seq, maxlen=100, padding="post")  # même maxlen qu’en train

# 4. Prétraitement des features numériques
X_eval_extra_raw = preprocess_extra_features(eval_data)
X_eval_extra_scaled = scaler.transform(X_eval_extra_raw)

# 5. Prédictions
predictions = model.predict([X_eval_pad, X_eval_extra_scaled]).flatten()

# 6. Sauvegarde au format Kaggle
with open("/home/julien/DL/Deep-learning-retweet-prediction/simple/keras_predictions.csv", 'w') as f:
    print("Saving predictions to simple/keras_predictions.csv")
    writer = csv.writer(f)
    writer.writerow(["TweetID", "NoRetweets"])
    for idx, pred in enumerate(predictions):
        writer.writerow([str(eval_data['id'].iloc[idx]), str(int(pred))])
