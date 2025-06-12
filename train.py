import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import joblib

import nltk
from nltk.corpus import stopwords

from data_utils import preprocess_extra_features
from model import build_model_with_embedding
print("charging datasets and model ...")

# Chargement des données
train_data = pd.read_csv("train.csv")
X_train, X_val, y_train, y_val = train_test_split(
    train_data,
    train_data['retweet_count'],
    test_size=0.3,
    random_state=42
)


nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    return " ".join(
        word for word in text.split() 
        if word.lower() not in stop_words
    )

# Appliquer au DataFrame
X_train['text_clean'] = X_train['text'].apply(remove_stopwords)
X_val['text_clean']   = X_val['text'].apply(remove_stopwords)

# --------- TEXTE ---------
# Tokenizer sur les tweets
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train["text_clean"])

X_train_seq = tokenizer.texts_to_sequences(X_train["text_clean"])
X_val_seq   = tokenizer.texts_to_sequences(X_val["text_clean"])

# Padding
maxlen = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding="post")
X_val_pad   = pad_sequences(X_val_seq,   maxlen=maxlen, padding="post")

# --------- FEATURES NUMÉRIQUES ---------
X_train_extra_raw = preprocess_extra_features(X_train)
X_val_extra_raw   = preprocess_extra_features(X_val)

scaler = StandardScaler()
X_train_extra_scaled = scaler.fit_transform(X_train_extra_raw)
X_val_extra_scaled   = scaler.transform(X_val_extra_raw)

# --------- MODÈLE ---------
model = build_model_with_embedding(text_maxlen=maxlen, meta_dim=X_train_extra_scaled.shape[1])


checkpoint = ModelCheckpoint(
    "final_model.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1
)
print("start training ...")

# Entraînement
history = model.fit(
    [X_train_pad, X_train_extra_scaled], y_train,
    validation_data=([X_val_pad, X_val_extra_scaled], y_val),
    epochs=20,
    batch_size=256,
    callbacks=[checkpoint],
    verbose=2
)

# Évaluation
loss, mae = model.evaluate([X_val_pad, X_val_extra_scaled], y_val, verbose=0)
print(f"Validation MAE = {mae:.4f}")

# Courbes d'apprentissage
plt.figure()
plt.plot(history.history['loss'], label='Train MAE')
plt.plot(history.history['val_loss'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.savefig("learning_curve.png")
plt.show()

# Sauvegarde des objets utiles
joblib.dump(tokenizer, "tokenizer.pkl")
print("Tokenizer saved ")
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved ")