import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from data_utils import process_text_and_features
from model import build_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

train_data = pd.read_csv("data/train.csv")
X_train, X_val, y_train, y_val = train_test_split(
    train_data,
    train_data['retweet_count'],
    test_size=0.3,
    random_state=42
)

vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
scaler = StandardScaler()

X_train_final, vectorizer, scaler = process_text_and_features(X_train, vectorizer, scaler, fit=True)
X_val_final, _, _ = process_text_and_features(X_val, vectorizer, scaler)

model = build_model(X_train_final.shape[1])

checkpoint = ModelCheckpoint("simple/simple_model.keras", monitor="val_loss", save_best_only=True, mode="min", verbose=1)

history = model.fit(
    X_train_final, y_train,
    validation_data=(X_val_final, y_val),
    epochs=20,
    batch_size=256,
    callbacks=[checkpoint],
    verbose=1
)

loss, mae = model.evaluate(X_val_final, y_val, verbose=0)
print(f"Validation MAE = {mae:.4f}")

# Courbes d'apprentissage
plt.figure()
plt.plot(history.history['loss'], label='Train MAE')
plt.plot(history.history['val_loss'], label='Val MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()
plt.savefig("learning_curve.png")
# Sauvegarde du modèle et des objets de transformation
import joblib
joblib.dump(vectorizer, "simple/tfidf_vectorizer.pkl")
joblib.dump(scaler, "simple/scaler.pkl")