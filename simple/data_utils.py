import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def count_items(cell):
    if pd.isnull(cell) or cell.strip() == "":
        return 0
    return len(cell.split(','))

def preprocess_extra_features(df):
    return pd.DataFrame({
        "user_verified": df["user_verified"].astype(int),
        "user_statuses_count": df["user_statuses_count"],
        "user_followers_count": df["user_followers_count"],
        "user_friends_count": df["user_friends_count"],
        "user_mentions_count": df["user_mentions"].apply(count_items),
        "urls_count": df["urls"].apply(count_items),
        "hashtags_count": df["hashtags"].apply(count_items)
    })

def process_text_and_features(df, vectorizer, scaler, fit=False):
    X_tfidf = vectorizer.transform(df['text']) if not fit else vectorizer.fit_transform(df['text'])
    X_dense = X_tfidf.toarray()

    extra = preprocess_extra_features(df)
    X_scaled = scaler.transform(extra) if not fit else scaler.fit_transform(extra)

    return np.hstack((X_dense, X_scaled)), vectorizer, scaler
