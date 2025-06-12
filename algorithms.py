import pandas as pd
from torch.nn import L1Loss as loss
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import joblib
import os

from data_utils import preprocess_extra_features


train_path = 'train.csv'
evaluation_path = 'evaluation.csv'

train_data = pd.read_csv(train_path) 
"""
For each tweet, we have a set of given features.

id: Its an incremental number that is used to identify each tweet
timestamp: The time (in timestamp format) that the tweet was published
retweet_count: the actual number of retweets the tweet receive at the time of the of the crawl(only in training dataset)
user_verified: A boolean field indicating if the user has been verified by Twitter
user_statuses_count: The total number of tweets (statuses) the user has published
user_followers_count: The number of followers the user has
user_friends_count: The number of users that the user is following, 
user_mentions: The users that are mentioned within the tweet (e.g. "@someuser"). The entries are comma separated (e.g. "someuser,anotheruser"). If nothing is mentioned the field is empty (nan in pandas)
urls: The URLs that are included within the tweet. The entries are comma separated (e.g. "someurl.com,anotherurl.fr") If nothing is mentioned the field is empty (nan in pandas)
hashtags: The hashtags that are included within the tweet. The entries are comma separated (e.g. "hashtag1,hashtag2") If nothing is mentioned the field is empty (nan in pandas)
text: The text of the tweet as posted by the user
"""
# Load evaluation data
evaluation_data = pd.read_csv(evaluation_path)

#dataset truncation (for performance)
N_sample = min(train_data.shape[0], 100000) 
train_data = train_data[:N_sample]



print("charging datasets and model ...")



X_train, X_test, y_train, y_test  =train_test_split(train_data, train_data["retweet_count"], train_size = 0.8, random_state=42)

print("done")

# --------- TEXTE ---------
# Tokenizer sur les tweets
print("tokenizing...")
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train["text"])




X_train_seq = tokenizer.texts_to_sequences(X_train["text"])
X_train_seq = [sorted(vec) for vec in X_train_seq]
X_test_seq = tokenizer.texts_to_sequences(X_test["text"])
X_test_seq = [sorted(vec) for vec in X_test_seq]
print("done")
# Padding
maxlen = 10
X_train_pad = np.array(pad_sequences(X_train_seq, maxlen=maxlen, padding="post"))
X_test_pad = np.array(pad_sequences(X_test_seq, maxlen=maxlen, padding="post"))
# -------------Eval data -----------
# Prepare X_eval from evaluation_data
X_eval = evaluation_data.copy()

# Tokenizer sur les tweets pour X_eval
X_eval_seq = tokenizer.texts_to_sequences(X_eval["text"])
X_eval_pad = np.array(pad_sequences(X_eval_seq, maxlen=maxlen, padding="post"))

# --------- FEATURES NUMÉRIQUES ---------
X_train_extra_raw = preprocess_extra_features(X_train)
X_test_extra_raw = preprocess_extra_features(X_test)
X_eval_extra_raw = preprocess_extra_features(X_eval)

scaler = StandardScaler()
X_train_extra_scaled = np.array(scaler.fit_transform(X_train_extra_raw))
X_test_extra_scaled = np.array(scaler.fit_transform(X_test_extra_raw))
X_eval_extra_scaled = np.array(scaler.fit_transform(X_eval_extra_raw))



"""X_train = np.concatenate([X_train_pad,X_train_extra_scaled], axis=1)
X_test = np.concatenate([X_test_pad,X_test_extra_scaled],axis=1)
X_eval = np.concatenate([X_eval_pad,X_eval_extra_raw],axis=1)"""

X_train = X_train_extra_scaled
X_test = X_test_extra_scaled
X_eval = X_eval_extra_scaled

# Impute missing values (if any)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_eval = imputer.transform(X_eval)

# Train model
print("training...")
#svm = RandomForestRegressor()
svm = SVR(kernel='rbf')
svm.fit(X_train, y_train)
print("done")

#approximate the loss

test_preds = svm.predict(X_test)
test_preds = np.round(test_preds).astype(int)

print("test loss:", np.mean(np.abs(test_preds-y_test)))
# Predict
preds = svm.predict(X_eval)
preds = np.round(preds).astype(int)

# Output
output = pd.DataFrame({
    'TweetId': evaluation_data['id'],
    'NoRetweets': preds
})
output.to_csv('prediction_algo.csv', index=False)