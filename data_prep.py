# data_prep.py

import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords

# 1) Chargement des stopwords (français + anglais)
nltk.download("stopwords", quiet=True)
french_sw = set(stopwords.words("french"))
english_sw = set(stopwords.words("english"))
all_stopwords = french_sw.union(english_sw)

def simple_tokenizer(text):
    """
    - minuscules
    - suppression des URLs
    - extraction des mots alphanumériques
    - suppression des stopwords
    - si aucun token, retourne ["<unk>"]
    """
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    tokens = re.findall(r"\w+", text)
    filtered = [tok for tok in tokens if tok not in all_stopwords]
    return filtered if filtered else ["<unk>"]


def build_data_loaders(csv_path, batch_size=1024):
    """
    Lit le fichier CSV, crée le vocabulaire et les DataLoaders (train/valid/test).
    Renvoie : train_loader, valid_loader, test_loader, pad_idx, unk_idx
    """
    # --- 1) Chargement du CSV et création de "full_text" ---
    df = pd.read_csv(csv_path)
    df = df.sample(n=50000, random_state=42).reset_index(drop=True) 
    df["full_text"] = df["text"].fillna("")

    # --- 2) Labels et variables numériques ---
    y = df["retweet_count"].values.astype(np.float32)
    num_cols = ["user_verified", "user_statuses_count", "user_followers_count", "user_friends_count"]
    df["user_verified"] = df["user_verified"].astype(int)
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols].values).astype(np.float32)

    # --- 3) Construction du vocabulaire (topk mots) ---
    from collections import Counter
    counter = Counter(tok for txt in df["full_text"] for tok in simple_tokenizer(txt))
    specials = ["<pad>", "<unk>"]
    topk = 30000
    most_common_tokens = [tok for tok, freq in counter.most_common(topk)]
    itos = specials + most_common_tokens
    stoi = {tok: i for i, tok in enumerate(itos)}
    pad_idx, unk_idx = stoi["<pad>"], stoi["<unk>"]

    # --- 4) Définition du Dataset ---
    class TweetDataset(Dataset):
        def __init__(self, texts, numerics, labels):
            self.texts = texts
            self.numerics = numerics
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            toks = simple_tokenizer(self.texts[i])
            token_ids = [stoi.get(t, unk_idx) for t in toks]
            return {
                "tokens": torch.tensor(token_ids, dtype=torch.long),
                "numerics": torch.tensor(self.numerics[i], dtype=torch.float32),
                "label": torch.tensor(self.labels[i], dtype=torch.float32),
            }

    def collate_fn(batch):
        seqs = [b["tokens"] for b in batch]
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
        nums = torch.stack([b["numerics"] for b in batch])
        labs = torch.stack([b["label"] for b in batch])
        return padded, lengths, nums, labs

    # --- 5) Création du Dataset complet et découpe (80/10/10) ---
    texts = df["full_text"].tolist()
    full_ds = TweetDataset(texts, X_num, y)
    n_train = int(0.8 * len(full_ds))
    n_valid = int(0.1 * len(full_ds))
    n_test = len(full_ds) - n_train - n_valid
    train_ds, valid_ds, test_ds = random_split(
        full_ds, [n_train, n_valid, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    #print les moyennes des retweet_count pour chaque split
    print("Train mean retweet_count:", np.mean(y[train_ds.indices]))
    print("Valid mean retweet_count:", np.mean(y[valid_ds.indices]))
    print("Test mean retweet_count:", np.mean(y[test_ds.indices]))

    # --- 6) DataLoaders ---
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    vocab_size = len(itos)
    print("data preparation complete")
    return train_loader, valid_loader, test_loader, pad_idx, unk_idx, vocab_size