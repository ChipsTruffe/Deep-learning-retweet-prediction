
import torch
import re
import string
from collections import Counter
import pandas as pd
# --- Tokenization and Vocabulary Helper Functions ---
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'



def hashtag_tokenizer(text):
    if pd.isna(text):
        return []
    return text.lower().split(",")
def space_tokenizer(text):
    """Basic word tokenizer: lowercase and split by space."""
    return text.lower().split()
def clean_tokenizer(text):
    """lowercase, removes punctuation, removes hashtags, splits by spaces, splits words longer than 6"""

    current_text = text.lower()

    # 2. Remove punctuation

    translator = str.maketrans('', '', string.punctuation)
    current_text = current_text.translate(translator)
    words = current_text.split()

    final_tokens = []
    for word in words:
        while len(word) > 4: #splits words in maximum 6 char sections
            final_tokens.append(word[:4])
            word = word[4:]
        final_tokens.append(word)
    
    return final_tokens

def build_vocab(texts, tokenizer_fn, max_vocab_size , min_freq=1):
    """Builds a vocabulary from a list of texts."""
    token_counts = Counter()
    for text in texts:
        tokens = tokenizer_fn(text)
        token_counts.update(tokens)

    vocab = {PAD_TOKEN: (0,1), UNK_TOKEN: (1,1), '#': (2,1)}
    token_id_counter = 2  # Start after PAD and UNK

    # Sort tokens by frequency, most common first
    sorted_tokens = sorted(token_counts.items(), key=lambda item: item[1], reverse=True)

    for token, count in sorted_tokens:
        if token_id_counter >= max_vocab_size:
            break
        if count >= min_freq: #filter by minimum frequency
            vocab[token] = (token_id_counter,count)
            token_id_counter += 1
    print(f"Vocabulary built with {len(vocab)} tokens (max_vocab_size was {max_vocab_size}).")
    return vocab

def texts_to_sequences(texts, vocab, tokenizer_fn, max_seq_len):
    """Converts texts to padded sequences of token IDs and generates padding masks."""
    pad_id = vocab[PAD_TOKEN]
    unk_id = vocab.get(UNK_TOKEN) 

    sequences = []
    padding_masks = []

    for text in texts:
        tokens = tokenizer_fn(text)
        token_ids = [vocab.get(token, unk_id) for token in tokens]

        # Pad or truncate
        if len(token_ids) < max_seq_len:
            # True in mask means this position IS PADDED and should be ignored
            padding_mask = [False] * len(token_ids) + [True] * (max_seq_len - len(token_ids))
            token_ids += [pad_id] * (max_seq_len - len(token_ids))
        else:
            token_ids = token_ids[:max_seq_len]
            padding_mask = [False] * max_seq_len # No padding if truncated or exact length
        
        sequences.append(token_ids)
        padding_masks.append(padding_mask)

    return torch.tensor(sequences, dtype=torch.long), torch.tensor(padding_masks, dtype=torch.bool)
