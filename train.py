import time
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
from datasetAnalysis import import_data, import_numerical_data
from models import MLP, CombinedModelWithMLP 
from tokenizer import *
import math


import matplotlib.pyplot as plt

# Select mode: "MLP" or "transformer"
#mode = "MLP"
mode = "transformer"

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Global hyperparameters
batch_size = 64
epochs = 1 # Shared epochs

# Hyperparameters for MLP
mlp_n_hidden_1 = 64 
mlp_n_hidden_2 = 32
mlp_n_hidden_3 = 16 
mlp_learning_rate = 0.005

# Hyperparameters for transformer + MLP
trans_learning_rate = 0.01 # Can be different from MLP's
vocab_size = 1000  # Size of the vocabulary
num_heads = 4      # Number of attention heads in Transformer
hidden_dim = 32  # Hidden dimension (nhid) for Transformer
num_layers = 3     # Number of Transformer encoder layers
dropout_rate = 0.1
numerical_vec_dim = 10 # Dimension of the numerical input vector for Transformer mode
mlp_layers_for_transformer = [64, 32]  # Hidden layers for the MLP part of CombinedModelWithMLP
seq_length = 50 # 280 characters, so about 5 characters / tokens
max_vocab_size = 10000

# Data loading and preprocessing
if mode == "MLP":
    X_all, y_all = import_numerical_data("/home/maloe/dev/SPEIT/Deep Learning/project/data/train.csv")
    # Convert to PyTorch Tensors
    X_all = torch.tensor(X_all, dtype=torch.float32)
    y_all = torch.tensor(y_all, dtype=torch.float32)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.1, random_state=42)
    N_train = min(100000, X_train.shape[0])
    N_test = min(10000, X_test.shape[0])
    X_train, y_train = X_train[:N_train], y_train[:N_train]
    X_test, y_test = X_test[:N_test], y_test[:N_test]

    model = MLP(X_train.shape[1], [mlp_n_hidden_1, mlp_n_hidden_2, mlp_n_hidden_3], 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=mlp_learning_rate)
    plot_title = f"Model: MLP, Hidden Dims: ({mlp_n_hidden_1},{mlp_n_hidden_2},{mlp_n_hidden_3})"

elif mode == "transformer":
    X_num_all, X_text_string_all, y_all = import_data("/home/maloe/dev/SPEIT/Deep Learning/project/data/train.csv")
    # Convert to PyTorch Tensors
    X_num_all = torch.tensor(X_num_all, dtype=torch.float32)
    y_all = torch.tensor(y_all, dtype=torch.float32) # Target variable

    X_num_train, X_num_test, \
    X_text_train_strings, X_text_test_strings, \
    y_train, y_test = train_test_split(
        X_num_all, X_text_string_all, y_all, test_size=0.1, random_state=42)
    
     # Build vocabulary from training text strings
    print("Building vocabulary for transformer mode...")
    vocab = build_vocab(X_text_train_strings, tokenizer, max_vocab_size)
    actual_ntoken = len(vocab) # actual vocab size

    # Convert text strings to sequences of IDs and padding masks
    print("Converting texts to sequences for transformer mode...")
    X_text_train_ids, X_text_train_masks = texts_to_sequences(X_text_train_strings, vocab, tokenizer, seq_length)
    X_text_test_ids, X_text_test_masks = texts_to_sequences(X_text_test_strings, vocab, tokenizer, seq_length)

    # Data truncation
    N_train = min(1000, y_train.shape[0])
    N_test = min(10000, y_test.shape[0])

    X_num_train, X_text_train_ids, X_text_train_masks, y_train = \
        X_num_train[:N_train], X_text_train_ids[:N_train], X_text_train_masks[:N_train], y_train[:N_train]
    X_num_test, X_text_test_ids, X_text_test_masks, y_test = \
        X_num_test[:N_test], X_text_test_ids[:N_test], X_text_test_masks[:N_test], y_test[:N_test]
    
    if X_num_train.shape[1] != numerical_vec_dim:
        print(f"Warning: numerical_vec_dim ({numerical_vec_dim}) does not match X_num_train feature dim ({X_num_train.shape[1]}). Using actual dim: {X_num_train.shape[1]}")
        current_numerical_vec_dim = X_num_train.shape[1]
    else:
        current_numerical_vec_dim = numerical_vec_dim
        
    model = CombinedModelWithMLP(
        ntoken=actual_ntoken,
        nhead=num_heads,
        nhid=hidden_dim,
        nlayers=num_layers,
        numerical_input_dim=current_numerical_vec_dim,
        mlp_hidden_dims=mlp_layers_for_transformer,
        dropout=dropout_rate
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=trans_learning_rate)
    plot_title = f"Model: Transformer+MLP, vocab: {actual_ntoken}, hid_dim: {hidden_dim}, layers: {num_layers}"
else:
    raise ValueError("Mode not recognized. Choose 'MLP' or 'transformer'.")



loss_function = nn.MSELoss() 

# Trains the model
train_losses_epoch = [] # Store average loss per epoch

print(f"\nStarting training for {mode} model...")
for epoch in range(epochs):
    t_epoch_start = time.time()
    model.train()

    current_epoch_total_loss = 0
    num_samples_processed_epoch = 0
    
    # Determine loop range based on mode
    if mode == "MLP":
        num_total_train_samples = X_train.shape[0]
    elif mode == "transformer":
        num_total_train_samples = y_train.shape[0]

    for i in range(0, num_total_train_samples, batch_size):
        optimizer.zero_grad()
        
        if mode == "MLP":
            X_batch = X_train[i:i+batch_size].to(device)
            y_batch = y_train[i:i+batch_size].to(device).float() # Ensure target is float
            
            output = model(X_batch).flatten()
        
        elif mode == "transformer":
            X_num_batch = X_num_train[i:i+batch_size].to(device)
            X_text_ids_batch = X_text_train_ids[i:i+batch_size].to(device) # (batch, seq_len)
            X_text_masks_batch = X_text_train_masks[i:i+batch_size].to(device) # (batch, seq_len)
            y_batch = y_train[i:i+batch_size].to(device).float()
            
            X_text_ids_batch_transposed = X_text_ids_batch.transpose(0, 1) # (seq_len, batch) for embedding
            
            # src_mask is for causal masking, usually None for encoders unless specific need
            # src_key_padding_mask is what we use for padded tokens
            output = model(text_src_ids=X_text_ids_batch_transposed, 
                           num_src=X_num_batch, 
                           src_key_padding_mask=X_text_masks_batch).flatten()
        loss = loss_function(output, y_batch)
        loss.backward()
        optimizer.step()

        current_epoch_total_loss += loss.item() * output.size(0) # loss.item() is avg loss for batch
        num_samples_processed_epoch += output.size(0)

    avg_epoch_loss = current_epoch_total_loss / num_samples_processed_epoch
    train_losses_epoch.append(avg_epoch_loss)
    
    print(f'Epoch: {epoch + 1:04d} | Avg Train Loss: {avg_epoch_loss:.4f} | Time: {time.time() - t_epoch_start:.2f}s')

print('Optimization finished!')


torch.save(model.state_dict(), f"/home/maloe/dev/SPEIT/Deep Learning/project/checkpoints/{mode}.json")

# Evaluates the model
model.eval()
test_loss_total = 0
test_samples_count = 0
t_eval_start = time.time()

loss_function = nn.L1Loss() #Because it's the one being used for the validation

if mode == "MLP":
    num_total_test_samples = X_test.shape[0]
elif mode == "transformer":
    num_total_test_samples = y_test.shape[0]


with torch.no_grad(): 
    for i in range(0, num_total_test_samples, batch_size):
        if mode == "MLP":
            X_batch = X_test[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device).float()
            output = model(X_batch).flatten()
        elif mode == "transformer":
            X_num_batch = X_num_test[i:i+batch_size].to(device)
            X_text_batch = X_text_test_ids[i:i+batch_size].to(device)
            X_text_masks_batch = X_text_test_masks[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device).float()
            X_text_batch = X_text_batch.transpose(0, 1)

            output = model(text_src_ids=X_text_batch, 
                           num_src=X_num_batch, 
                           src_key_padding_mask=X_text_masks_batch).flatten()

        loss = loss_function(output, y_batch)
        test_loss_total += loss.item() * output.size(0)
        test_samples_count += output.size(0)

avg_test_loss = test_loss_total / test_samples_count
print(f"\nTest Results for {mode} model:")
print(f'  Avg Test Loss : {avg_test_loss:.4f}')
print(f'  Evaluation Time: {time.time() - t_eval_start:.2f}s')

# Plotting
plt.figure(figsize=(10, 6))
plt.semilogy(range(1, epochs + 1), train_losses_epoch, label="Average Train Loss per Epoch")
plt.scatter([epochs], [avg_test_loss], label=f"Final Avg Test Loss: {avg_test_loss:.4f}", c="r", zorder=5)
plt.title(plot_title)
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.show()
