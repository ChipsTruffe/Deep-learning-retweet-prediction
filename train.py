import time
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from models import finalModel, myLSTM, MLP
from tokenizer import *
from utils import *



import matplotlib.pyplot as plt
import csv


# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Global hyperparameters
# 
learning_rate = 0.1
batch_size = 64
epochs = 10
tokenizer = clean_tokenizer

# Hyperparameters for regressor
mlp_hidden = [16]

# Hyperparameters for text interpretor
vocab_size = 10000  # Size of the vocabulary
dropout_rate = 0.1
seq_length = 50 # 280 characters, so about 5 characters / tokens
embedding_dim = 64

lstm_hidden_dim = 64
lstm_layers = 2
lstm_output_dim = 2
output_dim = 16 #arbitraire
max_vocab_size = 10000

#1 - Data import (from baseline)

train_data = pd.read_csv("/home/maloe/dev/SPEIT/Deep Learning/project/data/train.csv")

# Data truncation (for ressources)
N_sample = min(10000, len(train_data))

X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = split_data(train_data, N_sample)
#2 - Convert text to vectorized representation


#Build vocabulary from training text strings
print("Building vocabulary...") #long af cause doing on the full set every time (annoying)
vocab = build_vocab(X_text_train, tokenizer, vocab_size)
vocab = {key: value[0] for key, value in vocab.items()} #discard the tokens frequency
vocab_size = len(vocab) # actual vocab size

# Convert text strings to sequences of token IDs and padding masks
print("Converting texts to sequences...")
X_text_train_ids, X_text_train_masks = texts_to_sequences(X_text_train, vocab, tokenizer, seq_length)
X_text_test_ids, X_text_test_masks = texts_to_sequences(X_text_test, vocab, tokenizer, seq_length)


numerical_vec_dim = X_num_train.shape[1]
    
textInterpretor = myLSTM(vocab_size,
                        embedding_dim=embedding_dim,
                        hidden_dim= lstm_hidden_dim,
                        output_dim = lstm_output_dim,
                        num_layers = lstm_layers,
                        bidirectional= False,
                        padding_idx= vocab['<pad>']).to(device)
regressor = MLP(lstm_output_dim + numerical_vec_dim,
                mlp_hidden,
                1).to(device)

model = finalModel(textInterpretor, regressor).to(device)
randomize_model_weights(model) #to avoid having the model start at 0

optimizer = optim.Adam(model.parameters(recurse=True), lr=learning_rate)

loss_function = nn.MSELoss() 

# Trains the model
train_losses_epoch = [] # Store average loss per epoch

print("\nStarting training ...")
for epoch in range(epochs):
    t_epoch_start = time.time()
    model.train()

    current_epoch_total_loss = 0
    num_samples_processed_epoch = 0
    
    num_total_train_samples = y_train.shape[0]

    for i in range(0, num_total_train_samples, batch_size):
        optimizer.zero_grad()

        X_num_batch = X_num_train[i:i+batch_size].to(device)
        X_text_ids_batch = X_text_train_ids[i:i+batch_size].to(device) # (batch, seq_len)
        X_text_masks_batch = X_text_train_masks[i:i+batch_size].to(device) # (batch, seq_len)
        y_batch = y_train[i:i+batch_size].to(device).float()
        

        output = model((X_text_ids_batch, X_text_masks_batch),
                       X_num_batch,
                       ).flatten()
        loss = loss_function(output, y_batch)
        loss.backward()
        optimizer.step()

        current_epoch_total_loss += loss.item() * output.size(0) # loss.item() is avg loss for batch
        num_samples_processed_epoch += output.size(0)

    avg_epoch_loss = current_epoch_total_loss / num_samples_processed_epoch
    train_losses_epoch.append(np.sqrt(avg_epoch_loss))
    
    print(f'Epoch: {epoch + 1:04d} | Avg Train Loss (root MSE): {np.sqrt(avg_epoch_loss):.4f} | Time: {time.time() - t_epoch_start:.2f}s')

print('Optimization finished!')


torch.save(model.state_dict(), f"/home/maloe/dev/SPEIT/Deep Learning/project/checkpoints/finalModel.json")

# Evaluates the model
model.eval()
test_loss_total = 0
test_samples_count = 0
t_eval_start = time.time()

loss_function = nn.L1Loss() #Because it's the one being used for the validation


num_total_test_samples = y_test.shape[0]


with torch.no_grad(): 
    for i in range(0, num_total_test_samples, batch_size):
        X_num_batch = X_num_test[i:i+batch_size].to(device)
        X_text_ids_batch = X_text_test_ids[i:i+batch_size].to(device) # (batch, seq_len)
        X_text_masks_batch = X_text_test_masks[i:i+batch_size].to(device) # (batch, seq_len)
        y_batch = y_test[i:i+batch_size].to(device).float()
        

        output = model([X_text_ids_batch, X_text_masks_batch],
                       X_num_batch
                       ).flatten()
        loss = loss_function(output, y_batch)
        test_loss_total += loss.item() * output.size(0)
        test_samples_count += output.size(0)

avg_test_loss = test_loss_total / test_samples_count
print(f"\nTest Results for final model:")
print(f'    Avg Test Loss (MAE) : {avg_test_loss:.4f}')
print(f'    Avg value for test set : {float(torch.mean(y_test))}')
print(f'    Evaluation Time: {time.time() - t_eval_start:.2f}s')

# Plotting
plt.figure(figsize=(10, 6))
plt.semilogy(range(1, epochs + 1), train_losses_epoch, label="Average Train Loss per Epoch")
plt.scatter([epochs], [avg_test_loss], label=f"Final Avg Test Loss: {avg_test_loss:.4f}", c="r", zorder=5)

plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
#plt.show()


###################################
# Once we finalized our features and model we can train it using the whole training set and then produce prediction for the evaluating dataset
###################################
# Load the evaluation data
eval_data = pd.read_csv("/home/maloe/dev/SPEIT/Deep Learning/project/data/evaluation.csv")

with open("gbr_predictions.txt", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "NoRetweets"])

#Split data between text and scalar
N = None if avg_test_loss < float(torch.mean(y_test)) else batch_size #evaluate the full set only if it performs good
X_text_eval,_, X_num_eval,_,_,_ = split_data(eval_data, N_sample=N, train_size=0)
#vectorize
X_text_eval_ids, X_text_eval_masks = texts_to_sequences(X_text_eval, vocab, tokenizer, seq_length)

# Disable gradient computation for inference
with open("gbr_predictions.txt", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["TweetID", "NoRetweets"])
    with torch.no_grad():
        # Loop over the dataset in batches
        for start_idx in range(0, len(X_text_eval), batch_size):
            end_idx = min(start_idx + batch_size, len(X_text_eval))

            # Slice out the current batch and move tensors to the correct device
            batch_ids   = X_text_eval_ids[start_idx:end_idx].to(device)
            batch_masks = X_text_eval_masks[start_idx:end_idx].to(device)
            batch_num   = X_num_eval[start_idx:end_idx].to(device)

            # Run the model on this batch
            y_pred_batch = model([batch_ids, batch_masks], batch_num)

            # Move predictions back to CPU for writing
            y_pred_batch = y_pred_batch.cpu()

            # Write each prediction to the CSV file
            for i, pred in enumerate(y_pred_batch):
                absolute_idx = start_idx + i
                tweet_id = eval_data["id"].iloc[absolute_idx]
                writer.writerow([str(tweet_id), str(float(pred))])