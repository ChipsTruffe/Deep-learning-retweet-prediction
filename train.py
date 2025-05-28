import time
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
from datasetAnalysis import import_data, import_numerical_data
from models import MLP, CombinedModelWithMLP
import matplotlib.pyplot as plt


mode = "transformer"
#mode = "MLP"

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters for MLP
epochs = 20
batch_size = 64
n_hidden_1 = 6
n_hidden_2 = 6
n_hidden_3 = 6
learning_rate = 0.005

# Hyperparameters for transformer + MLP

vocab_size = 1000  # Size of the vocabulary
num_heads = 4      # Number of attention heads in Transformer
hidden_dim = 128   # Hidden dimension (nhid)
num_layers = 3     # Number of Transformer encoder layers
dropout_rate = 0.1
numerical_vec_dim = 10 # Dimension of the numerical input vector
mlp_layers = [64, 32]  # Hidden layers for the MLP

batch_size = 8
seq_length = 20

if mode == "MLP":
    X,y = import_numerical_data("/home/maloe/dev/SPEIT/Deep Learning/project/data/train.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
elif mode == "transformer":
    X_num, X_text,y = import_data("/home/maloe/dev/SPEIT/Deep Learning/project/data/train.csv")
    X_num_train, X_num_test, y_train, y_test = train_test_split([X_num, X_text], y, test_size=0.1)



N_train = min(100000,len(X_train))
N_test = min(10000, len(X_test))

X_train = X_train[:N_train]
y_train = y_train[:N_train]
X_test = X_test[:N_test]
y_test = y_test[:N_test]

# Initializes model and optimizer

if mode == "MLP":
    model = MLP(X_train.shape[1],[n_hidden_1,n_hidden_2,n_hidden_3],1).to(device)
elif mode == "transformer":
    model = CombinedModelWithMLP(
            ntoken=vocab_size,
            nhead=num_heads,
            nhid=hidden_dim,
            nlayers=num_layers,
            numerical_input_dim=numerical_vec_dim,
            mlp_hidden_dims=mlp_layers,
            dropout=dropout_rate
        )
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.L1Loss()

# Trains the model
losses = [0]*epochs
for epoch in range(epochs):
    t = time.time()
    model.train()

    train_loss = 0
    count = 0
    for i in range(0, N_train, batch_size):
        X_batch = X_train[i:i+batch_size].to(device)
        y_batch = y_train[i:i+batch_size].to(device)

        optimizer.zero_grad()
        output = model(X_batch).flatten()
        loss = loss_function(output, y_batch)
        loss.backward()
        count += output.size(0)
        train_loss += loss.item() * output.size(0)
        losses[epoch] = train_loss / count
        optimizer.step()

    #if epoch % 10 == 0:
    if True:
        print('Epoch: {:04d}'.format(epoch + 1),
              'L1 Loss: {:.4f}'.format(train_loss / count),
              #'acc_train: {:.4f}'.format(correct / count),
              'time: {:.4f}s'.format(time.time() - t))

print('Optimization finished!')



# Evaluates the model
model.eval()
test_loss = 0
count = 0
for i in range(0, N_test, batch_size):
    X_batch = X_test[i:i+batch_size].to(device)
    y_batch = y_test[i:i+batch_size].to(device)

    output = model(X_batch).flatten()
    loss = loss_function(output, y_batch)
    test_loss += loss.item() * output.size(0)
    count += output.size(0)

print("test loss:",
      'root MSE: {:.4f}'.format(test_loss / count),
      'time: {:.4f}s'.format(time.time() - t))
plt.semilogy(range(epochs),losses)
plt.title(f"model: MLP, hidden dims :{n_hidden_1,n_hidden_2,n_hidden_3}")
plt.scatter([epochs],[test_loss / count], label = "test loss", c="r")
plt.xlabel("number of epochs")
plt.ylabel("root MSE")
plt.legend()
plt.show()