import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_prep import build_data_loaders
from model_lstm import LSTM_MLP_Fusion  # On importe la classe Fusion

# -------------------------
# 1) Hyperparamètres
# -------------------------
csv_path = "data/train.csv"
model_save_path = "best_model_fusion.pth"  # Nouveau nom pour le modèle fusion
batch_size = 1024
lr = 1e-2
n_epochs = 20

# Dimensions et options du modèle
emb_dim = 300
lstm_hid = 512
num_feat_dim = 4
lstm_layers = 3
bidirectional = True
dropout_emb = 0.4
dropout_lstm = 0.4
dropout_num = 0.4
dropout_head = 0.4
use_pooling = "mean"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Entraînement sur :", device)

# -------------------------
# 2) Préparation des DataLoaders
# -------------------------
train_loader, valid_loader, test_loader, pad_idx, unk_idx, vocab_size = build_data_loaders(
    csv_path,
    batch_size=batch_size
)

# -------------------------
# 3) Instanciation du modèle Fusion
# -------------------------
model = LSTM_MLP_Fusion(
    vocab_size=vocab_size,
    pad_idx=pad_idx,
    emb_dim=emb_dim,
    lstm_hid=lstm_hid,
    num_feat_dim=num_feat_dim,
    lstm_layers=lstm_layers,
    bidirectional=bidirectional,
    dropout_emb=dropout_emb,
    dropout_lstm=dropout_lstm,
    dropout_num=dropout_num,
    dropout_head=dropout_head,
    pooling=use_pooling
).to(device)

# -------------------------
# 4) Optimiseur et fonction de perte (MAE = L1)
# -------------------------
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.L1Loss()

train_losses = []
val_losses = []
best_val_loss = float("inf")

# -------------------------
# 5) Boucle d'entraînement + validation
# -------------------------
for epoch in range(1, n_epochs + 1):
    # --- Phase entraînement ---
    model.train()
    total_train_loss = 0.0

    for tokens, lengths, numerics, labels in train_loader:
        tokens, lengths = tokens.to(device), lengths.to(device)
        numerics, labels = numerics.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(tokens, lengths, numerics)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * tokens.size(0)

    avg_train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch}/{n_epochs} — Train L1 : {avg_train_loss:.4f}")

    # --- Phase validation ---
    model.eval()
    total_valid_loss = 0.0
    with torch.no_grad():
        for tokens, lengths, numerics, labels in valid_loader:
            tokens, lengths = tokens.to(device), lengths.to(device)
            numerics, labels = numerics.to(device), labels.to(device)

            preds = model(tokens, lengths, numerics)
            total_valid_loss += criterion(preds, labels).item() * tokens.size(0)

    avg_valid_loss = total_valid_loss / len(valid_loader.dataset)
    val_losses.append(avg_valid_loss)
    print(f"Epoch {epoch}/{n_epochs} — Valid L1 : {avg_valid_loss:.4f}")

    # Sauvegarde du meilleur modèle
    if avg_valid_loss < best_val_loss:
        best_val_loss = avg_valid_loss
        torch.save(model.state_dict(), model_save_path)
        print(" ➔ Meilleur modèle sauvegardé !")

# -------------------------
# 6) Tracé des courbes d'erreur (Train / Valid)
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train L1", marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label="Valid L1", marker='x')
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("Evolution de la perte (Train / Valid)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("loss_lstm.png")
plt.show()

# -------------------------
# 7) Évaluation finale sur le test set
# -------------------------
model.load_state_dict(torch.load(model_save_path))
model.to(device)
model.eval()

total_test_loss = 0.0
with torch.no_grad():
    for tokens, lengths, numerics, labels in test_loader:
        tokens, lengths = tokens.to(device), lengths.to(device)
        numerics, labels = numerics.to(device), labels.to(device)

        preds = model(tokens, lengths, numerics)
        total_test_loss += criterion(preds, labels).item() * tokens.size(0)

avg_test_loss = total_test_loss / len(test_loader.dataset)
print(f"Final Test L1 : {avg_test_loss:.4f}")
