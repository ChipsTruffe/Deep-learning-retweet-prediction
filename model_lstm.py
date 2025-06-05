# model_lstm.py

import torch
import torch.nn as nn

class LSTM_MLP(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_idx,
        emb_dim=300,
        lstm_hid=512,
        num_feat_dim=4,
        lstm_layers=1,
        bidirectional=True,
        dropout_emb=0.4,
        dropout_lstm=0.4,
        dropout_num=0.4,
        dropout_head=0.4,
        use_pooling="mean"  # ou "last" / "max"
        
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm_hid = lstm_hid
        self.num_directions = 2 if bidirectional else 1
        self.use_pooling = use_pooling

        # 1) Embedding + Dropout
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout_emb)

        # 2) LSTM (multi-couches, bidirectionnel)
        self.lstm = nn.LSTM(
            emb_dim,
            lstm_hid,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_lstm if lstm_layers > 1 else 0.0
        )

        # 3) MLP pour variables numériques
        self.num_mlp = nn.Sequential(
            nn.Linear(num_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_num),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout_num),
        )

        # 4) Tête de fusion et sortie
        fusion_dim = self.num_directions * lstm_hid + 64
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_head),
            nn.Linear(64, 1)
        )

        # 5) Initialisation Xavier des poids
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, tokens, lengths, numerics):
        """
        - tokens   : LongTensor (B, L_max)
        - lengths  : LongTensor (B,) avec la longueur réelle de chaque séquence
        - numerics : FloatTensor (B, num_feat_dim)
        """
        # --- Embedding + Dropout ---
        emb = self.embedding(tokens)           # (B, L_max, emb_dim)
        emb = self.emb_dropout(emb)

        # --- Pack + LSTM ---
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_p, (h_n, c_n) = self.lstm(packed)

        # --- Pooling ou dernier hidden state ---
        if self.use_pooling == "last":
            if self.bidirectional:
                hidden_forward = h_n[-2]   # (B, lstm_hid)
                hidden_backward = h_n[-1]  # (B, lstm_hid)
                text_repr = torch.cat([hidden_forward, hidden_backward], dim=1)  # (B, 2*lstm_hid)
            else:
                text_repr = h_n[-1]  # (B, lstm_hid)
        else:
            out_unpacked, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True)
            mask = (tokens != self.embedding.padding_idx).unsqueeze(-1)  # (B, L_max_eff, 1)
            if self.use_pooling == "max":
                out_unpacked_masked = out_unpacked.masked_fill(~mask, float("-inf"))
                text_repr, _ = out_unpacked_masked.max(dim=1)
            elif self.use_pooling == "mean":
                summed = (out_unpacked * mask.float()).sum(dim=1)
                lengths_ = lengths.unsqueeze(1).float()
                text_repr = summed / lengths_
            else:
                raise ValueError(f"Pooling inconnu : {self.use_pooling}")

        # --- MLP pour numériques ---
        num_repr = self.num_mlp(numerics)  # (B, 64)

        # --- Fusion et sortie finale ---
        x = torch.cat([text_repr, num_repr], dim=1)   # (B, fusion_dim)
        out = self.head(x).squeeze(1)                 # (B,)
        return out
