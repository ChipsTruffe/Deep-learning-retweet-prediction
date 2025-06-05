import torch
import torch.nn as nn

def init_weights(module):
    for param in module.parameters():
        if param.dim() > 1:
            nn.init.xavier_normal_(param)
        else:
            nn.init.zeros_(param)

class TextLSTMEncoder(nn.Module):
    """
    Encodeur Texte : Embedding → LSTM → Pooling (mean / max / last).
    Renvoie un vecteur de taille (B, hidden_size * num_directions).
    """
    def __init__(
        self,
        vocab_size,
        pad_idx,
        emb_dim=300,
        lstm_hid=512,
        lstm_layers=1,
        bidirectional=True,
        dropout_emb=0.4,
        dropout_lstm=0.4,
        pooling="mean"  # "mean", "max" ou "last"
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.lstm_hid = lstm_hid

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout_emb)

        self.lstm = nn.LSTM(
            emb_dim,
            lstm_hid,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_lstm if lstm_layers > 1 else 0.0
        )

        init_weights(self)

    def forward(self, tokens, lengths):
        emb = self.emb_dropout(self.embedding(tokens))  # (B, L, emb_dim)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_p, (h_n, _) = self.lstm(packed)

        #def du pooling

        if self.pooling == "last":
            if self.bidirectional:
                fwd = h_n[-2]   # (B, hid)
                bwd = h_n[-1]   # (B, hid)
                return torch.cat([fwd, bwd], dim=1)  # (B, 2*hid)
            return h_n[-1]  # (B, hid)

        out_unp, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True)
        mask = (tokens != self.pad_idx).unsqueeze(-1)  # (B, L, 1)

        if self.pooling == "max":
            out_unp = out_unp.masked_fill(~mask, float("-inf"))
            return out_unp.max(dim=1)[0]  # (B, num_dir*hid)

        # pooling == "mean"
        summed = (out_unp * mask.float()).sum(dim=1)  # (B, num_dir*hid)
        lengths = lengths.unsqueeze(1).float()        # (B,1)
        return summed / lengths                       # (B, num_dir*hid)


class NumericMLPEncoder(nn.Module):
    """
    MLP pour features numériques. Renvoie (B, 64).
    """
    def __init__(self, num_feat_dim=4, hidden_dim=64, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        init_weights(self)

    def forward(self, x):
        return self.net(x)  # (B, 64)


class FusionHead(nn.Module):
    """
    Fusion : concat texte + numérique → MLP → prédiction scalaire.
    """
    def __init__(self, text_dim, num_dim=64, hidden_dim=64, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim + num_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        init_weights(self)

    def forward(self, text_repr, num_repr):
        x = torch.cat([text_repr, num_repr], dim=1)
        return self.net(x).squeeze(1)  # (B,)


class LSTM_MLP_Fusion(nn.Module):
    """
    Modèle global : TextLSTMEncoder + NumericMLPEncoder + FusionHead
    """
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
        pooling="mean"
    ):
        super().__init__()
        self.text_enc = TextLSTMEncoder(
            vocab_size, pad_idx,
            emb_dim=emb_dim,
            lstm_hid=lstm_hid,
            lstm_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout_emb=dropout_emb,
            dropout_lstm=dropout_lstm,
            pooling=pooling
        )
        self.num_enc = NumericMLPEncoder(
            num_feat_dim=num_feat_dim,
            hidden_dim=64,
            dropout=dropout_num
        )
        text_dim = lstm_hid * (2 if bidirectional else 1)
        self.fusion = FusionHead(
            text_dim=text_dim,
            num_dim=64,
            hidden_dim=64,
            dropout=dropout_head
        )

    def forward(self, tokens, lengths, numerics):
        t_repr = self.text_enc(tokens, lengths)   # (B, text_dim)
        n_repr = self.num_enc(numerics)           # (B, 64)
        return self.fusion(t_repr, n_repr)        # (B,)
