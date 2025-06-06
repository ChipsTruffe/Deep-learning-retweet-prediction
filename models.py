import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
import torch
import utils


class myDistilBert(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", hidden_dim=32, output_dim=2):
        super(myDistilBert, self).__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze BERT layers
        for param in self.bert.parameters():
            param.requires_grad = False

        # Custom regression/classification head
        self.head = nn.Sequential(
            nn.Linear(768, hidden_dim), #BERT outputs in dim 768 
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), 
            nn.Softplus()  # Ensure output is strictly positive
        )

        utils.randomize_model_weights(self.head)

    def forward(self, input_ids, attention_mask=None):
        #print("[DEBUG]", input_ids.device)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  
        positive_int_output = self.head(pooled_output)
        return positive_int_output  

def MLP(input_dim, hidden_dims, output_dim):
    """
    Creates a Multi-Layer Perceptron (MLP).

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dims (list of int): List containing the sizes of hidden layers.
        output_dim (int): Dimension of the output.

    Returns:
        nn.Sequential: The MLP model.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dims[0]),
        nn.ReLU()
    )
    for i in range(1, len(hidden_dims)):
        model.add_module(
            "linear_" + str(i), 
            nn.Linear(hidden_dims[i-1], hidden_dims[i])
        )
        model.add_module("relu_" + str(i), nn.ReLU())
    model.add_module("output", nn.Linear(hidden_dims[-1], output_dim))

    for module in model.modules(): ##tous les poids à 0 par défaut
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, 0)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0)

    return model


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens 
    in the sequence. The positional encodings have the same dimension as 
    the embeddings so that the two can be summed.
    """
    def __init__(self, nhid, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, nhid)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, nhid, 2).float() * (-math.log(10000.0) / nhid)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, nhid)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """
    Container module with an encoder, a recurrent or transformer module, and a decoder.
    """
    def __init__(self, ntoken, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        '''
        Args:
            ntoken: the size of vocabulary
            nhid: the hidden dimension of the model. We assume that embedding_dim = nhid
            nlayers: the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            nhead: the number of heads in the multiheadattention models
            dropout: the dropout value
        '''
        self.model_type = "Transformer"
        self.encoder = nn.Embedding(ntoken, nhid) # Embedding layer
        self.pos_encoder = PositionalEncoding(nhid, dropout) # Positional encoding
        
        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=nhid, 
            nhead=nhead, 
            dim_feedforward=nhid, # Often 2*nhid or 4*nhid, but can be nhid
            dropout=dropout,
            batch_first=False # Expects input as (seq_len, batch_size, features)
        )
        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers, 
            num_layers=nlayers
        )
        self.nhid = nhid
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        """
        Generates a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        Useful for auto-regressive tasks.
        Args:
            sz: size of the sequence
        Returns:
            torch.Tensor: a square mask of shape (sz, sz)
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
    def forward(self, src, src_mask=None, src_key_padding_mask=None): 
        src = self.encoder(src) * math.sqrt(self.nhid)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class myLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, 
                 bidirectional=False, padding_idx=None):
        """
        LSTM Text Processing Model
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: LSTM hidden state dimension
            output_dim: Dimension of output vector
            num_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            padding_idx: Padding index for embedding layer
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # LSTM configuration
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        
        # Create LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output dimension after LSTM
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Final projection layer
        self.fc = nn.Linear(lstm_output_dim, output_dim)

    def forward(self, input):
        """
        Forward pass
        Args:
            input_ids: (batch_size, seq_len) tensor of token ids
            padding_mask: (batch_size, seq_len) mask (1 = real token, 0 = pad)
        Returns:
            output: (batch_size, output_dim) tensor
        """
        input_ids, padding_mask = input[0], input[1]

        
        #get device from input_ids
        device = input_ids.device

        # Calculate sequence lengths from mask
        lengths = len(padding_mask[0]) - padding_mask.sum(dim=1)

        # Sort sequences by length (descending)
        lengths_sorted, sort_idx = torch.sort(lengths, descending=True)
        input_ids_sorted = input_ids[sort_idx]

        # Embed tokens
        embedded = self.embedding(input_ids_sorted)

        # Convert lengths to CPU for pack_padded_sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths_sorted.cpu(),  # Must be on CPU
            batch_first=True,
            enforce_sorted=True
        )
        # Process with LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Extract final hidden states
        if self.bidirectional or self.num_layers > 1:
            # Extract last layer hidden states
            if self.bidirectional:
                # Separate directions and layers
                hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
                # Take last layer's forward and backward states
                h_forward = hidden[-1, 0]  # (batch, hidden_dim)
                h_backward = hidden[-1, 1]  # (batch, hidden_dim)
                # Concatenate both directions
                h_final = torch.cat((h_forward, h_backward), dim=1)
            else:
                # Take last layer's hidden state
                h_final = hidden[-1]  # (batch, hidden_dim)
        else:
            # Single layer, single direction
            h_final = hidden.squeeze(0)

        # Project to output dimension
        output_sorted = self.fc(h_final)

        # Restore original batch order
        _, unsort_idx = torch.sort(sort_idx)
        output = output_sorted[unsort_idx]

        return output
class CombinedModelWithMLP(nn.Module):
    """
    Model that processes a sentence with a Transformer, gets a representation,
    concatenates it with a numerical vector, and processes the result through an MLP
    to get a single output value.
    """
    def __init__(self, ntoken, nhead, nhid, nlayers, 
                 numerical_input_dim, mlp_hidden_dims, dropout=0.5):
        """
        Args:
            ntoken (int): Vocabulary size for the text input.
            nhead (int): Number of heads in the Transformer's multi-head attention.
            nhid (int): Hidden dimension size for the Transformer and its embeddings.
            nlayers (int): Number of layers in the Transformer encoder.
            numerical_input_dim (int): Dimension of the auxiliary numerical input vector.
            mlp_hidden_dims (list of int): List of hidden layer sizes for the MLP.
            dropout (float): Dropout rate.
        """
        super(CombinedModelWithMLP, self).__init__()
        
        # Transformer base model for text processing
        self.transformer_base = TransformerModel(ntoken, nhead, nhid, nlayers, dropout)
        
        # Define the input dimension for the MLP:
        # It's the Transformer's hidden dimension + the numerical vector's dimension
        mlp_input_dim = nhid + numerical_input_dim
        
        # MLP head to process combined features and produce a single output
        self.mlp_head = MLP(input_dim=mlp_input_dim, 
                            hidden_dims=mlp_hidden_dims, 
                            output_dim=1) # Output a single value

    def forward(self, text_src_ids, num_src, src_mask=None, src_key_padding_mask=None):

        # 1. Process text input through Transformer
        # transformer_output shape: (seq_len, batch_size, nhid)
        transformer_output = self.transformer_base(text_src_ids, src_mask = src_mask,src_key_padding_mask=src_key_padding_mask) 
        
        # 2. Extract features from Transformer output.
        # A common strategy for sentence representation is to take the output 
        # of the first token (e.g., like a [CLS] token).
        # text_features shape: (batch_size, nhid)
        text_features = transformer_output[0, :, :] # Assumes first token output is representative

        # 3. Ensure num_src is correctly shaped (batch_size, numerical_input_dim).
        # (No explicit reshaping here, assuming input `num_src` is already correct)

        # 4. Concatenate text features and numerical vector features.
        # combined_features shape: (batch_size, nhid + numerical_input_dim)
        combined_features = torch.cat((text_features, num_src), dim=1)
        
        # 5. Pass combined features through MLP.
        # mlp_output shape: (batch_size, 1)
        output = self.mlp_head(combined_features)
        
        return output

class finalModel(nn.Module):
    
    def __init__(self, TextInterpreter, Regressor, *args, **kwargs ):
        """
        Combines a text model (ex : LSTM) with an interpretor(ex : MLP).
        Both should have a forward() method taking a vector in (vectorized text or latent representation) and outputting a vector.
        The regressor should output a int representing the expected number of retweets.
        Both model should have compatible sizes, i.e the output of text interpretor concatenated to the value vector should have the input dimention
        of the regressor
        
        """
        super().__init__(*args, **kwargs)
        self.TextInterpreter = TextInterpreter
        self.Regressor = Regressor
    
    def forward(self, text_vector, value_vector):
        latent_text = self.TextInterpreter(text_vector)
        latent_x = torch.cat((latent_text,value_vector), dim = 1) ##CHECK LA DIMENSION DE CONCATENATION
        output = self.Regressor(latent_x)
        return output



