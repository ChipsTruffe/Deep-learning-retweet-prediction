import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(hidden_dims[i-1], hidden_dims[i]) # Removed trailing comma
        )
        model.add_module("relu_" + str(i), nn.ReLU())
    model.add_module("output", nn.Linear(hidden_dims[-1], output_dim))
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

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len] (optional)
                      Mask for the source sequence.
        
        Returns:
            torch.Tensor: Output of the transformer encoder, shape [seq_len, batch_size, nhid]
        """
        src = self.encoder(src) * math.sqrt(self.nhid) # Embed and scale
        src = self.pos_encoder(src) # Add positional encoding
        output = self.transformer_encoder(src, src_mask) # Pass through transformer
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

    def forward(self, text_src, num_src, src_mask=None):
        """
        Forward pass of the model.

        Args:
            text_src (torch.Tensor): The input sentence tensor. 
                                     Shape: (seq_len, batch_size)
            num_src (torch.Tensor): The numerical input vector.
                                    Shape: (batch_size, numerical_input_dim)
            src_mask (torch.Tensor, optional): Mask for the `text_src` input to the Transformer.
                                               Shape: (seq_len, seq_len) for a subsequent mask,
                                               or (batch_size, seq_len) for a padding mask (if batch_first=True in Transformer)
                                               Note: TransformerModel expects (seq_len, seq_len) or specific key padding mask.

        Returns:
            torch.Tensor: The final single output value from the MLP.
                          Shape: (batch_size, 1)
        """
        # 1. Process text input through Transformer
        # transformer_output shape: (seq_len, batch_size, nhid)
        transformer_output = self.transformer_base(text_src, src_mask) 
        
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

if __name__ == '__main__':
    # Example Usage (Illustrative)
    
    # Parameters
    vocab_size = 1000  # Size of the vocabulary
    num_heads = 4      # Number of attention heads in Transformer
    hidden_dim = 128   # Hidden dimension (nhid)
    num_layers = 3     # Number of Transformer encoder layers
    dropout_rate = 0.1

    numerical_vec_dim = 10 # Dimension of the numerical input vector
    mlp_layers = [64, 32]  # Hidden layers for the MLP
    
    batch_size = 8
    seq_length = 20

    # Instantiate the model
    model = CombinedModelWithMLP(
        ntoken=vocab_size,
        nhead=num_heads,
        nhid=hidden_dim,
        nlayers=num_layers,
        numerical_input_dim=numerical_vec_dim,
        mlp_hidden_dims=mlp_layers,
        dropout=dropout_rate
    )

    # Create dummy input data
    # Text input (batch_size, seq_length) -> (seq_length, batch_size) for Transformer
    dummy_text_input = torch.randint(0, vocab_size, (seq_length, batch_size))
    
    # Numerical input vector (batch_size, numerical_vec_dim)
    dummy_numerical_input = torch.randn(batch_size, numerical_vec_dim)
    
    # Optional: Source mask (e.g., for preventing attention to future tokens)
    # For this example, we might not need a complex mask if just taking first token output,
    # but if the transformer part is pre-trained or used for other purposes, it might be.
    # For simplicity, let's pass None or a basic mask.
    # A square subsequent mask:
    # src_mask = model.transformer_base.generate_square_subsequent_mask(seq_length) 
    # src_mask = src_mask.to(dummy_text_input.device) # Move mask to the same device as input
    # Or, if no specific masking is needed for this type of feature extraction:
    src_mask = None

    # Forward pass
    try:
        output = model(dummy_text_input, dummy_numerical_input, src_mask)
        print("Model instantiated successfully.")
        print("Input text shape:", dummy_text_input.shape)
        print("Input numerical vector shape:", dummy_numerical_input.shape)
        print("Output shape:", output.shape) # Expected: (batch_size, 1)
        print("Example output:", output)
    except Exception as e:
        print(f"Error during model usage: {e}")

    # Example of using the original TransformerModel and ScoringHead (if needed separately)
    # transformer_only = TransformerModel(vocab_size, num_heads, hidden_dim, num_layers, dropout_rate)
    # transformer_out = transformer_only(dummy_text_input, src_mask) # (seq_len, batch_size, nhid)
    # features_for_scoring = transformer_out[0, :, :] # (batch_size, nhid)
    # scoring_head = ScoringHead(hidden_dim)
    # scores = scoring_head(features_for_scoring) # (batch_size, 1)
    # print("\nSeparate ScoringHead output shape:", scores.shape)
