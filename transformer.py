import torch
import torch.nn as nn
from encoding import PositionalEncoding

class AviationTransformer(nn.Module):
    def __init__(
        self, 
        num_features,
        num_meta_features,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=100
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder using PyTorch built-in
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Important: input shape (batch, seq, feature)
            norm_first=False   # Post-norm (standard); set True for pre-norm
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Metadata processing
        self.meta_fc = nn.Sequential(
            nn.Linear(num_meta_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, seq_input, meta_input):
        # seq_input: (batch, seq_len, num_features)
        # meta_input: (batch, num_meta_features)
        
        # Project sequence to d_model
        x = self.input_projection(seq_input)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Global average pooling
        seq_features = torch.mean(x, dim=1)  # (batch, d_model)
        
        # Process metadata
        meta_features = self.meta_fc(meta_input)  # (batch, 64)
        
        # Concatenate sequence and metadata features
        combined = torch.cat([seq_features, meta_features], dim=1)
        
        # Final classification
        logits = self.classifier(combined)
        return logits