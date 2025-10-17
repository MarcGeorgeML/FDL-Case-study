import torch
import torch.nn as nn
from encoding import PositionalEncoding

class TransformerBlockWithLayerDrop(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1,
                 layerdrop=0.1, batch_first=True, activation='relu', norm_first=False):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first
        )
        self.layerdrop = layerdrop

    def forward(self, src, *args, **kwargs):
        # Randomly skip this layer during training
        if self.training and torch.rand(1).item() < self.layerdrop:
            return src
        # Pass through the standard TransformerEncoderLayer
        return super().forward(src, *args, **kwargs)


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
        layerdrop=0.1,
        max_seq_len=100
    ):
        super().__init__()

        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Build TransformerEncoder with LayerDrop blocks
        encoder_layer = TransformerBlockWithLayerDrop(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layerdrop=layerdrop,
            batch_first=True,
            activation='relu',
            norm_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Metadata head
        self.meta_fc = nn.Sequential(
            nn.Linear(num_meta_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification head
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
        x = self.input_projection(seq_input)     # → (batch, seq_len, d_model)
        x = self.pos_encoder(x)                  # add positional encodings
        x = self.transformer_encoder(x)          # → (batch, seq_len, d_model)
        seq_features = x.mean(dim=1)             # global average pooling
        meta_features = self.meta_fc(meta_input) # metadata embedding
        combined = torch.cat([seq_features, meta_features], dim=1)
        logits = self.classifier(combined)       # → (batch, 1)
        return logits
