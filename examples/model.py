"""Minimal Transformer for MNIST classification.

Architecture:
  - Treat each row of 28 pixels as one token → sequence of 28 tokens
  - Prepend a learnable [CLS] token → sequence length 29
  - 2× TransformerEncoderLayer (d_model=128, nhead=4, dim_ff=256)
  - Classify from CLS token output
"""
import torch
import torch.nn as nn


class MNISTTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(28, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 29, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 10)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = x.squeeze(1)                        # (B, 28, 28)
        x = self.embed(x)                       # (B, 28, d_model)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)          # (B, 29, d_model)
        x = x + self.pos_embed
        x = self.encoder(x)                     # (B, 29, d_model)
        return self.classifier(x[:, 0])         # (B, 10)
