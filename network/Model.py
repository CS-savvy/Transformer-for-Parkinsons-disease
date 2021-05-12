import torch
import torch.nn as nn
from network.Encoder import Encoder
from network.Layers import FeatureEmbeddings
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, dropout=0.1, feature_length=700):
        super().__init__()

        self.feature_embedding = FeatureEmbeddings(d_model, d_model//2, feature_length)
        self.encoder = Encoder(d_model, N, heads, dropout)
        self.linear = nn.Linear(d_model*feature_length, 1)

    def forward(self, features):

        x = self.feature_embedding(features)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = F.sigmoid(self.linear(x))
        return x


if __name__ == '__main__':
    dummy_input = torch.randn(5, 700)
    f = Transformer(64, 6, 1)
    k = f(dummy_input)
    print()