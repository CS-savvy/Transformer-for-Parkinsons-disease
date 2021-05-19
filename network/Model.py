import torch
import torch.nn as nn
from network.Encoder import Encoder
from network.Layers import FeatureEmbeddings, FeatureEmbeddings_single, FeatureEmbeddingsGroup
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, dropout=0.1, feature_length=700):
        super().__init__()

        self.feature_embedding = FeatureEmbeddings(d_model, d_model//2, feature_length)
        self.encoder = Encoder(d_model, N, heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear_pre = nn.Linear(d_model*feature_length, 2048)
        self.linear_final = nn.Linear(2048, 1)

    def forward(self, features):

        x = self.feature_embedding(features)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        # x = F.adaptive_avg_pool2d(x, output_size=1)
        # x = x.view(x.size(0), -1)
        x = self.linear_pre(x)
        x = self.dropout(F.relu(x))
        x = self.linear_final(x)
        return x


class TransformerGroup(nn.Module):
    def __init__(self, d_model, N, heads, dropout=0.1, feature_set=[]):
        super().__init__()

        self.feature_embedding = FeatureEmbeddingsGroup(d_model, d_model//2, feature_set)
        self.encoder = Encoder(d_model, N, heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear_pre = nn.Linear(d_model*8, 2048)
        self.linear_final = nn.Linear(2048, 1)

    def forward(self, features):

        x = self.feature_embedding(features)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        # x = F.adaptive_avg_pool2d(x, output_size=1)
        # x = x.view(x.size(0), -1)
        x = self.linear_pre(x)
        x = self.dropout(F.relu(x))
        x = self.linear_final(x)
        return x


class FeatureEmbedMLP(nn.Module):
    def __init__(self, d_model, N, heads, dropout=0.1, feature_length=700):
        super().__init__()

        self.feature_embedding = FeatureEmbeddings_single(d_model, d_model//2, feature_length)
        # self.encoder = Encoder(d_model, N, heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear_pre = nn.Linear(d_model*feature_length, 2048)
        self.linear_final = nn.Linear(2048, 1)

    def forward(self, features):

        x = F.relu(self.feature_embedding(features))
        x = x.view(x.size(0), -1)
        x = self.linear_pre(x)
        x = self.dropout(F.relu(x))
        x = self.linear_final(x)
        return x


class MLP(nn.Module):
    def __init__(self, d_model, N, heads, dropout=0.1, feature_length=700):
        super().__init__()

        self.linear_1 = nn.Linear(feature_length, 2048)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(2048, 2048)
        self.dropout_2 = nn.Dropout(dropout)
        self.linear_3 = nn.Linear(2048, 512)
        self.dropout_3 = nn.Dropout(dropout)
        self.linear_4 = nn.Linear(512, 1)

    def forward(self, features):

        x = F.relu(self.linear_1(features))
        x = self.dropout_1(x)
        x = F.relu(self.linear_2(x))
        x = self.dropout_2(x)
        x = F.relu(self.linear_3(x))
        x = self.dropout_3(x)
        x = F.relu(self.linear_4(x))

        return x


if __name__ == '__main__':
    dummy_input = torch.randn(5, 700)
    f = Transformer(64, 6, 1)
    k = f(dummy_input)
    print()