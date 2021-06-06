import torch
import torch.nn as nn
from network.Encoder import Encoder
from network.Layers import FeatureEmbeddings, FeatureEmbeddings_single, FeatureEmbeddingsGroup, HiddenUnit
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
        x = self.linear_4(x)

        return x


class DeepMLP(nn.Module):
    def __init__(self, d_model, N, heads, dropout=0.1, feature_length=700):
        super().__init__()
        self.N = N
        print("Hidden UNIT", self.N)
        self.linear_1 = nn.Linear(feature_length, 2048)
        self.dropout_1 = nn.Dropout(dropout)
        self.hiddenunits = nn.ModuleList([HiddenUnit(2048, dropout) for _ in range(N)])
        self.linear_3 = nn.Linear(2048, 512)
        self.dropout_3 = nn.Dropout(dropout)
        self.linear_4 = nn.Linear(512, 1)

    def forward(self, features):

        x = F.relu(self.linear_1(features))
        x = self.dropout_1(x)
        for i in range(self.N):
            x = self.hiddenunits[i](x)
        x = F.relu(self.linear_3(x))
        x = self.dropout_3(x)
        x = self.linear_4(x)
        return x


class ConvModel(nn.Module):
    def __init__(self, C1, C2, F1, F2):
        super(ConvModel, self).__init__()
        # input [B, 2, 18]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=C1, kernel_size=3, padding=1)
        # [B, C1, 16]
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1)
        # [B, C1, 5]    (WARNING last column of activations in previous layer are ignored b/c of kernel alignment)
        self.conv2 = nn.Conv1d(C1, C2, kernel_size=3, padding=1)
        # [B, C2, 3]
        self.fc1 = nn.Linear(23968, F1)
        # [B, F1]
        self.fc2 = nn.Linear(F1, F2)
        # [B, F2]
        self.fc3 = nn.Linear(F2, 1)
        # [B, 2]

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.flatten(1) # flatten the tensor starting at dimension 1
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



if __name__ == '__main__':
    dummy_input = torch.randn(5, 1, 700)
    f = ConvModel(16, 32, 1024, 512)
    k = f(dummy_input)
    print()