import torch
import torch.nn as nn
from network.Encoder import Encoder
from network.Layers import FeatureEmbeddings, FeatureEmbeddings_single,\
    FeatureEmbeddingsGroup, HiddenUnit, AttentionReplacement
import torch.nn.functional as F


class ModelManager:
    def __init__(self):
        self.model_map = {'Transformer': Transformer, 'TransformerGroup': TransformerGroup,
                          'FeatureEmbedMLP': FeatureEmbedMLP, 'MLP': MLP,
                          'DeepMLP': DeepMLP, 'ConvModel': ConvModel}

    def get_model(self, name: str):
        return self.model_map[name]


class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.feature_embedding = FeatureEmbeddings(params['EmbeddingDim'], params['EmbeddingDim']//2,
                                                   params['num_features'])
        self.encoder = Encoder(params['EmbeddingDim'], params['EncoderStack'],
                               params['AttentionHead'], params['Dropout'])
        # self.encoder = AttentionReplacement(params['EmbeddingDim'], params['num_features'])
        self.dropout = nn.Dropout(params['Dropout'])
        self.linear_pre = nn.Linear(params['EmbeddingDim']*params['num_features'], params['HD-1'])
        self.linear_final = nn.Linear(params['HD-1'], params['OutDim'])

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
    def __init__(self, params):
        super().__init__()

        self.feature_embedding = FeatureEmbeddingsGroup(params['EmbeddingDim'], params['EmbeddingDim']//2,
                                                        params['FeatureSet'])
        self.encoder = Encoder(params['EmbeddingDim'], params['EncoderStack'],
                               params['AttentionHead'], params['Dropout'])
        self.dropout = nn.Dropout(params['Dropout'])
        self.linear_pre = nn.Linear(params['EmbeddingDim']*8, params['HD-1'])
        self.linear_final = nn.Linear(params['HD-1'], params['OutDim'])

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
    def __init__(self, params):
        super().__init__()
        self.feature_embedding = FeatureEmbeddings_single(params['EmbeddingDim'], params['EmbeddingDim']//2,
                                                          params['num_features'])
        self.dropout = nn.Dropout(params['Dropout'])
        self.linear_pre = nn.Linear(params['EmbeddingDim']*params['num_features'], params['HD-1'])
        self.linear_final = nn.Linear(params['HD-1'], params['OutDim'])

    def forward(self, features):

        x = F.relu(self.feature_embedding(features))
        x = x.view(x.size(0), -1)
        x = self.linear_pre(x)
        x = self.dropout(F.relu(x))
        x = self.linear_final(x)
        return x


class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.linear_1 = nn.Linear(params['num_features'], params['HD-1'])
        self.dropout_1 = nn.Dropout(params['Dropout'])
        self.linear_2 = nn.Linear(params['HD-1'], params['HD-1'])
        self.dropout_2 = nn.Dropout(params['Dropout'])
        self.linear_3 = nn.Linear(params['HD-1'], params['HD-2'])
        self.dropout_3 = nn.Dropout(params['Dropout'])
        self.linear_4 = nn.Linear(params['HD-2'], params['OutDim'])

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
    def __init__(self, params):
        super().__init__()
        self.N = params['Stack']
        print("Hidden UNIT", self.N)
        self.linear_1 = nn.Linear(params['num_features'], params['HD-1'])
        self.dropout_1 = nn.Dropout(params['Dropout'])
        self.linear_2 = nn.Linear(params['HD-1'], params['HD-1'])
        self.dropout_2 = nn.Dropout(params['Dropout'])
        self.hiddenunits = nn.ModuleList([HiddenUnit(params['HD-1'], params['Dropout']) for _ in range(self.N)])
        self.linear_3 = nn.Linear(params['HD-1'], params['HD-2'])
        self.dropout_3 = nn.Dropout(params['Dropout'])
        self.linear_4 = nn.Linear(params['HD-2'], params['OutDim'])

    def forward(self, features):

        x = F.relu(self.linear_1(features))
        x = self.dropout_1(x)
        x = F.relu(self.linear_2(x))
        x = self.dropout_2(x)
        for i in range(self.N):
            x = self.hiddenunits[i](x)
        x = F.relu(self.linear_3(x))
        x = self.dropout_3(x)
        x = self.linear_4(x)
        return x


class ConvModel(nn.Module):
    def __init__(self, params):
        super(ConvModel, self).__init__()
        # input [B, 2, 18]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=params['Conv1'], kernel_size=3, padding=1)
        # [B, C1, 16]
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1)
        # [B, C1, 5]    (WARNING last column of activations in previous layer are ignored b/c of kernel alignment)
        self.conv2 = nn.Conv1d(params['Conv1'], params['Conv2'], kernel_size=3, padding=1)
        # [B, C2, 3]
        self.fc1 = nn.Linear(23968, params['HD-1'])
        # [B, F1]
        self.fc2 = nn.Linear(params['HD-1'], params['HD-2'])
        # [B, F2]
        self.fc3 = nn.Linear(params['HD-2'], params['OutDim'])
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
    mm = ModelManager()
    model = mm.get_model('ConvModel')
    f = model(16, 32, 1024, 512)
    k = f(dummy_input)
    print(k)