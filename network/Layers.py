import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_model*32, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    # def forward(self, x, mask):
    #     x2 = self.norm_1(x)
    #     x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
    #     x2 = self.norm_2(x)
    #     x = x + self.dropout_2(self.ff(x2))
    #     return x

    def forward(self, x, mask):
        x1 = x + self.dropout_1(self.attn(x, x, x, mask))
        x2 = self.norm_1(x1)
        x3 = x2 + self.dropout_2(self.ff(x2))
        return self.norm_2(x3)


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def _attention(self, q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = self._attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class FeatureEmbedder(nn.Module):
    def __init__(self, d_model, hidden, input):
        super().__init__()
        self.linear_1 = nn.Linear(input, hidden)
        self.linear_2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.dropout(x)
        return x


class FeatureEmbeddings(nn.Module):
    def __init__(self, d_model, hidden, feature_length=700):
        super().__init__()
        self.feature_length = feature_length
        self.feature_layers = self._get_clones(FeatureEmbedder(d_model, hidden, 1), feature_length)

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, x):
        features = []
        for i in range(self.feature_length):
            features.append(self.feature_layers[i](x[:, i].unsqueeze(dim=1)))
        return torch.stack(features, dim=1)


class FeatureEmbeddingsGroup(nn.Module):
    def __init__(self, d_model, hidden, feature_set):
        super().__init__()
        self.feature_length = feature_set
        self.feature_layers = nn.ModuleList([FeatureEmbedder(d_model, hidden, fsize) for fsize in feature_set])

    def forward(self, x):
        features = []
        for i, size in enumerate(self.feature_length):
            features.append(self.feature_layers[i](x[:, i, :size]))
        return torch.stack(features, dim=1)


class FeatureEmbeddings_single(nn.Module):
    def __init__(self, d_model, hidden, feature_length=700):
        super().__init__()
        self.feature_length = feature_length
        self.feature_layers = FeatureEmbedder(d_model, hidden, 1)

    # def _get_clones(self, module, N):
    #     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, x):
        features = []
        for i in range(self.feature_length):
            features.append(self.feature_layers(x[:, i].unsqueeze(dim=1)))
        return torch.stack(features, dim=1)


class HiddenUnit(nn.Module):
    def __init__(self, hidden, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        return x


class AttentionReplacement(nn.Module):

    def __init__(self, d_model, features=32):
        super().__init__()
        self.features = features
        self.feature_layers = self._get_clones(HiddenUnit(d_model), features)

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, x):
        output = []
        for i in range(self.features):
            output.append(self.feature_layers[i](x[:, i, :]))
        return torch.stack(output, dim=1)



if __name__ == '__main__':

    dummy_input = torch.randn(5, 700)
    f = FeatureEmbeddingsGroup(64, 32, feature_set=[21, 3, 4, 4, 22, 84, 182, 432])
    E = EncoderLayer(25, 1)
    k = f(dummy_input)
    k = E(k, None)

    print()