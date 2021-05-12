import torch.nn as nn
from network.Layers import EncoderLayer
from network.Layers import Norm
import copy


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = self._get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, src, mask=None):
        for i in range(self.N):
            x = self.layers[i](src, mask)
        return self.norm(x)
