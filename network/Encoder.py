import torch.nn as nn
from Layers import EncoderLayer
from Layers import Norm
import copy


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = self._get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
