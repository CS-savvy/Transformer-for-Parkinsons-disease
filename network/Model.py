import torch
import torch.nn as nn
from Encoder import Encoder
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()

        self.Encoder = Encoder(d_model, N, heads, dropout)
        