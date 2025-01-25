from torch import nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        mean = torch.mean(x, -1, keepdim=True)
        std = torch.std(x, -1, keepdim=True)
        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out
