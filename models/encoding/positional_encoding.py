from torch import nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=512, temperature=10000, device='cuda'):
        super().__init__()
        self.device = device

        pos = torch.arange(0, max_len).unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, 2)

        self.pe = torch.zeros(max_len, d_model, device=device)
        self.pe.requires_grad = False

        self.pe[:, 0::2] = torch.sin(pos / (temperature ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (temperature ** (_2i / d_model)))        
        
    def forward(self, x):        
        _, length = x.shape
        return self.pe[:length, :].to(self.device)
