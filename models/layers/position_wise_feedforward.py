from torch import nn
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden):
        super().__init__()
        self.W1 = nn.Linear(d_model, ffn_hidden)
        self.W2 = nn.Linear(ffn_hidden, d_model)
        
    def forward(self, x):
        x = self.W1(x)
        x = F.relu(x)
        x = self.W2(x)
        return x